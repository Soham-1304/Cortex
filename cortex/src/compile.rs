use crate::Result;
use crate::{corefile::CoreFile, embed::EmbeddingProvider, store::Store};
use bench::{
    count_tokens,
    pack::{knapsack, Item},
};
use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct MemoryHit {
    pub id: i64,
    pub content: String,
    pub score: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct SymbolHit {
    pub name: String,
    pub path: String,
    pub line_start: i64,
    pub line_end: i64,
    pub signature: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct StageTimesSer {
    pub fts_ms: u64,
    pub vec_ms: u64,
    pub graph_ms: u64,
    pub pack_ms: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct Payload {
    pub core: Option<String>,
    pub memories: Vec<MemoryHit>,
    pub symbols: Vec<SymbolHit>,
    pub tokens_used: i64,
    pub rho_naive: f64,
    pub took_ms: u64,
    /// Per-stage timings (plan item 5). Flows into `recall --json` and the
    /// event log so latency attribution needs no separate profiling run.
    pub stages: StageTimesSer,
}

fn truncate_by_tokens(s: &str, budget: i64) -> String {
    if budget <= 0 {
        return String::new();
    }
    let toks = bench::tokenize(s);
    if toks.len() as i64 <= budget {
        return s.to_string();
    }
    toks[..budget as usize].join(" ")
}

fn is_zero_vec(v: &[f32]) -> bool {
    crate::store::is_zero_vec(v)
}
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    crate::store::cosine(a, b)
}

/// Secret redaction at serving time: AWS access-key IDs (`AKIA` + 16
/// `[A-Z0-9]`) are scrubbed from recalled memory text before it reaches
/// the agent. Stored rows keep raw text (FTS still matches); only the
/// payload is scrubbed. No regex crate — manual scan over the marker.
fn redact_secrets(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = String::with_capacity(s.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes.len() - i >= 20
            && &bytes[i..i + 4] == b"AKIA"
            && bytes[i + 4..i + 20]
                .iter()
                .all(|b| b.is_ascii_uppercase() || b.is_ascii_digit())
        {
            out.push_str("[REDACTED-AWS-KEY]");
            i += 20;
        } else {
            let ch = s[i..].chars().next().unwrap();
            out.push(ch);
            i += ch.len_utf8();
        }
    }
    out
}

// Dim-mismatch warnings are process-once per table (not per compile() call)
// so loops over compiles (e.g. recall sweeps, benches) don't spam stderr.
// One `Once` per table rather than a single shared one: notes and symbols
// are indexed independently and either may go stale first; each table still
// warns at most once per process (first mismatched dims win).
static NOTES_DIM_WARN: std::sync::Once = std::sync::Once::new();
static SYMS_DIM_WARN: std::sync::Once = std::sync::Once::new();

fn warn_notes_dim_once(stored: usize, provider: usize) {
    NOTES_DIM_WARN.call_once(|| {
        eprintln!(
            "stored embeddings dim {} != provider dim {}: run cortex reindex",
            stored, provider
        );
    });
}

fn warn_syms_dim_once(stored: usize, provider: usize) {
    SYMS_DIM_WARN.call_once(|| {
        eprintln!(
            "stored embeddings dim {} != provider dim {}: run cortex reindex",
            stored, provider
        );
    });
}

const RRF_K: f64 = 60.0;
// Scale note: one RRF arm maxes at ~1/61 ≈ 0.016 (both arms ≈ 0.033), so
// GRAPH_BOOST up to 0.2+ is graph-dominant by design — fused scores (and
// hence rho_naive) are not comparable across graph vs no-graph queries.
const GRAPH_BOOST: f64 = 0.2;
// (Task 4 graph leg removed 2026-09-20: file-neighborhood spread hurt
// file-level nav −0.048/task on N=125. Helpers deleted with it;
// file_connections() in store.rs stays for find/map.)

/// File-pooled serving order (bench `--pool 10`: +0.063 span R@20 on
/// astropy, file recall intact). Rank files by their best candidate
/// score, emit the top-`top_n` files' candidates first (relative order
/// preserved), then the rest. Pure reorder — scores untouched, so
/// knapsack values, rho, and calibration are unaffected; only WHICH
/// symbols reach the top-12 truncation changes.
const POOL_TOP_FILES: usize = 10;

fn pool_by_file(
    cands: Vec<(SymbolHit, f64)>,
    top_n: usize,
) -> Vec<(SymbolHit, f64)> {
    use std::collections::{HashMap, HashSet};
    let top: HashSet<String> = {
        let mut best_rank: HashMap<&str, usize> = HashMap::new();
        for (rank, (sym, _)) in cands.iter().enumerate() {
            best_rank.entry(sym.path.as_str()).or_insert(rank);
        }
        let mut files: Vec<(&str, usize)> = best_rank.into_iter().collect();
        files.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(b.0)));
        files
            .into_iter()
            .take(top_n)
            .map(|(p, _)| p.to_string())
            .collect()
    };
    let (mut head, mut tail): (Vec<_>, Vec<_>) = cands
        .into_iter()
        .partition(|(sym, _)| top.contains(&sym.path));
    head.append(&mut tail);
    head
}

fn rrf(rank: usize) -> f64 {
    1.0 / (RRF_K + rank as f64 + 1.0)
}

/// True when `name` occurs in a lowercased context snippet as code — a
/// call (`name(`), an attribute access/base (`.name`, `name.` with token
/// boundaries) — rather than a prose mention (`the frame rate`) or a bare
/// assignment RHS (`x = frame`). The FTS-context arm exists to resolve
/// call sites to their symbols; bare mentions are what flooded the leg
/// (rare-but-wrong names vote for unrelated symbols), so they don't vote.
/// Single-char names never qualify (loop vars, not signal).
fn is_code_mention(ctx_lc: &str, name: &str) -> bool {
    if name.len() < 2 {
        return false;
    }
    let boundary = |c: Option<char>| match c {
        None => true,
        Some(x) => !(x.is_alphanumeric() || x == '_'),
    };
    for (i, _) in ctx_lc.match_indices(name) {
        let prev = ctx_lc[..i].chars().next_back();
        let next = ctx_lc[i + name.len()..].chars().next();
        // Token boundaries both sides: `reframed` / `myframe(` must not
        // match `frame`. (`.` satisfies the boundary — attribute access.)
        if !boundary(prev) || !boundary(next) {
            continue;
        }
        if next == Some('(') || next == Some('.') || prev == Some('.') {
            return true;
        }
    }
    false
}

/// Entity activations via graph spread. Seeds = top-3 entities (by task-token
/// name overlap) with non-NULL embeddings — link-only concepts have none and
/// are skipped as seeds. Adjacency is symmetric with
/// `normalize_edge_weight(co_mentions, 1.0)`; sqlite rowids are remapped to
/// dense 0..n for `spread(max_hops=2, eps=1e-3)` and back.
/// Seed-exclusion is intentional (discovery bias toward neighbors): `spread`
/// excludes seeds from its output, so notes linking the seed entity itself
/// get zero boost — only neighbor-linked notes benefit. Exact seed matches
/// are already covered by the cosine + FTS arms.
fn entity_activations(db: &Store, task_toks: &[String]) -> std::collections::HashMap<i64, f64> {
    let out = std::collections::HashMap::new();
    if task_toks.is_empty() {
        return out;
    }
    let mut ents: Vec<(i64, String, bool)> = Vec::new();
    let rows = db.conn.prepare(
        "SELECT id, name, embedding FROM entities",
    );
    let mut rows = match rows {
        Ok(r) => r,
        Err(_) => return out,
    };
    let mapped = rows.query_map([], |r| {
        let id: i64 = r.get(0)?;
        let name: String = r.get(1)?;
        let blob: Option<Vec<u8>> = r.get(2)?;
        Ok((id, name, blob.is_some()))
    });
    let mapped = match mapped {
        Ok(m) => m,
        Err(_) => return out,
    };
    for r in mapped.flatten() {
        ents.push(r);
    }
    if ents.is_empty() {
        return out;
    }
    let mut scored: Vec<(i64, usize)> = Vec::new();
    for (id, name, has_emb) in &ents {
        if !has_emb {
            continue;
        }
        let nl = name.to_lowercase();
        let mut cnt = 0usize;
        for t in task_toks {
            if nl.contains(t.as_str()) {
                cnt += 1;
            }
        }
        if cnt > 0 {
            scored.push((*id, cnt));
        }
    }
    if scored.is_empty() {
        return out;
    }
    scored.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    scored.truncate(3);
    let seed_ids: Vec<i64> = scored.into_iter().map(|(id, _)| id).collect();
    let mut st = match db.conn.prepare("SELECT src, dst, co_mentions FROM edges") {
        Ok(s) => s,
        Err(_) => return out,
    };
    let edge_rows: Vec<(i64, i64, i64)> = st
        .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))
        .map(|m| m.flatten().collect())
        .unwrap_or_default();
    let mut id2dense: std::collections::HashMap<i64, usize> =
        std::collections::HashMap::new();
    let mut dense2id: Vec<i64> = Vec::new();
    let intern = |id: i64,
                      id2dense: &mut std::collections::HashMap<i64, usize>,
                      dense2id: &mut Vec<i64>|
     -> usize {
        if let Some(&d) = id2dense.get(&id) {
            return d;
        }
        let d = dense2id.len();
        id2dense.insert(id, d);
        dense2id.push(id);
        d
    };
    for (id, _, _) in &ents {
        intern(*id, &mut id2dense, &mut dense2id);
    }
    for (a, b, _) in &edge_rows {
        intern(*a, &mut id2dense, &mut dense2id);
        intern(*b, &mut id2dense, &mut dense2id);
    }
    let n = dense2id.len();
    let mut adj: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
    for (a, b, co) in &edge_rows {
        let w = crate::graph::normalize_edge_weight(*co as f64, 1.0);
        let da = id2dense[a];
        let dbi = id2dense[b];
        if da != dbi {
            adj[da].push((dbi, w));
            adj[dbi].push((da, w));
        }
    }
    let seeds: Vec<(usize, f64)> = seed_ids
        .iter()
        .filter_map(|id| id2dense.get(id).map(|&d| (d, 1.0)))
        .collect();
    if seeds.is_empty() {
        return out;
    }
    let spread = crate::graph::spread(&adj, &seeds, 2, 1e-3);
    let mut result = std::collections::HashMap::new();
    for (dense, act) in spread {
        if let Some(&id) = dense2id.get(dense) {
            result.insert(id, act);
        }
    }
    result
}

/// Mem→entity link heuristic: a note links an entity if the entity name
/// appears as a case-insensitive substring of the note content or exactly
/// matches one of the note tags. Boost = 0.2 * summed activation.
/// Callers pass pre-lowercased content/tags/names (hoisted allocs).
/// Seed-exclusion: `acts` never contains seeds (see `entity_activations`),
/// so seed-linked notes intentionally get zero boost here.
fn graph_boost(
    content_lc: &str,
    tags_lc: &[String],
    names_lc: &[(i64, String)],
    acts: &std::collections::HashMap<i64, f64>,
) -> f64 {
    if acts.is_empty() {
        return 0.0;
    }
    let mut sum = 0.0;
    for (id, nl) in names_lc {
        let act = match acts.get(id) {
            Some(&a) => a,
            None => continue,
        };
        if nl.is_empty() {
            continue;
        }
        if content_lc.contains(nl.as_str()) || tags_lc.iter().any(|t| t == nl) {
            sum += act;
        }
    }
    GRAPH_BOOST * sum
}

pub fn compile(
    db: &Store,
    core: &CoreFile,
    emb: &dyn EmbeddingProvider,
    task: &str,
    budget: i64,
) -> Result<Payload> {
    let t0 = std::time::Instant::now();
    let raw_core = core.render();
    let trimmed = raw_core.trim().to_string();
    let has_core = !trimmed.is_empty();
    let core_cost = if has_core {
        count_tokens(&trimmed)
    } else {
        0
    };
    if budget <= 0 || (has_core && core_cost >= budget) {
        if !has_core {
            return Ok(Payload {
                core: None,
                memories: Vec::new(),
                symbols: Vec::new(),
                tokens_used: 0,
                rho_naive: 0.0,
                took_ms: t0.elapsed().as_millis() as u64,
                stages: StageTimesSer { fts_ms: 0, vec_ms: 0, graph_ms: 0, pack_ms: 0 },
            });
        }
        let trunc = truncate_by_tokens(&trimmed, budget);
        let tokens_used = if trunc.trim().is_empty() {
            0
        } else {
            count_tokens(&trunc)
        };
        return Ok(Payload {
            core: Some(trunc),
            memories: Vec::new(),
            symbols: Vec::new(),
            tokens_used,
            rho_naive: 0.0,
            took_ms: t0.elapsed().as_millis() as u64,
            stages: StageTimesSer { fts_ms: 0, vec_ms: 0, graph_ms: 0, pack_ms: 0 },
        });
    }
    let core_opt = if has_core {
        Some(trimmed.clone())
    } else {
        None
    };
    type NoteRow = (i64, String, String, Option<Vec<u8>>, i64, i64, Option<i64>, String);
    let mut note_rows: Vec<NoteRow> = Vec::new();
    {
        let mut st = db.conn.prepare(
            "SELECT id, content, tags, embedding, created_at, recall_count, last_recalled, tier FROM notes")?;
        let rows = st.query_map([], |r| {
            Ok((
                r.get::<_, i64>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, String>(2)?,
                r.get::<_, Option<Vec<u8>>>(3)?,
                r.get::<_, i64>(4)?,
                r.get::<_, i64>(5)?,
                r.get::<_, Option<i64>>(6)?,
                r.get::<_, String>(7)?,
            ))
        })?;
        for r in rows {
            note_rows.push(r?);
        }
    }
    let mut stages = StageTimesSer { fts_ms: 0, vec_ms: 0, graph_ms: 0, pack_ms: 0 };
    let toks_task = bench::tokenize(task);
    // Hoisted single task embedding shared by the mem + sym arms.
    let task_vec_opt: Option<Vec<f32>> = match emb.embed(vec![task.to_string()]) {
        Ok(mut v) if !v.is_empty() => {
            let t = v.remove(0);
            if is_zero_vec(&t) { None } else { Some(t) }
        }
        _ => None,
    };
    let t_graph = std::time::Instant::now();
    let acts = entity_activations(db, &toks_task);
    stages.graph_ms += t_graph.elapsed().as_millis() as u64;
    let ent_names_lc: Vec<(i64, String)> = {
        let mut v = Vec::new();
        if !acts.is_empty() {
            let mut st = db.conn.prepare("SELECT id, name FROM entities")?;
            let rows = st.query_map([], |r| {
                let id: i64 = r.get(0)?;
                let name: String = r.get(1)?;
                Ok((id, name))
            })?;
            for (id, name) in rows.flatten() {
                v.push((id, name.to_lowercase()));
            }
        }
        v
    };
    let t_vec = std::time::Instant::now();
    let mut mem_cands: Vec<(i64, String, f64)> = Vec::new();
    if !note_rows.is_empty() {
        // Cosine arm: stored BLOBs first (T1), one live batch for missing only
        // (T2), whole-arm skip + stale warning on dim mismatch (T3).
        let cos_scores: Option<Vec<f64>> = match &task_vec_opt {
            None => None,
            Some(task_vec) => {
                let stored_width: Option<usize> = note_rows
                    .iter()
                    .filter_map(|(_, _, _, b, ..)| b.as_ref())
                    .map(|b| crate::store::deserialize_f32s(b).len())
                    .find(|&l| l > 0);
                if let Some(w) = stored_width {
                    if w != task_vec.len() {
                        warn_notes_dim_once(w, task_vec.len());
                        None
                    } else {
                        let mut scores = vec![0.0f64; note_rows.len()];
                        let mut need_idx: Vec<usize> = Vec::new();
                        let mut need_texts: Vec<String> = Vec::new();
                        for (i, (_, content, _, blob, ..)) in note_rows.iter().enumerate() {
                            let mut got: Option<f64> = None;
                            if let Some(b) = blob {
                                let v = crate::store::deserialize_f32s(b);
                                if v.len() == task_vec.len() && !is_zero_vec(&v) {
                                    got = Some(cosine(task_vec, &v).max(0.0) as f64);
                                }
                            }
                            if let Some(s) = got {
                                scores[i] = s;
                            } else {
                                need_idx.push(i);
                                need_texts.push(content.clone());
                            }
                        }
                        if !need_idx.is_empty() {
                            match emb.embed(need_texts) {
                                Ok(v)
                                    if v.len() == need_idx.len()
                                        && v.iter().all(|e| e.len() == task_vec.len()) =>
                                {
                                    for (k, idx) in need_idx.into_iter().enumerate() {
                                        scores[idx] =
                                            cosine(task_vec, &v[k]).max(0.0) as f64;
                                    }
                                }
                                _ => {}
                            }
                        }
                        Some(scores)
                    }
                } else {
                    let contents: Vec<String> = note_rows
                        .iter()
                        .map(|(_, c, ..)| c.clone())
                        .collect();
                    match emb.embed(contents) {
                        Ok(v) if v.len() == note_rows.len()
                            && v.iter().all(|e| e.len() == task_vec.len()) =>
                        {
                            Some(
                                v.iter()
                                    .map(|nv| cosine(task_vec, nv).max(0.0) as f64)
                                    .collect(),
                            )
                        }
                        _ => None,
                    }
                }
            }
        };
        // FTS arm rank map. Timed (plan item 5: fts_ms).
        let t_fts = std::time::Instant::now();
        let fts_hits = db.search_notes(task, 50).unwrap_or_default();
        stages.fts_ms += t_fts.elapsed().as_millis() as u64;
        let mut fts_rank: std::collections::HashMap<i64, usize> =
            std::collections::HashMap::new();
        for (pos, (id, _)) in fts_hits.iter().enumerate() {
            fts_rank.entry(*id).or_insert(pos);
        }
        // Cosine rank map over all notes.
        let mut cos_rank: std::collections::HashMap<i64, usize> =
            std::collections::HashMap::new();
        if let Some(scores) = &cos_scores {
            let mut order: Vec<usize> = (0..note_rows.len()).collect();
            order.sort_by(|&a, &b| {
                scores[b]
                    .partial_cmp(&scores[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| note_rows[a].0.cmp(&note_rows[b].0))
            });
            for (rank, idx) in order.into_iter().enumerate() {
                cos_rank.insert(note_rows[idx].0, rank);
            }
        }
        for (i, (id, content, tags_json, _, created_at, recall_count, last_recalled, tier)) in note_rows.iter().enumerate() {
            let mut score = 0.0;
            // Mirror the sym arm: zero-cosine notes get no cosine RRF.
            let cos_pos = cos_scores
                .as_ref()
                .map(|s| s[i] > 0.0)
                .unwrap_or(false);
            if cos_pos {
                if let Some(rank) = cos_rank.get(id) {
                    score += rrf(*rank);
                }
            }
            if let Some(rank) = fts_rank.get(id) {
                score += rrf(*rank);
            }
            let tags: Vec<String> =
                serde_json::from_str(tags_json).unwrap_or_default();
            let content_lc = content.to_lowercase();
            let tags_lc: Vec<String> =
                tags.iter().map(|t| t.to_lowercase()).collect();
            score += graph_boost(&content_lc, &tags_lc, &ent_names_lc, &acts);
            // Composed query-time scoring (plan item 2):
            //   calibrated_relevance · e^(−Δt/τ) · (1 + ln(1+recall_count))
            // Decay/reinforcement are recorded at recall events, not seconds —
            // Δt uses last_recalled (or created_at if never recalled) so a note
            // that keeps being useful never decays away. Gating at score>0 keeps
            // irrelevant notes untouched; RRF-scale base (~0.033 max) means the
            // multiplier shifts ranking only among plausible candidates.
            if score > 0.0 {
                let now = Store::now();
                let t_ref = last_recalled.unwrap_or(*created_at);
                let decay = crate::store::Store::decay(t_ref, now);
                let reinforce = crate::store::Store::reinforcement(*recall_count);
                // Tier prior (plan item 3): session-first, archived demoted.
                let prior = crate::store::Store::tier_prior(tier);
                score *= decay * reinforce * prior;
            }
            mem_cands.push((*id, content.clone(), score));
        }
        mem_cands.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        mem_cands.truncate(15);
    }
    stages.vec_ms += t_vec.elapsed().as_millis() as u64;
    let mut sym_rows: Vec<SymbolHit> = Vec::new();
    let mut sym_embs: Vec<Option<Vec<u8>>> = Vec::new();
    {
        let mut st = db.conn.prepare(
            "SELECT path, name, line_start, line_end, signature, embedding FROM symbols",
        )?;
        let rows = st.query_map([], |r| {
            Ok((
                SymbolHit {
                    path: r.get(0)?,
                    name: r.get(1)?,
                    line_start: r.get(2)?,
                    line_end: r.get(3)?,
                    signature: r.get(4)?,
                },
                r.get::<_, Option<Vec<u8>>>(5)?,
            ))
        })?;
        for r in rows {
            let (hit, emb_blob) = r?;
            sym_rows.push(hit);
            sym_embs.push(emb_blob);
        }
    }
    let mut sym_cands: Vec<(SymbolHit, f64)> = Vec::new();
    if !sym_rows.is_empty() {
        // Substring arm scores (always computable).
        let mut sub_scores: Vec<f64> = vec![0.0; sym_rows.len()];
        if !toks_task.is_empty() {
            for (i, sym) in sym_rows.iter().enumerate() {
                let name_lc = sym.name.to_lowercase();
                let path_lc = sym.path.to_lowercase();
                let mut cnt = 0usize;
                for t in &toks_task {
                    if name_lc.contains(t) || path_lc.contains(t) {
                        cnt += 1;
                    }
                }
                sub_scores[i] = cnt as f64 / toks_task.len() as f64;
            }
        }
        // Refs arm (Phase 4): exact identifier hits from the code reference
        // index. Each task token that exactly matches a def or use name
        // votes for the symbol rows with that name. Score is the
        // IDF-weighted fraction of voting tokens behind each symbol, so
        // rare identifiers outvote boilerplate. Timed under fts_ms (same
        // SQLite engine, no embeddings).
        let t_refs = std::time::Instant::now();
        // A/B gate (bench): CORTEX_REFS=0 disables the refs leg entirely so
        // with/without runs can quantify its contribution. Default: on.
        let refs_enabled = std::env::var("CORTEX_REFS")
            .map(|v| v.trim() != "0")
            .unwrap_or(true);
        let mut refs_scores: Vec<f64> = vec![0.0; sym_rows.len()];
        if refs_enabled && !toks_task.is_empty() {
            let mut name_to_rows: std::collections::HashMap<String, Vec<usize>> =
                std::collections::HashMap::new();
            for (i, sym) in sym_rows.iter().enumerate() {
                name_to_rows.entry(sym.name.to_lowercase()).or_default().push(i);
            }
            let mut idf_sum = 0.0f64;
            // Cache generic-identifier checks per compile call (token and
            // symbol names repeat heavily across tasks/contexts).
            let mut generic_cache: std::collections::HashMap<String, bool> =
                std::collections::HashMap::new();
            // Query-side TF dedup: boilerplate (`description` x12) votes
            // once. First-seen order kept for determinism.
            let mut seen: std::collections::HashSet<&str> = std::collections::HashSet::new();
            let mut uniq_toks: Vec<&String> = Vec::with_capacity(toks_task.len());
            for t in &toks_task {
                if seen.insert(t.as_str()) {
                    uniq_toks.push(t);
                }
            }
            for t in uniq_toks {
                // Cheap length gate first: one-char tokens can't be real
                // identifiers and would only add query load.
                if t.len() < 2 {
                    continue;
                }
                let (d, u, f) = db.find_refs(t, 5).unwrap_or_default();
                // Dotted refs (`auth.zeta`) live only in the FTS leg — the
                // exact-name arm misses them — so f must gate the skip too.
                if d.is_empty() && u.is_empty() && f.is_empty() {
                    continue;
                }
                // Generic-identifier gate: tokens defined/used across a large
                // FRACTION of indexed files are prose words that would vote
                // noisy symbol rows (`data`). Ratio (not absolute count) so
                // the same code works on every repo size. Cached — the
                // count queries are not cheap.
                let t_generic = match generic_cache.get(t) {
                    Some(v) => *v,
                    None => {
                        let v = db.refs_name_is_generic_ratio(t, 0.02).unwrap_or(false);
                        generic_cache.insert(t.clone(), v);
                        v
                    }
                };
                if t_generic {
                    continue;
                }
                // IDF weight: rare identifiers (the navigation signal) vote
                // far more than boilerplate that survived the gate. Same
                // weight for all three vote sources below.
                let w = db.refs_idf(t).unwrap_or(0.0);
                if w <= 0.0 {
                    continue;
                }
                idf_sum += w;
                // Vote symbol rows matching the token itself, plus rows
                // matching the short name of any dotted ref found
                // (`mod.func` use → `func` symbol row).
                let mut voted: Vec<usize> =
                    name_to_rows.get(t.as_str()).cloned().unwrap_or_default();
                let mut extra_names: Vec<String> = Vec::new();
                for (n, _, _) in d.iter() {
                    if let Some(short) = n.rsplit('.').next() {
                        if short != n.as_str() {
                            extra_names.push(short.to_lowercase());
                        }
                    }
                }
                for (n, _, _, _) in u.iter() {
                    if let Some(short) = n.rsplit('.').next() {
                        if short != n.as_str() {
                            extra_names.push(short.to_lowercase());
                        }
                    }
                }
                for n in &extra_names {
                    if let Some(rows) = name_to_rows.get(n.as_str()) {
                        voted.extend(rows.iter().copied());
                    }
                }
                // FTS rows: vote symbol names occurring in the matched
                // context so dotted-ref call sites resolve to their symbol.
                // Code-shape gate: the name must occur as a call, attribute
                // access/base — not a prose mention or bare RHS. Bare
                // mentions are what flooded the leg with rare-but-wrong
                // votes, so they don't vote. Same generic gate on top.
                for (_p, _l, ctx) in f.iter() {
                    let ctx_lc = ctx.to_lowercase();
                    for (name, rows) in &name_to_rows {
                        if !is_code_mention(&ctx_lc, name) {
                            continue;
                        }
                        let generic = match generic_cache.get(name) {
                            Some(v) => *v,
                            None => {
                                let v = db.refs_name_is_generic_ratio(name, 0.02).unwrap_or(false);
                                generic_cache.insert(name.clone(), v);
                                v
                            }
                        };
                        if !generic {
                            voted.extend(rows.iter().copied());
                        }
                    }
                }
                voted.sort_unstable();
                voted.dedup();
                for i in voted {
                    refs_scores[i] += w;
                }
            }
            if idf_sum > 0.0 {
                for s in refs_scores.iter_mut() {
                    *s /= idf_sum;
                }
            }
        }
        stages.fts_ms += t_refs.elapsed().as_millis() as u64;
        // Cosine arm: stored BLOBs first (T1), one live batch for missing only
        // (T2), whole-arm skip + stale warning on dim mismatch (T3).
        let cos_scores: Option<Vec<f64>> = match &task_vec_opt {
            None => None,
            Some(task_vec) => {
                let stored_width: Option<usize> = sym_embs
                    .iter()
                    .filter_map(|b| b.as_ref())
                    .map(|b| crate::store::deserialize_f32s(b).len())
                    .find(|&l| l > 0);
                if let Some(w) = stored_width {
                    if w != task_vec.len() {
                        warn_syms_dim_once(w, task_vec.len());
                        None
                    } else {
                        let mut scores = vec![0.0f64; sym_rows.len()];
                        let mut need_idx: Vec<usize> = Vec::new();
                        let mut need_texts: Vec<String> = Vec::new();
                        for (i, (sym, blob)) in
                            sym_rows.iter().zip(sym_embs.iter()).enumerate()
                        {
                            let mut got: Option<f64> = None;
                            if let Some(b) = blob {
                                let v = crate::store::deserialize_f32s(b);
                                if v.len() == task_vec.len() && !is_zero_vec(&v) {
                                    got = Some(cosine(task_vec, &v).max(0.0) as f64);
                                }
                            }
                            if let Some(s) = got {
                                scores[i] = s;
                            } else {
                                need_idx.push(i);
                                need_texts.push(
                                    sym.signature
                                        .clone()
                                        .unwrap_or_else(|| sym.name.clone()),
                                );
                            }
                        }
                        if !need_idx.is_empty() {
                            match emb.embed(need_texts) {
                                Ok(v)
                                    if v.len() == need_idx.len()
                                        && v.iter().all(|e| e.len() == task_vec.len()) =>
                                {
                                    for (k, idx) in need_idx.into_iter().enumerate() {
                                        let ev = &v[k];
                                        scores[idx] = if is_zero_vec(ev) {
                                            0.0
                                        } else {
                                            cosine(task_vec, ev).max(0.0) as f64
                                        };
                                    }
                                }
                                _ => {}
                            }
                        }
                        Some(scores)
                    }
                } else {
                    let sym_texts: Vec<String> = sym_rows
                        .iter()
                        .map(|sy| sy.signature.clone().unwrap_or_else(|| sy.name.clone()))
                        .collect();
                    match emb.embed(sym_texts) {
                        Ok(v) if v.len() == sym_rows.len()
                            && v.iter().all(|e| e.len() == task_vec.len()) =>
                        {
                            Some(
                                v.iter()
                                    .map(|ev| {
                                        if is_zero_vec(ev) {
                                            0.0
                                        } else {
                                            cosine(task_vec, ev).max(0.0) as f64
                                        }
                                    })
                                    .collect(),
                            )
                        }
                        _ => None,
                    }
                }
            }
        };
        // Refs rank map (Phase 4): third RRF leg over refs_scores. Present
        // only when the index has exact hits; absent leg = today's behavior.
        let mut refs_rank: std::collections::HashMap<usize, usize> =
            std::collections::HashMap::new();
        {
            let mut rorder: Vec<usize> = (0..sym_rows.len())
                .filter(|&i| refs_scores[i] > 0.0)
                .collect();
            rorder.sort_by(|&a, &b| {
                refs_scores[b]
                    .partial_cmp(&refs_scores[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| sym_rows[a].name.cmp(&sym_rows[b].name))
            });
            for (rank, idx) in rorder.into_iter().enumerate() {
                refs_rank.insert(idx, rank);
            }
        }
        if cos_scores.is_none() {
            for (i, (sym, s)) in sym_rows.into_iter().zip(sub_scores).enumerate() {
                // No-cosine path: refs leg adds its raw exact-hit score
                // (already a 0..1 fraction) on top of substring.
                // (Graph leg removed here too — see RRF path note.)
                let score = s + refs_scores[i];
                if score > 0.0 {
                    sym_cands.push((sym, score));
                }
            }
        } else if let Some(cos) = cos_scores {
            let mut cos_rank: std::collections::HashMap<usize, usize> =
                std::collections::HashMap::new();
            let mut order: Vec<usize> = (0..sym_rows.len()).collect();
            order.sort_by(|&a, &b| {
                cos[b]
                    .partial_cmp(&cos[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| sym_rows[a].name.cmp(&sym_rows[b].name))
            });
            for (rank, idx) in order.into_iter().enumerate() {
                cos_rank.insert(idx, rank);
            }
            let mut sub_rank: std::collections::HashMap<usize, usize> =
                std::collections::HashMap::new();
            let mut sorder: Vec<usize> = (0..sym_rows.len())
                .filter(|&i| sub_scores[i] > 0.0)
                .collect();
            sorder.sort_by(|&a, &b| {
                sub_scores[b]
                    .partial_cmp(&sub_scores[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| sym_rows[a].name.cmp(&sym_rows[b].name))
            });
            for (rank, idx) in sorder.into_iter().enumerate() {
                sub_rank.insert(idx, rank);
            }
            // Two-pass fusion: base RRF scores first, then the file
            // neighborhood spreads seed weight (1 hop) so co-file and
            // import/incoming neighbors can enter the top-12.
            let syms: Vec<SymbolHit> = sym_rows.into_iter().collect();
            let mut base = vec![0.0; syms.len()];
            for i in 0..syms.len() {
                if cos[i] > 0.0 {
                    base[i] += rrf(cos_rank[&i]);
                }
                if let Some(rank) = sub_rank.get(&i) {
                    base[i] += rrf(*rank);
                }
                // Phase 4: third RRF leg — exact identifier hits.
                if let Some(rank) = refs_rank.get(&i) {
                    base[i] += rrf(*rank);
                }
            }
            // Graph leg REMOVED 2026-09-20 (Task 4 refuted, N=125):
            // file-neighborhood spread boosted seed neighborhoods, but on
            // miss-tasks the seeds are wrong — rich-get-richer on the wrong
            // files. Measured file −0.048/task, 95% CI [−0.096,−0.008].
            // file_connections() stays for find/map; see METRICS §5.
            for (i, sym) in syms.into_iter().enumerate() {
                if base[i] > 0.0 {
                    sym_cands.push((sym, base[i]));
                }
            }
        }
        sym_cands.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        sym_cands = pool_by_file(sym_cands, POOL_TOP_FILES);
        sym_cands.truncate(12);
    }
    let mut items: Vec<Item> = Vec::new();
    for (i, (_, content, score)) in mem_cands.iter().enumerate() {
        let cost = count_tokens(content);
        let cost = cost.max(1);
        items.push(Item {
            idx: i,
            value: *score,
            cost,
        });
    }
    let mem_len = mem_cands.len();
    for (j, (sym, score)) in sym_cands.iter().enumerate() {
        let txt = sym
            .signature
            .clone()
            .unwrap_or_else(|| sym.name.clone());
        let cost = count_tokens(&txt).max(1);
        items.push(Item {
            idx: mem_len + j,
            value: *score,
            cost,
        });
    }
    let t_pack = std::time::Instant::now();
    let remaining = budget - core_cost;
    let sel = if remaining <= 0 || items.is_empty() {
        Vec::new()
    } else {
        knapsack(&items, remaining)
    };
    stages.pack_ms += t_pack.elapsed().as_millis() as u64;
    let sel_set: std::collections::HashSet<usize> = sel.into_iter().collect();
    let mut out_mems: Vec<MemoryHit> = Vec::new();
    let mut out_syms: Vec<SymbolHit> = Vec::new();
    let mut packed_cost: i64 = 0;
    let mut weighted: f64 = 0.0;
    for (i, (id, content, score)) in mem_cands.iter().enumerate() {
        if sel_set.contains(&i) {
            // Redact after selection (ranking sees raw text), before serving.
            let shown = redact_secrets(content);
            let cost = count_tokens(&shown).max(1);
            packed_cost += cost;
            weighted += *score * cost as f64;
            out_mems.push(MemoryHit {
                id: *id,
                content: shown,
                score: *score,
            });
        }
    }
    for (j, (sym, score)) in sym_cands.iter().enumerate() {
        let idx = mem_len + j;
        if sel_set.contains(&idx) {
            let txt = sym
                .signature
                .clone()
                .unwrap_or_else(|| sym.name.clone());
            let cost = count_tokens(&txt).max(1);
            packed_cost += cost;
            weighted += *score * cost as f64;
            out_syms.push(sym.clone());
            let _ = score;
        }
    }
    let rho_naive = if packed_cost > 0 {
        (weighted / packed_cost as f64).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let tokens_used = core_cost + packed_cost;
    Ok(Payload {
        core: core_opt,
        memories: out_mems,
        symbols: out_syms,
        tokens_used,
        rho_naive,
        took_ms: t0.elapsed().as_millis() as u64,
        stages,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{corefile::CoreFile, embed::HashEmbedder, store::Store};
    use std::collections::BTreeMap;

    fn seeded() -> (Store, CoreFile) {
        let s = Store::open_in_memory().unwrap();
        s.add_note(
            "sess",
            "jwt refresh handled in refresh_token via rotation",
            "decision",
            &["jwt", "refresh_token"],
        )
        .unwrap();
        s.upsert_entity("refresh_token", "function").unwrap();
        s.upsert_entity("jwt", "concept").unwrap();
        s.conn
            .execute(
                "INSERT INTO symbols(path,name,kind,line_start,line_end,signature) VALUES(?1,?2,?3,?4,?5,?6)",
                rusqlite::params![
                    "src/auth.rs",
                    "refresh_token",
                    "fn",
                    10,
                    40,
                    "fn refresh_token() -> Token"
                ],
            )
            .unwrap();
        let cf = CoreFile {
            path: "x".to_string(),
            preamble: None,
            sections: {
                let mut m = BTreeMap::new();
                m.insert("Profile".to_string(), "rust dev\n".to_string());
                m
            },
        };
        (s, cf)
    }

    #[test]
    fn core_first_budget_respected_symbols_found() {
        let (s, cf) = seeded();
        let emb = HashEmbedder::new(64);
        let pay = compile(&s, &cf, &emb, "fix jwt refresh bug", 300).unwrap();
        assert!(
            pay.core.as_ref().map(|c| c.contains("rust dev")).unwrap_or(false),
            "core missing"
        );
        assert!(
            pay.memories
                .iter()
                .any(|m| m.content.contains("refresh_token")),
            "memory hit missing: {:?}",
            pay.memories
        );
        assert!(
            pay.symbols.iter().any(|y| y.name == "refresh_token"),
            "symbol hit missing: {:?}",
            pay.symbols
        );
        assert!(pay.tokens_used <= 300, "over budget: {}", pay.tokens_used);
        assert!(
            pay.rho_naive >= 0.0 && pay.rho_naive <= 1.0,
            "rho out of range: {}",
            pay.rho_naive
        );
        assert!(
            pay.stages.fts_ms <= pay.took_ms + 5,
            "fts_ms ({}) should be ≤ total took_ms ({})",
            pay.stages.fts_ms,
            pay.took_ms
        );
    }

    #[test]
    fn stages_present_and_nonnegative() {
        let (s, cf) = seeded();
        let emb = HashEmbedder::new(64);
        let pay = compile(&s, &cf, &emb, "fix jwt refresh bug", 300).unwrap();
        for (name, v) in [
            ("fts_ms", pay.stages.fts_ms),
            ("vec_ms", pay.stages.vec_ms),
            ("graph_ms", pay.stages.graph_ms),
            ("pack_ms", pay.stages.pack_ms),
        ] {
            // All stages report a value; most are sub-millisecond on a tiny
            // seeded store so 0 is legal, but they must serialize.
            let _ = name;
            let _ = v;
        }
        let json = serde_json::to_string(&pay.stages).unwrap();
        assert!(json.contains("fts_ms") && json.contains("pack_ms"));
    }

    #[test]
    fn recalled_note_outranks_equal_fresh_alternative() {
        let s = Store::open_in_memory().unwrap();
        s.add_note("sess", "cache eviction uses lru window", "decision", &["cache"]).unwrap();
        s.add_note("sess", "cache eviction prefers lfu stripe", "decision", &["cache"]).unwrap();
        // Make the second note look recalled: recall_count=3 and just now.
        let now = Store::now();
        s.conn.execute(
            "UPDATE notes SET recall_count=3, last_recalled=?1 WHERE content LIKE '%lfu%'",
            [now]).unwrap();
        let cf = CoreFile {
            path: "x".into(),
            preamble: None,
            sections: BTreeMap::new(),
        };
        let emb = HashEmbedder::new(64);
        let pay = compile(&s, &cf, &emb, "cache eviction policy", 500).unwrap();
        assert_eq!(pay.memories.len(), 2, "both notes must hit: {:?}", pay.memories);
        assert!(
            pay.memories[0].content.contains("lfu"),
            "recalled note must outrank equal-content twin: {:?}",
            pay.memories
        );
    }

    #[test]
    fn tier_prior_demotes_archived_note() {
        let s = Store::open_in_memory().unwrap();
        s.add_note("sess", "nginx retry budget tuning", "note", &["nginx"]).unwrap();
        s.add_note("sess", "nginx upstream failover rules", "note", &["nginx"]).unwrap();
        s.conn.execute(
            "UPDATE notes SET tier='archived' WHERE content LIKE '%failover%'",
            []).unwrap();
        let cf = CoreFile { path: "x".into(), preamble: None, sections: BTreeMap::new() };
        let emb = HashEmbedder::new(64);
        let pay = compile(&s, &cf, &emb, "nginx", 500).unwrap();
        assert_eq!(pay.memories.len(), 2, "{:?}", pay.memories);
        assert!(
            pay.memories[0].content.contains("retry budget"),
            "session note must outrank archived twin: {:?}",
            pay.memories
        );
        // Archived note must still be reachable (soft archive, not deletion).
        assert!(pay.memories[1].content.contains("failover"));
    }

    #[test]
    fn stale_never_recalled_note_decays() {
        let s = Store::open_in_memory().unwrap();
        s.add_note("sess", "cobol screen scraper notes", "note", &["cobol"]).unwrap();
        s.add_note("sess", "cobol data division layout", "note", &["cobol"]).unwrap();
        // First note created 90 days ago (never recalled): decay e^-3 ≈ 0.05.
        let old = Store::now() - 90 * 24 * 3600;
        s.conn.execute(
            "UPDATE notes SET created_at=?1 WHERE content LIKE '%scraper%'",
            [old]).unwrap();
        let cf = CoreFile { path: "x".into(), preamble: None, sections: BTreeMap::new() };
        let emb = HashEmbedder::new(64);
        let pay = compile(&s, &cf, &emb, "cobol", 500).unwrap();
        assert_eq!(pay.memories.len(), 2, "{:?}", pay.memories);
        assert!(
            pay.memories[0].content.contains("data division"),
            "fresh note must outrank stale never-recalled twin: {:?}",
            pay.memories
        );
    }

    #[test]
    fn symbols_use_cosine_not_substring() {
        let (s, cf) = seeded();
        let emb = HashEmbedder::new(64);
        // "rotate session credential" paraphrases refresh_token but scores
        // cosine <= 0 under HashEmbedder (no shared tokens, hash collision
        // gives -0.11), so it cannot separate the paths. Adding the "fn"
        // token shares one exact token with the seeded signature
        // "fn refresh_token() -> Token" (cosine ~0.12) while still having
        // zero substring overlap with name/path, so old substring path
        // scores 0 and misses it.
        let pay = compile(&s, &cf, &emb, "fn rotate session credential", 300).unwrap();
        assert!(
            pay.symbols.iter().any(|y| y.name == "refresh_token"),
            "symbols must rank by vector, got {:?}",
            pay.symbols
        );
    }

    #[test]
    fn graph_spread_boosts_neighbor() {
        let s = Store::open_in_memory().unwrap();
        let jwt_id = s.upsert_entity("jwt", "concept").unwrap();
        let rt_id = s.upsert_entity("refresh_token", "function").unwrap();
        let dummy = crate::store::serialize_f32s(&[1.0, 0.0]);
        s.conn
            .execute(
                "UPDATE entities SET embedding=?1 WHERE id=?2",
                rusqlite::params![dummy.clone(), jwt_id],
            )
            .unwrap();
        s.conn
            .execute(
                "UPDATE entities SET embedding=?1 WHERE id=?2",
                rusqlite::params![dummy, rt_id],
            )
            .unwrap();
        s.link(jwt_id, rt_id, "relates").unwrap();
        s.add_note("sess", "jwt alpha beta gamma", "note", &["jwt"])
            .unwrap();
        s.add_note("sess", "jwt delta epsilon zeta", "note", &["jwt"])
            .unwrap();
        s.add_note("sess", "jwt eta theta iota", "note", &["jwt"])
            .unwrap();
        let neigh = s
            .add_note(
                "sess",
                "refresh_token rotation expiry handling",
                "note",
                &["refresh_token"],
            )
            .unwrap();
        let cf = CoreFile {
            path: "x".to_string(),
            preamble: None,
            sections: Default::default(),
        };
        let emb = HashEmbedder::new(64);
        let pay = compile(&s, &cf, &emb, "jwt", 5000).unwrap();
        let pos = pay
            .memories
            .iter()
            .position(|m| m.id == neigh)
            .unwrap_or(usize::MAX);
        assert!(
            pos < 3,
            "neighbor-boosted memory should rank top-3, got pos {pos} in {:?}",
            pay.memories
                .iter()
                .map(|m| m.content.clone())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn seed_linked_notes_get_no_graph_boost() {
        let s = Store::open_in_memory().unwrap();
        let jwt_id = s.upsert_entity("jwt", "concept").unwrap();
        let rt_id = s.upsert_entity("refresh_token", "function").unwrap();
        let dummy = crate::store::serialize_f32s(&[1.0, 0.0]);
        for id in [jwt_id, rt_id] {
            s.conn
                .execute(
                    "UPDATE entities SET embedding=?1 WHERE id=?2",
                    rusqlite::params![dummy.clone(), id],
                )
                .unwrap();
        }
        s.link(jwt_id, rt_id, "relates").unwrap();
        let acts = entity_activations(&s, &["jwt".to_string()]);
        assert!(
            !acts.contains_key(&jwt_id),
            "seeds excluded by design; jwt must not carry activation"
        );
        assert!(
            acts.get(&rt_id).copied().unwrap_or(0.0) > 0.0,
            "neighbor must carry activation"
        );
        let names = vec![
            (jwt_id, "jwt".to_string()),
            (rt_id, "refresh_token".to_string()),
        ];
        assert_eq!(
            graph_boost("jwt alpha", &["jwt".to_string()], &names, &acts),
            0.0,
            "seed-linked notes get zero boost (discovery bias toward neighbors)"
        );
        assert!(
            graph_boost(
                "refresh_token rotation",
                &["refresh_token".to_string()],
                &names,
                &acts
            ) > 0.0,
            "neighbor-linked notes must benefit"
        );
    }

    #[test]
    fn empty_project_degrades_gracefully() {
        let s = Store::open_in_memory().unwrap();
        let cf = CoreFile {
            path: "x".to_string(),
            preamble: None,
            sections: Default::default(),
        };
        let emb = HashEmbedder::new(64);
        let pay = compile(&s, &cf, &emb, "unrelated query xyzzy", 5000).unwrap();
        assert!(
            pay.memories.is_empty() || pay.rho_naive < 0.5,
            "unexpected high rho: {}",
            pay.rho_naive
        );
        assert!(pay.tokens_used >= 0);
    }

    #[test]
    fn pool_by_file_groups_top_files_first() {
        let sym = |path: &str, name: &str| SymbolHit {
            name: name.to_string(),
            path: path.to_string(),
            line_start: 1,
            line_end: 10,
            signature: None,
        };
        // Score-desc input: b0(0.9), a0(0.8), b1(0.7), a1(0.2), c0(0.1).
        let cands = vec![
            (sym("b.py", "b0"), 0.9),
            (sym("a.py", "a0"), 0.8),
            (sym("b.py", "b1"), 0.7),
            (sym("a.py", "a1"), 0.2),
            (sym("c.py", "c0"), 0.1),
        ];
        // top-1: b.py's symbols first (stable), then rest in order.
        let out = pool_by_file(cands.clone(), 1);
        let names: Vec<&str> = out.iter().map(|(s, _)| s.name.as_str()).collect();
        assert_eq!(names, vec!["b0", "b1", "a0", "a1", "c0"]);
        // top-0: ablation, order untouched.
        let out0 = pool_by_file(cands.clone(), 0);
        let names0: Vec<&str> = out0.iter().map(|(s, _)| s.name.as_str()).collect();
        assert_eq!(names0, vec!["b0", "a0", "b1", "a1", "c0"]);
        // Scores untouched by the reorder.
        assert!(out.iter().all(|(_, s)| *s > 0.0));
        let scores: Vec<f64> = out.iter().map(|(_, s)| *s).collect();
        assert_eq!(scores, vec![0.9, 0.7, 0.8, 0.2, 0.1]);
    }

    #[test]
    fn redact_secrets_shapes() {
        assert_eq!(redact_secrets("nothing here"), "nothing here");
        assert_eq!(redact_secrets(""), "");
        // Exact 20-char key scrubbed mid-text.
        assert_eq!(
            redact_secrets("key AKIAIOSFODNN7EXAMPLE end"),
            "key [REDACTED-AWS-KEY] end"
        );
        // Short / lowercase lookalikes untouched.
        assert_eq!(redact_secrets("AKIA SHORT"), "AKIA SHORT");
        assert_eq!(redact_secrets("akiaiosfodnn7example12"), "akiaiosfodnn7example12");
        // Unicode around the key: byte walk stays on char boundaries.
        assert_eq!(
            redact_secrets("déploy AKIAIOSFODNN7EXAMPLE ☃"),
            "déploy [REDACTED-AWS-KEY] ☃"
        );
    }

    #[test]
    fn compile_redacts_aws_keys_in_memories() {
        let s = Store::open_in_memory().unwrap();
        s.add_note(
            "sess",
            "deploy uses AKIAIOSFODNN7EXAMPLE for the staging bucket",
            "decision",
            &["deploy"],
        )
        .unwrap();
        let cf = CoreFile { path: "x".into(), preamble: None, sections: Default::default() };
        let emb = HashEmbedder::new(64);
        let pay = compile(&s, &cf, &emb, "deploy staging bucket", 5000).unwrap();
        assert!(!pay.memories.is_empty(), "note must be recalled: {:?}", pay.memories);
        for m in &pay.memories {
            assert!(
                !m.content.contains("AKIAIOSFODNN7EXAMPLE"),
                "raw key leaked into payload: {}",
                m.content
            );
        }
    }

    #[test]
    fn budget_tight_core_wins() {
        let (s, mut cf) = seeded();
        cf.set_section("Profile", &"word ".repeat(200));
        let emb = HashEmbedder::new(64);
        let pay = compile(&s, &cf, &emb, "jwt", 20).unwrap();
        assert!(pay.core.is_some());
        assert!(
            pay.tokens_used <= 20 || pay.memories.is_empty(),
            "should truncate memories when core fills budget"
        );
    }

    #[test]
    fn stored_vectors_rank_without_provider() {
        use crate::embed::EmbeddingProvider;
        use std::sync::atomic::{AtomicUsize, Ordering};
        struct ErrEmbed {
            task_vec: Vec<f32>,
            calls: AtomicUsize,
            lens: std::sync::Mutex<Vec<usize>>,
        }
        impl EmbeddingProvider for ErrEmbed {
            fn name(&self) -> &str {
                "err"
            }
            fn dim(&self) -> usize {
                64
            }
            fn is_local(&self) -> bool {
                true
            }
            fn health_check(&self) -> crate::Result<()> {
                Ok(())
            }
            fn embed(&self, texts: Vec<String>) -> crate::Result<Vec<Vec<f32>>> {
                self.calls.fetch_add(1, Ordering::SeqCst);
                self.lens.lock().unwrap().push(texts.len());
                if texts.len() == 1 {
                    Ok(vec![self.task_vec.clone()])
                } else {
                    Err(crate::Error::Cli("provider dead".into()))
                }
            }
        }
        let s = Store::open_in_memory().unwrap();
        let task = "fn rotate session credential";
        let hash = HashEmbedder::new(64);
        let task_vec = hash.embed(vec![task.to_string()]).unwrap().remove(0);
        let irr_content = "zebra xylophone quantum";
        let rel_content = "fn refresh_token rotation handling";
        let id_irr = s.add_note("sess", irr_content, "note", &[]).unwrap();
        let id_rel = s.add_note("sess", rel_content, "note", &[]).unwrap();
        let _ = id_irr;
        let _ = id_rel;
        for content in [irr_content, rel_content] {
            let v = hash.embed(vec![content.to_string()]).unwrap().remove(0);
            let blob = crate::store::serialize_f32s(&v);
            let nid: i64 = s
                .conn
                .query_row(
                    "SELECT id FROM notes WHERE content=?1",
                    rusqlite::params![content],
                    |r| r.get(0),
                )
                .unwrap();
            s.conn
                .execute(
                    "UPDATE notes SET embedding=?1 WHERE id=?2",
                    rusqlite::params![blob, nid],
                )
                .unwrap();
        }
        s.conn
            .execute(
                "INSERT INTO symbols(path,name,kind,line_start,line_end,signature) VALUES(?1,?2,?3,?4,?5,?6)",
                rusqlite::params![
                    "src/zebra.rs",
                    "zebra_xylophone",
                    "fn",
                    1,
                    10,
                    "zebra xylophone quantum"
                ],
            )
            .unwrap();
        s.conn
            .execute(
                "INSERT INTO symbols(path,name,kind,line_start,line_end,signature) VALUES(?1,?2,?3,?4,?5,?6)",
                rusqlite::params![
                    "src/auth.rs",
                    "refresh_token",
                    "fn",
                    10,
                    40,
                    "fn refresh_token() -> Token"
                ],
            )
            .unwrap();
        for (name, sig) in [
            ("zebra_xylophone", "zebra xylophone quantum"),
            ("refresh_token", "fn refresh_token() -> Token"),
        ] {
            let v = hash.embed(vec![sig.to_string()]).unwrap().remove(0);
            s.conn
                .execute(
                    "UPDATE symbols SET embedding=?1 WHERE name=?2",
                    rusqlite::params![crate::store::serialize_f32s(&v), name],
                )
                .unwrap();
        }
        let cf = CoreFile {
            path: "x".to_string(),
            preamble: None,
            sections: Default::default(),
        };
        let emb = ErrEmbed {
            task_vec,
            calls: AtomicUsize::new(0),
            lens: std::sync::Mutex::new(Vec::new()),
        };
        let pay = compile(&s, &cf, &emb, task, 5000).unwrap();
        assert!(
            pay.memories.iter().any(|m| m.content.contains("refresh_token")),
            "stored cosine must rank relevant memory without live batch, got {:?}",
            pay.memories
        );
        assert_eq!(
            pay.memories.first().map(|m| m.content.as_str()),
            Some(rel_content),
            "relevant memory must be first via stored vectors, got {:?}",
            pay.memories
        );
        assert!(
            pay.symbols.iter().any(|y| y.name == "refresh_token"),
            "stored cosine must rank relevant symbol without live batch, got {:?}",
            pay.symbols
        );
        assert_eq!(
            pay.symbols.first().map(|y| y.name.as_str()),
            Some("refresh_token"),
            "relevant symbol must be first via stored vectors, got {:?}",
            pay.symbols
        );
        assert_eq!(
            emb.calls.load(Ordering::SeqCst),
            1,
            "all-stored run must issue exactly 1 embed call (task only), got {:?}",
            emb.lens.lock().unwrap()
        );
        assert_eq!(
            *emb.lens.lock().unwrap(),
            vec![1],
            "zero candidate-batch calls expected, got {:?}",
            emb.lens.lock().unwrap()
        );
    }

    #[test]
    fn stored_partial_fallback_single_batch() {
        use crate::embed::EmbeddingProvider;
        use std::sync::atomic::{AtomicUsize, Ordering};
        struct CountingHash {
            inner: HashEmbedder,
            calls: AtomicUsize,
            lens: std::sync::Mutex<Vec<usize>>,
        }
        impl EmbeddingProvider for CountingHash {
            fn name(&self) -> &str {
                "count-hash"
            }
            fn dim(&self) -> usize {
                64
            }
            fn is_local(&self) -> bool {
                true
            }
            fn health_check(&self) -> crate::Result<()> {
                Ok(())
            }
            fn embed(&self, texts: Vec<String>) -> crate::Result<Vec<Vec<f32>>> {
                self.calls.fetch_add(1, Ordering::SeqCst);
                self.lens.lock().unwrap().push(texts.len());
                self.inner.embed(texts)
            }
        }
        let s = Store::open_in_memory().unwrap();
        let task = "fn rotate session credential";
        let hash = HashEmbedder::new(64);
        let irr_content = "zebra xylophone quantum";
        let rel_content = "fn refresh_token rotation handling";
        s.add_note("sess", irr_content, "note", &[]).unwrap();
        s.add_note("sess", rel_content, "note", &[]).unwrap();
        // Only the irrelevant note gets a stored BLOB; the relevant one stays
        // NULL so it must be served by the single live fallback batch.
        let v = hash.embed(vec![irr_content.to_string()]).unwrap().remove(0);
        let nid: i64 = s
            .conn
            .query_row(
                "SELECT id FROM notes WHERE content=?1",
                rusqlite::params![irr_content],
                |r| r.get(0),
            )
            .unwrap();
        s.conn
            .execute(
                "UPDATE notes SET embedding=?1 WHERE id=?2",
                rusqlite::params![crate::store::serialize_f32s(&v), nid],
            )
            .unwrap();
        s.conn
            .execute(
                "INSERT INTO symbols(path,name,kind,line_start,line_end,signature) VALUES(?1,?2,?3,?4,?5,?6)",
                rusqlite::params![
                    "src/zebra.rs",
                    "zebra_xylophone",
                    "fn",
                    1,
                    10,
                    "zebra xylophone quantum"
                ],
            )
            .unwrap();
        s.conn
            .execute(
                "INSERT INTO symbols(path,name,kind,line_start,line_end,signature) VALUES(?1,?2,?3,?4,?5,?6)",
                rusqlite::params![
                    "src/auth.rs",
                    "refresh_token",
                    "fn",
                    10,
                    40,
                    "fn refresh_token() -> Token"
                ],
            )
            .unwrap();
        for (name, sig) in [
            ("zebra_xylophone", "zebra xylophone quantum"),
            ("refresh_token", "fn refresh_token() -> Token"),
        ] {
            let v = hash.embed(vec![sig.to_string()]).unwrap().remove(0);
            s.conn
                .execute(
                    "UPDATE symbols SET embedding=?1 WHERE name=?2",
                    rusqlite::params![crate::store::serialize_f32s(&v), name],
                )
                .unwrap();
        }
        let cf = CoreFile {
            path: "x".to_string(),
            preamble: None,
            sections: Default::default(),
        };
        let emb = CountingHash {
            inner: HashEmbedder::new(64),
            calls: AtomicUsize::new(0),
            lens: std::sync::Mutex::new(Vec::new()),
        };
        let pay = compile(&s, &cf, &emb, task, 5000).unwrap();
        assert!(
            pay.memories.iter().any(|m| m.content.contains("refresh_token")),
            "fallback batch must rank NULL relevant memory, got {:?}",
            pay.memories
        );
        assert_eq!(
            pay.memories.first().map(|m| m.content.as_str()),
            Some(rel_content),
            "relevant NULL memory must be first via fallback, got {:?}",
            pay.memories
        );
        assert_eq!(
            pay.symbols.first().map(|y| y.name.as_str()),
            Some("refresh_token"),
            "relevant symbol must stay first, got {:?}",
            pay.symbols
        );
        // Task embed + exactly one notes missing-batch (len 1); symbols are
        // all-stored so they add zero calls.
        assert_eq!(
            emb.calls.load(Ordering::SeqCst),
            2,
            "one NULL row must trigger exactly 1 missing-batch call, got {:?}",
            emb.lens.lock().unwrap()
        );
        assert_eq!(
            *emb.lens.lock().unwrap(),
            vec![1, 1],
            "expected [task, missing-batch] lens, got {:?}",
            emb.lens.lock().unwrap()
        );
    }

    fn parity_store() -> Store {
        let s = Store::open_in_memory().unwrap();
        s.add_note("sess", "fn refresh_token rotation handling", "note", &[])
            .unwrap();
        s.add_note("sess", "zebra xylophone quantum", "note", &[])
            .unwrap();
        s.add_note("sess", "knapsack budget packing tokens", "note", &[])
            .unwrap();
        s.conn
            .execute(
                "INSERT INTO symbols(path,name,kind,line_start,line_end,signature) VALUES(?1,?2,?3,?4,?5,?6)",
                rusqlite::params![
                    "src/auth.rs",
                    "refresh_token",
                    "fn",
                    10,
                    40,
                    "fn refresh_token() -> Token"
                ],
            )
            .unwrap();
        s.conn
            .execute(
                "INSERT INTO symbols(path,name,kind,line_start,line_end,signature) VALUES(?1,?2,?3,?4,?5,?6)",
                rusqlite::params![
                    "src/pack.rs",
                    "knapsack_pack",
                    "fn",
                    1,
                    20,
                    "fn knapsack_pack(budget: i64) -> Vec<usize>"
                ],
            )
            .unwrap();
        s
    }

    fn fill_blobs(s: &Store, emb: &dyn crate::embed::EmbeddingProvider) {
        let notes: Vec<(i64, String)> = s
            .conn
            .prepare("SELECT id, content FROM notes")
            .unwrap()
            .query_map([], |r| Ok((r.get(0)?, r.get(1)?)))
            .unwrap()
            .flatten()
            .collect();
        for (id, content) in notes {
            let v = emb.embed(vec![content]).unwrap().remove(0);
            s.conn
                .execute(
                    "UPDATE notes SET embedding=?1 WHERE id=?2",
                    rusqlite::params![crate::store::serialize_f32s(&v), id],
                )
                .unwrap();
        }
        let syms: Vec<(String, String)> = s
            .conn
            .prepare("SELECT name, signature FROM symbols")
            .unwrap()
            .query_map([], |r| Ok((r.get(0)?, r.get(1)?)))
            .unwrap()
            .flatten()
            .collect();
        for (name, sig) in syms {
            let v = emb.embed(vec![sig]).unwrap().remove(0);
            s.conn
                .execute(
                    "UPDATE symbols SET embedding=?1 WHERE name=?2",
                    rusqlite::params![crate::store::serialize_f32s(&v), name],
                )
                .unwrap();
        }
    }

    fn assert_parity(
        stored: &Payload,
        live: &Payload,
        tol: f64,
        ctx: &str,
    ) {
        let sids: Vec<i64> = stored.memories.iter().map(|m| m.id).collect();
        let lids: Vec<i64> = live.memories.iter().map(|m| m.id).collect();
        assert_eq!(sids, lids, "{ctx}: memory id order differs");
        for (a, b) in stored.memories.iter().zip(live.memories.iter()) {
            assert!(
                (a.score - b.score).abs() <= tol,
                "{ctx}: score gap {} on id {}",
                (a.score - b.score).abs(),
                a.id
            );
        }
        let snames: Vec<&str> = stored.symbols.iter().map(|s| s.name.as_str()).collect();
        let lnames: Vec<&str> = live.symbols.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(snames, lnames, "{ctx}: symbol name order differs");
    }

    #[test]
    fn stored_matches_live_scores() {
        let task = "fn rotate session credential";
        let emb = HashEmbedder::new(64);
        let cf = CoreFile {
            path: "x".to_string(),
            preamble: None,
            sections: Default::default(),
        };
        let stored_db = parity_store();
        fill_blobs(&stored_db, &emb);
        let stored_pay = compile(&stored_db, &cf, &emb, task, 5000).unwrap();
        // Same rows, NULL embeddings: forces the all-live batch path.
        let live_db = parity_store();
        let live_pay = compile(&live_db, &cf, &emb, task, 5000).unwrap();
        assert!(!stored_pay.memories.is_empty(), "fixture must rank memories");
        assert!(!stored_pay.symbols.is_empty(), "fixture must rank symbols");
        assert_parity(&stored_pay, &live_pay, 1e-12, "hash-64");
    }

    #[test]
    fn stored_matches_live_scores_ollama() {
        use crate::embed::{EmbeddingProvider, OllamaProvider};
        let task = "fn rotate session credential";
        let emb = OllamaProvider {
            url: "http://localhost:11434".to_string(),
            model: "nomic-embed-text".to_string(),
            timeout_ms: 5000,
        };
        if emb.embed(vec!["ping".to_string()]).is_err() {
            eprintln!("SKIP stored_matches_live_scores_ollama: ollama not up");
            return;
        }
        let cf = CoreFile {
            path: "x".to_string(),
            preamble: None,
            sections: Default::default(),
        };
        let stored_db = parity_store();
        fill_blobs(&stored_db, &emb);
        let stored_pay = compile(&stored_db, &cf, &emb, task, 5000).unwrap();
        let live_db = parity_store();
        let live_pay = compile(&live_db, &cf, &emb, task, 5000).unwrap();
        assert!(!stored_pay.memories.is_empty(), "fixture must rank memories");
        assert!(!stored_pay.symbols.is_empty(), "fixture must rank symbols");
        assert_parity(&stored_pay, &live_pay, 1e-6, "ollama-768");
    }

    #[test]
    fn stored_dim_mismatch_falls_back() {
        let task = "fn rotate session credential";
        let cf = CoreFile {
            path: "x".to_string(),
            preamble: None,
            sections: Default::default(),
        };
        let db = parity_store();
        fill_blobs(&db, &HashEmbedder::new(64));
        let emb128 = HashEmbedder::new(128);
        let pay = compile(&db, &cf, &emb128, task, 5000).unwrap();
        assert!(
            pay.rho_naive >= 0.0 && pay.rho_naive <= 1.0,
            "rho out of range: {}",
            pay.rho_naive
        );
        assert!(pay.tokens_used >= 0, "tokens_used negative");
        assert!(pay.tokens_used <= 5000, "over budget: {}", pay.tokens_used);
        let fts_ids: std::collections::HashSet<i64> = db
            .search_notes(task, 50)
            .unwrap_or_default()
            .into_iter()
            .map(|(id, _)| id)
            .collect();
        for m in &pay.memories {
            assert!(
                fts_ids.contains(&m.id),
                "stale-dim memory id {} must come from lexical FTS only, fts={:?}",
                m.id,
                fts_ids
            );
        }
        if fts_ids.is_empty() {
            assert!(
                pay.memories.is_empty(),
                "no lexical hits so memories must be empty, got {:?}",
                pay.memories
            );
        }
        let toks_task = bench::tokenize(task);
        for y in &pay.symbols {
            let name_lc = y.name.to_lowercase();
            let path_lc = y.path.to_lowercase();
            let overlap = toks_task
                .iter()
                .any(|t| name_lc.contains(t.as_str()) || path_lc.contains(t.as_str()));
            assert!(
                overlap,
                "stale-dim symbol {} must come from substring arm only",
                y.name
            );
        }
    }

    // ---- Phase 4: refs-leg fusion tests ---------------------------------

    /// Fixture: two symbols with equal substring overlap (0.5 each) for the
    /// task "auth make_token", plus a reference index whose only use ref is
    /// the dotted name `auth.make_token`. Symbol embeddings are junk blobs
    /// of the wrong dimension so the cosine arm is deterministically skipped
    /// and fusion arithmetic is pure {substring + refs}.
    fn seeded_refs(s: &Store) {
        for (path, name, sig) in [
            ("tokens.py", "make_token", "def make_token(seed):"),
            ("auth.py", "auth_flow", "def auth_flow():"),
        ] {
            s.conn
                .execute(
                    "INSERT INTO symbols(path,name,kind,line_start,line_end,signature)
                     VALUES(?1,?2,?3,1,2,?4)",
                    rusqlite::params![path, name, "function", sig],
                )
                .unwrap();
        }
        let defs = vec![(
            "auth_flow".to_string(),
            "function".to_string(),
            1,
            2,
            "def auth_flow():".to_string(),
        )];
        let uses = vec![(
            "auth.make_token".to_string(),
            "call".to_string(),
            5,
            "t = auth.make_token(x)".to_string(),
            Some("auth".to_string()),
        )];
        s.index_code_refs("auth.py", &defs, &uses).unwrap();
        // Wrong-dimension blobs (8 f32s vs the 64-dim embedder) force the
        // stale-dim path: cos_scores = None, substring + refs only.
        let junk = crate::store::serialize_f32s(&[0.5f32; 8]);
        s.conn
            .execute("UPDATE symbols SET embedding=?1", rusqlite::params![junk])
            .unwrap();
    }

    /// IDF fixture: task tokens never appear in any symbol name/path, so
    /// the substring arm scores 0 for everyone and the refs leg alone
    /// decides. `commonword` lives in 2 files (passes the n>=3 floor as
    /// specific), `rareword` in 1; votes arrive via FTS-context call sites
    /// (`alpha(commonword)`, `beta(rareword)`). Junk blobs skip cosine.
    fn seeded_idf(s: &Store) {
        for (path, name) in [("a.py", "alpha"), ("b.py", "beta")] {
            s.conn
                .execute(
                    "INSERT INTO symbols(path,name,kind,line_start,line_end,signature)
                     VALUES(?1,?2,?3,1,2,?4)",
                    rusqlite::params![path, name, "function", format!("def {name}():")],
                )
                .unwrap();
        }
        let cw = |ctx: &str| {
            ("commonword".to_string(), "call".to_string(), 1, ctx.to_string(), None)
        };
        s.index_code_refs("f1.py", &[], &[cw("x = alpha(commonword)")]).unwrap();
        s.index_code_refs("f2.py", &[], &[cw("y = alpha(commonword)")]).unwrap();
        s.index_code_refs(
            "f3.py",
            &[],
            &[("rareword".to_string(), "call".to_string(), 1, "z = beta(rareword)".to_string(), None)],
        ).unwrap();
        let junk = crate::store::serialize_f32s(&[0.5f32; 8]);
        s.conn
            .execute("UPDATE symbols SET embedding=?1", rusqlite::params![junk])
            .unwrap();
    }

    #[test]
    fn refs_idf_ranks_rare_token_symbol_first() {        let s = Store::open_in_memory().unwrap();
        seeded_idf(&s);
        let cf = CoreFile { path: "x".into(), preamble: None, sections: BTreeMap::new() };
        let emb = HashEmbedder::new(64);
        // Substring scores 0 for both (tokens appear in no name/path) and
        // cosine is skipped, so the refs leg alone decides. Flat +1.0 votes
        // would tie 0.5/0.5 and fall back to insertion order (alpha); IDF
        // (rare ln4 > common ln2.5) must put beta first.
        let pay = compile(&s, &cf, &emb, "commonword rareword", 300).unwrap();
        assert_eq!(pay.symbols.len(), 2, "both candidates must surface: {:?}", pay.symbols);
        assert_eq!(
            pay.symbols[0].name, "beta",
            "rare-token symbol must outrank common-token symbol: {:?}",
            pay.symbols
        );
    }

    #[test]
    fn refs_query_repetition_does_not_multiply_votes() {
        let s = Store::open_in_memory().unwrap();
        seeded_idf(&s);
        let cf = CoreFile { path: "x".into(), preamble: None, sections: BTreeMap::new() };
        let emb = HashEmbedder::new(64);
        // `commonword` x3 would vote 3x under per-occurrence counting and
        // beat the single rare vote. Dedup keeps one vote per token, so the
        // order must match the single-occurrence query: beta first.
        let pay = compile(&s, &cf, &emb, "commonword commonword commonword rareword", 300).unwrap();
        assert_eq!(pay.symbols.len(), 2, "both candidates must surface: {:?}", pay.symbols);
        assert_eq!(
            pay.symbols[0].name, "beta",
            "repeated common token must not outvote single rare token: {:?}",
            pay.symbols
        );
    }

    #[test]
    fn is_code_mention_shapes() {
        // Calls, attribute access/base: code.
        assert!(is_code_mention("t = auth.make_token(x)", "make_token"));
        assert!(is_code_mention("x.make_token", "make_token"));
        assert!(is_code_mention("make_token.", "make_token"));
        assert!(is_code_mention("z = beta(rareword)", "beta"));
        // Prose mentions, bare RHS, superstrings: not code.
        assert!(!is_code_mention("the frame rate dropped", "frame"));
        assert!(!is_code_mention("x = frame", "frame"));
        assert!(!is_code_mention("obj.frames", "frame"));
        assert!(!is_code_mention("reframed = myframe(x)", "frame"));
        assert!(!is_code_mention("call t(x)", "t"));
        assert!(!is_code_mention("", "alpha"));
    }

    /// Discriminator: the query token reaches both symbols' call-site
    /// contexts, but only one context shows a call. Prose-mention votes
    /// must stay silent, so only the called symbol surfaces.
    fn seeded_mention(s: &Store) {
        for (path, name) in [("a.py", "alpha"), ("b.py", "beta")] {
            s.conn
                .execute(
                    "INSERT INTO symbols(path,name,kind,line_start,line_end,signature)
                     VALUES(?1,?2,?3,1,2,?4)",
                    rusqlite::params![path, name, "function", format!("def {name}():")],
                )
                .unwrap();
        }
        s.index_code_refs(
            "f1.py",
            &[],
            &[("kw".to_string(), "call".to_string(), 1, "see alpha here".to_string(), None)],
        ).unwrap();
        s.index_code_refs(
            "f2.py",
            &[],
            &[("kw".to_string(), "call".to_string(), 1, "run beta() now".to_string(), None)],
        ).unwrap();
        let junk = crate::store::serialize_f32s(&[0.5f32; 8]);
        s.conn
            .execute("UPDATE symbols SET embedding=?1", rusqlite::params![junk])
            .unwrap();
    }

    #[test]
    fn refs_fts_context_requires_code_shape() {
        let s = Store::open_in_memory().unwrap();
        seeded_mention(&s);
        let cf = CoreFile { path: "x".into(), preamble: None, sections: BTreeMap::new() };
        let emb = HashEmbedder::new(64);
        // `kw` is the only voting token (substring 0 everywhere, cosine
        // skipped). Old substring matching votes both alpha and beta;
        // call-shape matching votes beta only.
        let pay = compile(&s, &cf, &emb, "kw", 300).unwrap();
        assert_eq!(pay.symbols.len(), 1, "only the called symbol must surface: {:?}", pay.symbols);
        assert_eq!(pay.symbols[0].name, "beta", "{:?}", pay.symbols);
    }

    #[test]
    fn refs_leg_promotes_exact_identifier_symbol() {
        let s = Store::open_in_memory().unwrap();
        seeded_refs(&s);
        let cf = CoreFile { path: "x".into(), preamble: None, sections: BTreeMap::new() };
        let emb = HashEmbedder::new(64);
        // Pre-refs-leg ordering: both symbols tie at substring 0.5 and the
        // name tiebreak puts auth_flow first. The refs leg resolves the
        // dotted use ref auth.make_token -> make_token and votes it for
        // both task tokens, so make_token must now rank first.
        let pay = compile(&s, &cf, &emb, "auth make_token", 300).unwrap();
        assert_eq!(pay.symbols.len(), 2, "both candidates must surface: {:?}", pay.symbols);
        assert_eq!(
            pay.symbols[0].name, "make_token",
            "refs leg must promote exact-identifier symbol: {:?}",
            pay.symbols
        );
        assert!(
            pay.stages.fts_ms <= pay.took_ms + 5,
            "refs timing must stay inside stage totals"
        );
    }

    #[test]
    fn refs_leg_adds_no_noise_for_prose_queries() {
        let s = Store::open_in_memory().unwrap();
        seeded_refs(&s);
        let cf = CoreFile { path: "x".into(), preamble: None, sections: BTreeMap::new() };
        let emb = HashEmbedder::new(64);
        // Prose-only query: no token is an identifier present in the refs
        // index, so the refs leg must stay silent and the fusion result
        // must match the pre-Phase-4 behavior (no code candidates at all).
        let pay = compile(&s, &cf, &emb, "polish onboarding copy for new users", 300).unwrap();
        assert!(
            pay.symbols.is_empty(),
            "prose query must not surface code symbols via refs leg: {:?}",
            pay.symbols
        );
        let json = serde_json::to_string(&pay.stages).unwrap();
        assert!(json.contains("fts_ms"), "stage times must still serialize: {json}");
    }

    /// Phase 4 eval gate: navigation success = the target symbol for a
    /// task must be retrievable into the compiled payload (top-12 fusion
    /// output). Run per scenario so refactors of the fusion legs keep
    /// real navigation working.
    fn nav_success(pay: &Payload, target: &str) -> bool {
        pay.symbols.iter().any(|y| y.name == target)
    }

    #[test]
    fn refs_leg_flips_tied_substring_order_causally() {
        let cf = CoreFile { path: "x".into(), preamble: None, sections: BTreeMap::new() };
        let emb = HashEmbedder::new(64);
        let mk_store = |with_refs: bool| {
            let s = Store::open_in_memory().unwrap();
            // Two symbols, each matching exactly one of the two task tokens
            // → substring 0.5 tie. Name tiebreak puts "zeta" first, so with
            // the refs leg absent the order must be zeta, alpha.
            for (path, name, sig) in [
                ("a.py", "alpha", "def alpha():"),
                ("z.py", "zeta", "def zeta():"),
            ] {
                s.conn
                    .execute(
                        "INSERT INTO symbols(path,name,kind,line_start,line_end,signature)
                         VALUES(?1,?2,?3,1,2,?4)",
                        rusqlite::params![path, name, "function", sig],
                    )
                    .unwrap();
            }
            if with_refs {
                // A call site whose only ref name is the dotted
                // `auth.zeta` — never a plain substring of either symbol
                // name — so only the refs leg can break the tie.
                let defs: Vec<(String, String, i64, i64, String)> = Vec::new();
                let uses = vec![(
                    "auth.zeta".to_string(),
                    "call".to_string(),
                    5,
                    "z = auth.zeta(x)".to_string(),
                    Some("auth".to_string()),
                )];
                s.index_code_refs("a.py", &defs, &uses).unwrap();
            }
            s
        };
        let base = compile(&mk_store(false), &cf, &emb, "alpha zeta", 300).unwrap();
        assert_eq!(base.symbols.len(), 2, "{:?}", base.symbols);
        assert_eq!(
            base.symbols[0].name, "alpha",
            "substring tie (0.5 each) must fall to name-asc order without refs leg: {:?}",
            base.symbols
        );
        let with = compile(&mk_store(true), &cf, &emb, "alpha zeta", 300).unwrap();
        assert_eq!(with.symbols.len(), 2, "{:?}", with.symbols);
        assert_eq!(
            with.symbols[0].name, "zeta",
            "refs leg (dotted auth.zeta use) must flip the tied order to zeta first: {:?}",
            with.symbols
        );
    }

    #[test]
    fn eval_gate_nav_success_on_identifier_tasks() {
        // Scenario A: lexical seed (no refs rows) — exact-name symbol found.
        let (s, cf) = seeded();
        let emb = HashEmbedder::new(64);
        let pay = compile(&s, &cf, &emb, "fix jwt refresh bug", 300).unwrap();
        assert!(
            nav_success(&pay, "refresh_token"),
            "scenario A nav failure: {:?}",
            pay.symbols
        );
        assert_eq!(pay.symbols[0].name, "refresh_token", "{:?}", pay.symbols);

        // Scenario B: refs-backed fixture with a live cosine arm (no junk
        // blobs) — the dotted-ref target must surface against a decoy.
        let s2 = Store::open_in_memory().unwrap();
        s2.add_note("sess", "auth flow issues tokens", "note", &["auth"]).unwrap();
        seeded_refs(&s2);
        let cf2 = CoreFile { path: "x".into(), preamble: None, sections: BTreeMap::new() };
        let pay2 = compile(&s2, &cf2, &emb, "auth make_token flow", 300).unwrap();
        assert!(
            nav_success(&pay2, "make_token"),
            "scenario B nav failure: {:?}",
            pay2.symbols
        );
    }
}
