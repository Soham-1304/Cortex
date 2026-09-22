//! Agentic-loop A/B harness: "normal agentic loop" vs "same loop with Cortex memory".
//!
//! There is no live LLM in the loop. The agent is simulated deterministically as
//! the search-read-learn cycle every coding agent performs:
//!   1. search (BM25 over raw docs for baseline; Cortex `compile()` for the
//!      memory arm),
//!   2. read whatever fits in the per-turn context budget (paying those tokens),
//!   3. learn identifier-like tokens (t7, s3, d7) from what was read,
//!   4. re-search, until the task's gold doc is exposed or turns run out.
//!
//! Both arms run the SAME tasks over the SAME synthetic repo and are compared
//! paired per task (VALIDATION.md §9): token reduction with bootstrap 95% CI,
//! paired t, Cohen's d. Success = the gold doc's marker token appears in the
//! context the agent paid for.
//!
//! FAIRNESS: the Cortex arm pays a one-time memory-seeding cost (the notes a
//! previous session would have left). It is reported separately AND amortized
//! per task, so the headline reduction never hides the write cost.
//!
//! The synthetic repo encodes the real value proposition: a symptom (failing
//! test `s{j}`) points at its cause (`t{g}`) only from inside one doc and —
//! once learned — only in memory. A memoryless agent must re-discover that
//! mapping by exploration every task; a memory-backed agent gets the
//! compressed pointer from `compile_context` on turn 1.

use crate::Result;
use crate::compile::{compile, Payload};
use crate::corefile::CoreFile;
use crate::embed::{EmbeddingProvider, HashEmbedder};
use crate::session::Session;
use crate::store::Store;
use bench::bm25::Bm25Index;
use bench::eval::bootstrap_ci;
use bench::{count_tokens, tokenize, Rng};
use serde::Serialize;
use std::collections::HashSet;

// ---------------------------------------------------------------------------
// Synthetic project generation (seeded, deterministic)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct SimDoc {
    pub id: String,
    pub path: String,
    pub text: String,
}

#[derive(Debug, Clone)]
pub struct SimTask {
    pub id: String,
    pub symptom: String,
    pub task: String,
    pub gold_doc: String,
    pub gold_topic: String,
}

#[derive(Debug, Clone)]
pub struct SimProject {
    pub docs: Vec<SimDoc>,
    pub tasks: Vec<SimTask>,
}

const ROLES: &[&str] = &["auth", "storage", "network", "parsing", "scheduling", "rendering"];
const FILLERS: &[&str] = &[
    "parser", "buffer", "worker", "queue", "cache", "retry", "timeout", "handler", "encoder",
    "decoder", "router", "stream", "writer", "reader", "pool", "registry", "factory", "adapter",
];

/// Identifier-like token: one lowercase letter followed by digits (t7, s3, d12).
pub fn is_ident(tok: &str) -> bool {
    let b = tok.as_bytes();
    b.len() >= 2 && b[0].is_ascii_lowercase() && b[1..].iter().all(|c| c.is_ascii_digit())
}

pub fn generate(docs_n: usize, tasks_n: usize, seed: u64, distractors: usize) -> Result<SimProject> {
    if docs_n < 2 {
        return Err(crate::Error::Cli("eval needs docs >= 2".into()));
    }
    if tasks_n == 0 {
        return Err(crate::Error::Cli("eval needs tasks >= 1".into()));
    }
    if tasks_n > docs_n {
        return Err(crate::Error::Cli("eval needs tasks <= docs".into()));
    }
    let mut rng = Rng::new(seed);

    // Distinct gold docs via partial Fisher-Yates.
    let mut order: Vec<usize> = (0..docs_n).collect();
    for i in 0..tasks_n {
        let j = i + (rng.next_u64() as usize) % (docs_n - i);
        order.swap(i, j);
    }
    let golds: Vec<usize> = order[..tasks_n].to_vec();

    let mut docs: Vec<SimDoc> = Vec::with_capacity(docs_n);
    for i in 0..docs_n {
        let role = ROLES[(rng.next_u64() as usize) % ROLES.len()];
        let mut text = format!("module t{} implements {} features", i, role);
        for _ in 0..8 {
            text.push(' ');
            text.push_str(FILLERS[(rng.next_u64() as usize) % FILLERS.len()]);
        }
        text.push_str(&format!(" file d{}", i));
        docs.push(SimDoc {
            id: format!("d{}", i),
            path: format!("src/d{}.rs", i),
            text,
        });
    }

    struct Spec {
        g: usize,
        p: usize,
        symptom: String,
    }
    let mut specs: Vec<Spec> = Vec::with_capacity(tasks_n);
    for (j, &g) in golds.iter().enumerate() {
        // Symptom doc p != g: the failing test lives elsewhere from the fix.
        let p = if g + 1 < docs_n { g + 1 } else { g - 1 };
        specs.push(Spec { g, p, symptom: format!("s{}", j) });
    }

    for s in &specs {
        // Symptom doc: heavy symptom mentions + the cause pointer (the only doc
        // in the repo linking symptom -> fix location, besides memory).
        docs[s.p].text.push_str(&format!(
            " failing test {} {} {} root cause t{}",
            s.symptom, s.symptom, s.symptom, s.g
        ));
        // The fix location does NOT mention the symptom string — a memoryless
        // agent cannot jump there without first reading the symptom doc.
    }
    for s in &specs {
        // Distractor docs that merely mention the symptom: search noise the
        // baseline must chew through within its per-turn budget.
        let mut added = 0usize;
        let mut guard = 0usize;
        while added < distractors && guard < docs_n * 4 {
            guard += 1;
            let di = (rng.next_u64() as usize) % docs_n;
            if di == s.g || di == s.p || docs[di].text.contains(&s.symptom) {
                continue;
            }
            docs[di].text.push_str(&format!(" mentions {}", s.symptom));
            added += 1;
        }
    }

    let tasks: Vec<SimTask> = specs
        .iter()
        .enumerate()
        .map(|(j, s)| SimTask {
            id: format!("task-{}", j),
            symptom: s.symptom.clone(),
            task: format!("fix failing {} then verify behavior", s.symptom),
            gold_doc: format!("d{}", s.g),
            gold_topic: format!("t{}", s.g),
        })
        .collect();

    Ok(SimProject { docs, tasks })
}

// ---------------------------------------------------------------------------
// Baseline arm: normal agentic loop (no memory)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize)]
pub struct BaselineRun {
    pub tokens: i64,
    pub turns: usize,
    pub success: bool,
}

/// Simulated memoryless agent: BM25 search -> read docs that fit the per-turn
/// budget (paying full doc tokens) -> learn identifiers from what it read ->
/// re-search. Context does not persist between turns beyond the learned query
/// terms; nothing persists between tasks.
pub fn baseline_loop(
    index: &Bm25Index,
    docs: &[SimDoc],
    task: &SimTask,
    per_turn: i64,
    max_turns: usize,
) -> BaselineRun {
    let base_toks: Vec<String> = tokenize(&task.task);
    let mut learned: HashSet<String> = HashSet::new();
    for t in &base_toks {
        if is_ident(t) {
            learned.insert(t.clone());
        }
    }
    let mut read: HashSet<usize> = HashSet::new();
    let mut query = task.task.clone();
    let mut tokens = 0i64;
    let mut turns = 0usize;
    let mut success = false;

    for turn in 0..max_turns {
        turns += 1;
        let ranking = index.ranking(&query);
        let mut spent = 0i64;
        let mut newly_read: Vec<usize> = Vec::new();
        for &di in &ranking {
            if read.contains(&di) {
                continue;
            }
            let c = count_tokens(&docs[di].text) as i64;
            if spent + c > per_turn {
                continue; // too big for what's left this turn; try smaller docs
            }
            read.insert(di);
            spent += c;
            tokens += c;
            newly_read.push(di);
            if docs[di].id == task.gold_doc {
                success = true;
                break;
            }
        }
        if success || newly_read.is_empty() || turn + 1 == max_turns {
            break;
        }
        let mut new_info = false;
        for di in &newly_read {
            for t in tokenize(&docs[*di].text) {
                if is_ident(&t) && learned.insert(t.clone()) {
                    new_info = true;
                }
            }
        }
        if !new_info {
            break; // exploration exhausted: nothing new to search for
        }
        let mut q = base_toks.clone();
        q.extend(learned.iter().cloned());
        query = q.join(" ");
    }

    BaselineRun { tokens, turns, success }
}

// ---------------------------------------------------------------------------
// Cortex arm: same loop, memory-backed
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize)]
pub struct CortexRun {
    pub tokens: i64,
    pub turns: usize,
    pub success: bool,
    pub mean_rho: f64,
}

/// Exact-token containment (no substring false positives: "d7" must not match
/// "d70").
pub fn tokens_contain(text: &str, tok: &str) -> bool {
    tokenize(text).iter().any(|t| t == tok)
}

/// Success = gold doc marker exposed in the compiled context.
pub fn cortex_hit(payload: &Payload, gold_doc: &str) -> bool {
    payload
        .memories
        .iter()
        .any(|m| tokens_contain(&m.content, gold_doc))
        || payload
            .symbols
            .iter()
            .any(|s| tokens_contain(&s.path, gold_doc))
}

/// Memory-backed agent: `compile()` under the same per-turn budget, same
/// learn-and-retry loop. The compiled payload IS the context the agent pays for.
pub fn cortex_loop(
    store: &Store,
    core: &CoreFile,
    emb: &dyn EmbeddingProvider,
    task: &SimTask,
    budget: i64,
    max_turns: usize,
) -> Result<CortexRun> {
    let base_toks: Vec<String> = tokenize(&task.task);
    let mut learned: HashSet<String> = HashSet::new();
    for t in &base_toks {
        if is_ident(t) {
            learned.insert(t.clone());
        }
    }
    let mut tokens = 0i64;
    let mut rho_sum = 0.0;
    let mut turns = 0usize;

    for turn in 0..max_turns {
        turns += 1;
        let mut q = base_toks.clone();
        q.extend(learned.iter().cloned());
        let payload = compile(store, core, emb, &q.join(" "), budget)?;
        tokens += payload.tokens_used;
        rho_sum += payload.rho_naive;
        if cortex_hit(&payload, &task.gold_doc) {
            return Ok(CortexRun {
                tokens,
                turns,
                success: true,
                mean_rho: rho_sum / turns as f64,
            });
        }
        if turn + 1 == max_turns {
            break;
        }
        let mut new_info = false;
        for m in &payload.memories {
            for t in tokenize(&m.content) {
                if is_ident(&t) && learned.insert(t.clone()) {
                    new_info = true;
                }
            }
        }
        for s in &payload.symbols {
            let txt = format!("{} {}", s.path, s.signature.clone().unwrap_or_default());
            for t in tokenize(&txt) {
                if is_ident(&t) && learned.insert(t.clone()) {
                    new_info = true;
                }
            }
        }
        if !new_info {
            break;
        }
    }

    Ok(CortexRun {
        tokens,
        turns,
        success: false,
        mean_rho: if turns > 0 { rho_sum / turns as f64 } else { 0.0 },
    })
}

/// One-time memory seeding: the notes + symbols a previous session would have
/// left behind (content = compressed symptom -> cause -> fix pointer). Returns
/// the total token cost of seeding so the A/B never hides it.
pub fn seed_memory(store: &Store, project: &SimProject, emb: &dyn EmbeddingProvider) -> Result<i64> {
    let mut sess = Session::start(store, "eval-seed")?;
    let mut seed_tokens = 0i64;
    for t in &project.tasks {
        // Compressed memory note: identifiers only — exactly what a digest of
        // a past session leaves behind.
        let content = format!("{} {} {}", t.symptom, t.gold_topic, t.gold_doc);
        sess.write(
            &content,
            "decision",
            &[t.symptom.clone(), t.gold_topic.clone(), t.gold_doc.clone()],
            &[(t.symptom.as_str(), t.gold_topic.as_str())],
            emb,
        )?;
        seed_tokens += count_tokens(&content);
        store.conn.execute(
            "INSERT OR REPLACE INTO symbols(path,name,kind,line_start,line_end,signature)
             VALUES(?1,?2,'fn',1,1,?3)",
            rusqlite::params![
                format!("src/{}.rs", t.gold_doc),
                t.gold_topic,
                format!("{} {}", t.gold_topic, t.symptom)
            ],
        )?;
    }
    Ok(seed_tokens)
}

// ---------------------------------------------------------------------------
// Harness + paired statistics (VALIDATION.md §9)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
pub struct HarnessConfig {
    pub docs: usize,
    pub tasks: usize,
    pub seed: u64,
    pub budget: i64,   // cortex per-turn compile budget
    pub per_turn: i64, // baseline per-turn read budget
    pub max_turns: usize,
    pub distractors: usize,
}

impl Default for HarnessConfig {
    fn default() -> Self {
        HarnessConfig {
            docs: 48,
            tasks: 24,
            seed: 42,
            budget: 2000,
            per_turn: 120,
            max_turns: 8,
            distractors: 3,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct TaskOutcome {
    pub id: String,
    pub baseline_tokens: i64,
    pub cortex_tokens: i64,
    pub baseline_turns: usize,
    pub cortex_turns: usize,
    pub baseline_success: bool,
    pub cortex_success: bool,
    pub reduction: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct ArmSummary {
    pub success_rate: f64,
    pub mean_tokens: f64,
    pub mean_turns: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct HarnessReport {
    pub docs: usize,
    pub tasks: usize,
    pub seed: u64,
    pub budget: i64,
    pub per_turn: i64,
    pub max_turns: usize,
    pub distractors: usize,
    pub baseline: ArmSummary,
    pub cortex: ArmSummary,
    pub seed_tokens_total: i64,
    pub seed_tokens_amortized: f64,
    pub cortex_tokens_with_seed: f64,
    pub mean_reduction: f64,
    pub reduction_ci_lo: f64,
    pub reduction_ci_hi: f64,
    pub paired_t: f64,
    pub cohens_d: f64,
    pub mean_rho_cortex: f64,
    pub outcomes: Vec<TaskOutcome>,
}

pub fn mean(v: &[f64]) -> f64 {
    if v.is_empty() {
        0.0
    } else {
        v.iter().sum::<f64>() / v.len() as f64
    }
}

/// Paired t statistic: t = mean / (sd / sqrt(n)); sd is the sample sd of the
/// differences. Degenerate (constant) differences -> ±inf by sign, 0 if all 0.
pub fn paired_t(reductions: &[f64]) -> f64 {
    let n = reductions.len();
    if n < 2 {
        return 0.0;
    }
    let m = mean(reductions);
    let var = reductions.iter().map(|r| (r - m) * (r - m)).sum::<f64>() / (n - 1) as f64;
    let sd = var.sqrt();
    if sd < 1e-12 {
        if m > 0.0 {
            f64::INFINITY
        } else if m < 0.0 {
            f64::NEG_INFINITY
        } else {
            0.0
        }
    } else {
        m / (sd / (n as f64).sqrt())
    }
}

/// Cohen's d for paired samples: mean difference / sample sd of differences.
pub fn cohens_d(reductions: &[f64]) -> f64 {
    let n = reductions.len();
    if n < 2 {
        return 0.0;
    }
    let m = mean(reductions);
    let var = reductions.iter().map(|r| (r - m) * (r - m)).sum::<f64>() / (n - 1) as f64;
    let sd = var.sqrt();
    if sd < 1e-12 {
        if m > 0.0 {
            f64::INFINITY
        } else if m < 0.0 {
            f64::NEG_INFINITY
        } else {
            0.0
        }
    } else {
        m / sd
    }
}

fn summarize(runs: &[(i64, usize, bool)]) -> ArmSummary {
    let n = runs.len().max(1) as f64;
    ArmSummary {
        success_rate: runs.iter().filter(|(_, _, s)| *s).count() as f64 / n,
        mean_tokens: runs.iter().map(|(t, _, _)| *t as f64).sum::<f64>() / n,
        mean_turns: runs.iter().map(|(_, u, _)| *u as f64).sum::<f64>() / n,
    }
}

pub fn run_harness(cfg: &HarnessConfig) -> Result<HarnessReport> {
    let project = generate(cfg.docs, cfg.tasks, cfg.seed, cfg.distractors)?;

    // ---- Cortex arm: seed memory once (previous-session knowledge) ----
    let store = Store::open_in_memory()?;
    let emb = HashEmbedder::new(64);
    let seed_tokens_total = seed_memory(&store, &project, &emb)?;
    let core = CoreFile {
        path: "eval".to_string(),
        preamble: None,
        sections: Default::default(),
    };

    // ---- Baseline arm: BM25 over the raw repo ----
    let texts: Vec<String> = project.docs.iter().map(|d| d.text.clone()).collect();
    let index = Bm25Index::build(&texts);

    let mut outcomes: Vec<TaskOutcome> = Vec::with_capacity(cfg.tasks);
    let mut baseline_runs: Vec<(i64, usize, bool)> = Vec::with_capacity(cfg.tasks);
    let mut cortex_runs: Vec<(i64, usize, bool)> = Vec::with_capacity(cfg.tasks);
    let mut rhos: Vec<f64> = Vec::with_capacity(cfg.tasks);

    for task in &project.tasks {
        let b = baseline_loop(&index, &project.docs, task, cfg.per_turn, cfg.max_turns);
        let c = cortex_loop(&store, &core, &emb, task, cfg.budget, cfg.max_turns)?;
        rhos.push(c.mean_rho);
        let reduction = if b.tokens > 0 {
            (b.tokens - c.tokens) as f64 / b.tokens as f64
        } else {
            0.0
        };
        baseline_runs.push((b.tokens, b.turns, b.success));
        cortex_runs.push((c.tokens, c.turns, c.success));
        outcomes.push(TaskOutcome {
            id: task.id.clone(),
            baseline_tokens: b.tokens,
            cortex_tokens: c.tokens,
            baseline_turns: b.turns,
            cortex_turns: c.turns,
            baseline_success: b.success,
            cortex_success: c.success,
            reduction,
        });
    }

    let reductions: Vec<f64> = outcomes.iter().map(|o| o.reduction).collect();
    let (lo, hi) = bootstrap_ci(&reductions, 10_000, 42);
    let n = cfg.tasks as f64;
    let cortex_total: f64 = cortex_runs.iter().map(|(t, _, _)| *t as f64).sum();
    let seed_amortized = seed_tokens_total as f64 / n;

    Ok(HarnessReport {
        docs: cfg.docs,
        tasks: cfg.tasks,
        seed: cfg.seed,
        budget: cfg.budget,
        per_turn: cfg.per_turn,
        max_turns: cfg.max_turns,
        distractors: cfg.distractors,
        baseline: summarize(&baseline_runs),
        cortex: summarize(&cortex_runs),
        seed_tokens_total,
        seed_tokens_amortized: seed_amortized,
        cortex_tokens_with_seed: cortex_total / n + seed_amortized,
        mean_reduction: mean(&reductions),
        reduction_ci_lo: lo,
        reduction_ci_hi: hi,
        paired_t: paired_t(&reductions),
        cohens_d: cohens_d(&reductions),
        mean_rho_cortex: mean(&rhos),
        outcomes,
    })
}

/// Human-readable report (mirrors bench/src/main.rs report style).
pub fn print_report(r: &HarnessReport) {
    println!("=== cortex eval — agentic loop A/B (seed {}) ===", r.seed);
    println!(
        "docs={} tasks={} budget={} per_turn={} max_turns={} distractors={}",
        r.docs, r.tasks, r.budget, r.per_turn, r.max_turns, r.distractors
    );
    println!(
        "Baseline (no memory)  success {:>5.1}%  tokens/task {:>8.1}  turns/task {:.2}",
        r.baseline.success_rate * 100.0,
        r.baseline.mean_tokens,
        r.baseline.mean_turns
    );
    println!(
        "Cortex (memory)       success {:>5.1}%  tokens/task {:>8.1}  turns/task {:.2}  mean rho {:.3}",
        r.cortex.success_rate * 100.0,
        r.cortex.mean_tokens,
        r.cortex.mean_turns,
        r.mean_rho_cortex
    );
    println!(
        "Memory seeding (one-time): {} tokens total, {:.1} amortized/task",
        r.seed_tokens_total, r.seed_tokens_amortized
    );
    println!(
        "Token reduction (paired, n={}): {:.1}% [95% CI {:.1}% - {:.1}%]",
        r.tasks,
        r.mean_reduction * 100.0,
        r.reduction_ci_lo * 100.0,
        r.reduction_ci_hi * 100.0
    );
    println!(
        "  paired t={:.2}  Cohen's d={:.2}  incl. amortized seed: {:.1} tokens/task vs {:.1} baseline",
        r.paired_t,
        r.cohens_d,
        r.cortex_tokens_with_seed,
        r.baseline.mean_tokens
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compile::{MemoryHit, SymbolHit};

    #[test]
    fn ident_detector_basics() {
        assert!(is_ident("t7"));
        assert!(is_ident("s3"));
        assert!(is_ident("d12"));
        assert!(!is_ident("fix"));
        assert!(!is_ident("7t"));
        assert!(!is_ident("t"));
        assert!(!is_ident(""));
        assert!(!is_ident("topic_7"));
    }

    #[test]
    fn generator_is_deterministic_same_seed() {
        let a = generate(20, 6, 99, 2).unwrap();
        let b = generate(20, 6, 99, 2).unwrap();
        assert_eq!(a.docs.len(), b.docs.len());
        for (da, db) in a.docs.iter().zip(b.docs.iter()) {
            assert_eq!(da.text, db.text, "same seed must reproduce same corpus");
        }
        assert_eq!(a.tasks.len(), b.tasks.len());
        let c = generate(20, 6, 100, 2).unwrap();
        assert_ne!(a.docs[0].text, c.docs[0].text, "different seed should differ");
    }

    #[test]
    fn generator_shapes_and_constraints() {
        let p = generate(16, 5, 7, 3).unwrap();
        assert_eq!(p.docs.len(), 16);
        assert_eq!(p.tasks.len(), 5);
        let ids: HashSet<&str> = p.docs.iter().map(|d| d.id.as_str()).collect();
        let mut golds: HashSet<&str> = HashSet::new();
        for t in &p.tasks {
            assert!(ids.contains(t.gold_doc.as_str()), "gold doc must exist");
            assert!(
                tokens_contain(&t.task, &t.symptom),
                "task text must contain the symptom"
            );
            golds.insert(t.gold_doc.as_str());
        }
        assert_eq!(golds.len(), 5, "gold docs must be distinct across tasks");
        // symptom doc must differ from gold doc: gold text never contains its symptom
        for t in &p.tasks {
            let g = p.docs.iter().find(|d| d.id == t.gold_doc).unwrap();
            assert!(
                !tokens_contain(&g.text, &t.symptom),
                "gold must not leak the symptom token"
            );
        }
    }

    #[test]
    fn generate_rejects_degenerate_configs() {
        assert!(generate(1, 1, 1, 0).is_err());
        assert!(generate(4, 0, 1, 0).is_err());
        assert!(generate(4, 9, 1, 0).is_err());
    }

    #[test]
    fn tokens_contain_is_exact_token_not_substring() {
        assert!(tokens_contain("fix implemented in d7", "d7"));
        assert!(!tokens_contain("fix implemented in d70", "d7"));
        assert!(!tokens_contain("", "d7"));
    }

    #[test]
    fn cortex_hit_detects_marker_in_memories_and_symbols() {
        let payload = Payload {
            core: None,
            memories: vec![MemoryHit {
                id: 1,
                content: "s3 t7 d7".into(),
                score: 0.5,
            }],
            symbols: vec![],
            tokens_used: 3,
            rho_naive: 0.0,
            took_ms: 0,
            stages: crate::compile::StageTimesSer { fts_ms: 0, vec_ms: 0, graph_ms: 0, pack_ms: 0 },
        };
        assert!(cortex_hit(&payload, "d7"));
        let sym_only = Payload {
            core: None,
            memories: vec![],
            symbols: vec![SymbolHit {
                name: "t7".into(),
                path: "src/d7.rs".into(),
                line_start: 1,
                line_end: 2,
                signature: None,
            }],
            tokens_used: 3,
            rho_naive: 0.0,
            took_ms: 0,
            stages: crate::compile::StageTimesSer { fts_ms: 0, vec_ms: 0, graph_ms: 0, pack_ms: 0 },
        };
        assert!(cortex_hit(&sym_only, "d7"));
        let miss = Payload {
            core: None,
            memories: vec![MemoryHit {
                id: 2,
                content: "unrelated noise".into(),
                score: 0.1,
            }],
            symbols: vec![],
            tokens_used: 2,
            rho_naive: 0.0,
            took_ms: 0,
            stages: crate::compile::StageTimesSer { fts_ms: 0, vec_ms: 0, graph_ms: 0, pack_ms: 0 },
        };
        assert!(!cortex_hit(&miss, "d7"));
    }

    #[test]
    fn paired_t_and_cohens_d_hand_case() {
        let r = vec![0.2, 0.4, 0.6];
        let m = 0.4;
        let sd = ((0.2f64 * 0.2 + 0.0 + 0.2 * 0.2) / 2.0).sqrt(); // 0.2
        assert!((paired_t(&r) - m / (sd / 3f64.sqrt())).abs() < 1e-12);
        assert!((cohens_d(&r) - 2.0).abs() < 1e-12);
        // degenerate: constant nonzero -> infinite t/d by sign
        assert_eq!(paired_t(&[0.5; 5]), f64::INFINITY);
        assert_eq!(cohens_d(&[0.5; 5]), f64::INFINITY);
        assert_eq!(paired_t(&[0.0; 5]), 0.0);
        assert_eq!(paired_t(&[0.3]), 0.0);
        assert_eq!(cohens_d(&[0.3]), 0.0);
        // negative direction
        assert_eq!(paired_t(&[-0.5; 4]), f64::NEG_INFINITY);
    }

    #[test]
    fn seed_memory_persists_notes_and_symbols() {
        let store = Store::open_in_memory().unwrap();
        let p = generate(10, 4, 5, 2).unwrap();
        let emb = HashEmbedder::new(64);
        let seed_tokens = seed_memory(&store, &p, &emb).unwrap();
        assert!(seed_tokens > 0);
        let notes: i64 = store
            .conn
            .query_row("SELECT count(*) FROM notes", [], |r| r.get(0))
            .unwrap();
        assert_eq!(notes, 4, "one memory note per task");
        let syms: i64 = store
            .conn
            .query_row("SELECT count(*) FROM symbols", [], |r| r.get(0))
            .unwrap();
        assert_eq!(syms, 4, "one symbol pointer per task");
        let edges: i64 = store
            .conn
            .query_row("SELECT count(*) FROM edges", [], |r| r.get(0))
            .unwrap();
        assert!(edges >= 4, "symptom->cause links recorded");
    }

    #[test]
    fn cortex_seeded_solves_first_turn_within_budget() {
        let store = Store::open_in_memory().unwrap();
        let p = generate(20, 8, 11, 3).unwrap();
        let emb = HashEmbedder::new(64);
        seed_memory(&store, &p, &emb).unwrap();
        let core = CoreFile {
            path: "eval".into(),
            preamble: None,
            sections: Default::default(),
        };
        for t in &p.tasks {
            let run = cortex_loop(&store, &core, &emb, t, 2000, 8).unwrap();
            assert!(run.success, "task {} must hit gold with memory", t.id);
            assert_eq!(run.turns, 1, "task {} should resolve on turn 1", t.id);
            assert!(run.tokens <= 2000, "task {} over budget: {}", t.id, run.tokens);
        }
    }

    #[test]
    fn baseline_loop_terminates_and_finds_gold_eventually() {
        let p = generate(12, 4, 3, 2).unwrap();
        let texts: Vec<String> = p.docs.iter().map(|d| d.text.clone()).collect();
        let index = Bm25Index::build(&texts);
        for t in &p.tasks {
            let run = baseline_loop(&index, &p.docs, t, 60, 64);
            assert!(run.success, "baseline must eventually find gold {}", t.id);
            assert!(run.tokens > 0);
            assert!(run.turns <= 64);
        }
    }

    #[test]
    fn baseline_never_exceeds_per_turn_budget_total() {
        // With one huge (~501 token) doc and per_turn=120, the loop must skip
        // docs that don't fit the remaining budget: no turn's spend can exceed
        // per_turn, so total tokens stay under per_turn * turns.
        let p = generate(8, 2, 13, 1).unwrap();
        let mut docs = p.docs.clone();
        docs[0].text = format!("{} huge", "filler ".repeat(500));
        let texts: Vec<String> = docs.iter().map(|d| d.text.clone()).collect();
        let index = Bm25Index::build(&texts);
        let run = baseline_loop(&index, &docs, &p.tasks[0], 120, 64);
        assert!(run.tokens > 0);
        assert!(run.tokens <= 120 * run.turns as i64, "per-turn budget violated");
    }

    #[test]
    fn harness_end_to_end_report_and_json() {
        let cfg = HarnessConfig {
            docs: 24,
            tasks: 8,
            seed: 7,
            budget: 2000,
            per_turn: 120,
            max_turns: 8,
            distractors: 3,
        };
        let report = run_harness(&cfg).unwrap();
        assert_eq!(report.outcomes.len(), 8);
        assert!(report.baseline.mean_tokens > 0.0);
        assert!(report.cortex.mean_tokens > 0.0);
        assert_eq!(report.baseline.success_rate, 1.0, "baseline solves all in <=8 turns");
        assert_eq!(report.cortex.success_rate, 1.0, "cortex solves all in <=8 turns");
        for o in &report.outcomes {
            assert!(o.baseline_tokens > 0);
            assert!(o.cortex_tokens > 0);
            assert!(
                (o.reduction
                    - (o.baseline_tokens - o.cortex_tokens) as f64 / o.baseline_tokens as f64)
                    .abs()
                    < 1e-12
            );
        }
        assert!(report.reduction_ci_lo <= report.mean_reduction + 1e-9);
        assert!(report.mean_reduction <= report.reduction_ci_hi + 1e-9);
        let json = serde_json::to_string(&report).unwrap();
        assert!(json.contains("\"mean_reduction\""));
        assert!(json.contains("\"paired_t\""));
        assert!(json.contains("\"cohens_d\""));
        assert!(json.contains("\"seed_tokens_amortized\""));
    }

    #[test]
    fn cortex_uses_fewer_tokens_and_turns_than_baseline() {
        let cfg = HarnessConfig {
            docs: 32,
            tasks: 12,
            seed: 42,
            budget: 2000,
            per_turn: 120,
            max_turns: 8,
            distractors: 3,
        };
        let report = run_harness(&cfg).unwrap();
        assert!(
            report.cortex_tokens_with_seed < report.baseline.mean_tokens,
            "cortex incl. amortized seed ({:.1}) must beat baseline ({:.1})",
            report.cortex_tokens_with_seed,
            report.baseline.mean_tokens
        );
        assert!(report.mean_reduction > 0.0, "reduction must be positive");
        assert!(report.paired_t > 0.0, "paired t must favor cortex");
        assert!(report.cortex.mean_turns < report.baseline.mean_turns);
    }

    #[test]
    fn harness_rejects_bad_configs() {
        let bad = HarnessConfig {
            tasks: 0,
            ..HarnessConfig::default()
        };
        assert!(run_harness(&bad).is_err());
        let bad2 = HarnessConfig {
            docs: 4,
            tasks: 9,
            ..HarnessConfig::default()
        };
        assert!(run_harness(&bad2).is_err());
    }
}








