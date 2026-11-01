use rusqlite::{params, Connection};
use crate::Result;

pub struct Store { pub conn: Connection }

/// Code reference lookup result: definition hits (path, line, kind),
/// use hits (path, line, kind, context), FTS fuzzy hits (path, line, ctx).
pub type RefDefs = Vec<(String, i64, String)>;
pub type RefUses = Vec<(String, i64, String, String)>;
pub type RefFts = Vec<(String, i64, String)>;

const SCHEMA: &str = r#"
PRAGMA journal_mode=WAL;
-- negative = KiB; caps page cache ≈8MB per spec §10 footprint budget
PRAGMA cache_size=-8000;
CREATE TABLE IF NOT EXISTS notes(
  id INTEGER PRIMARY KEY, session_id TEXT NOT NULL, content TEXT NOT NULL,
  kind TEXT NOT NULL, tags TEXT NOT NULL DEFAULT '[]', created_at INTEGER NOT NULL,
  embedding BLOB);
CREATE TABLE IF NOT EXISTS entities(
  id INTEGER PRIMARY KEY, name TEXT UNIQUE NOT NULL, kind TEXT NOT NULL,
  weight REAL NOT NULL DEFAULT 1.0, last_seen INTEGER NOT NULL, embedding BLOB);
CREATE TABLE IF NOT EXISTS edges(
  src INTEGER NOT NULL, dst INTEGER NOT NULL, kind TEXT NOT NULL,
  weight REAL NOT NULL DEFAULT 1.0, co_mentions INTEGER NOT NULL DEFAULT 1,
  last_seen INTEGER NOT NULL, PRIMARY KEY(src,dst,kind));
CREATE TABLE IF NOT EXISTS sessions(
  id TEXT PRIMARY KEY, started_at INTEGER NOT NULL, ended_at INTEGER,
  summary TEXT, tokens_used INTEGER NOT NULL DEFAULT 0);
CREATE TABLE IF NOT EXISTS symbols(
  id INTEGER PRIMARY KEY, path TEXT NOT NULL, name TEXT NOT NULL, kind TEXT NOT NULL,
  line_start INTEGER NOT NULL, line_end INTEGER NOT NULL, signature TEXT,
  embed_text TEXT, embedding BLOB, UNIQUE(path,name,line_start));
CREATE TABLE IF NOT EXISTS indexed_files(
  path TEXT PRIMARY KEY, mtime INTEGER NOT NULL, content_hash TEXT NOT NULL,
  sym_count INTEGER NOT NULL DEFAULT 0);
-- embed_text: the exact string that was embedded at index time (full body
-- for Python, signature for regex langs). Reindex reads it back so vectors
-- never silently downgrade to signature-only. Backfilled by migration.
CREATE INDEX IF NOT EXISTS idx_notes_session ON notes(session_id);
CREATE VIRTUAL TABLE IF NOT EXISTS notes_fts USING fts5(content, tags, content='notes', content_rowid='id');
-- Code reference index (Phase 2): word -> (file, line, context).
-- code_refs_def: one row per definition (function/class/method symbol).
-- code_refs_use: one row per call-site / import occurrence (many per file).
CREATE TABLE IF NOT EXISTS code_refs_def(
  id INTEGER PRIMARY KEY, path TEXT NOT NULL, name TEXT NOT NULL,
  kind TEXT NOT NULL, line_start INTEGER NOT NULL, line_end INTEGER NOT NULL,
  signature TEXT, UNIQUE(path,name,line_start));
CREATE TABLE IF NOT EXISTS code_refs_use(
  id INTEGER PRIMARY KEY, path TEXT NOT NULL, name TEXT NOT NULL,
  kind TEXT NOT NULL, line INTEGER NOT NULL, context TEXT NOT NULL,
  target TEXT);
CREATE INDEX IF NOT EXISTS idx_refs_def_name ON code_refs_def(name COLLATE NOCASE);
CREATE INDEX IF NOT EXISTS idx_refs_def_path ON code_refs_def(path);
CREATE INDEX IF NOT EXISTS idx_refs_use_name ON code_refs_use(name COLLATE NOCASE);
CREATE INDEX IF NOT EXISTS idx_refs_use_path ON code_refs_use(path);
CREATE VIRTUAL TABLE IF NOT EXISTS code_refs_fts USING fts5(
  name, context, content='code_refs_use', content_rowid='id');
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TRIGGER IF NOT EXISTS notes_ai AFTER INSERT ON notes BEGIN
  INSERT INTO notes_fts(rowid,content,tags) VALUES(new.id,new.content,new.tags);
END;
-- Notes are append-only in MVP: no delete/update triggers keep notes_fts in
-- sync with notes. Revisit if note pruning ever lands.
"#;

// Recall feedback loop (plan item 1): notes track how often they were recalled
// and when. Added as idempotent ALTERs rather than editing SCHEMA so existing
// databases migrate in place on open.
const MIGRATIONS: &[&str] = &[
    "ALTER TABLE notes ADD COLUMN recall_count INTEGER NOT NULL DEFAULT 0",
    "ALTER TABLE notes ADD COLUMN last_recalled INTEGER",
    // Tier lifecycle (plan item 3): every note starts as session-tier.
    "ALTER TABLE notes ADD COLUMN tier TEXT NOT NULL DEFAULT 'session'",
    // Reindex fidelity: the exact embedded string per symbol (NULL on
    // legacy DBs — reindex falls back to signature there).
    "ALTER TABLE symbols ADD COLUMN embed_text TEXT",
    "CREATE INDEX IF NOT EXISTS idx_refs_def_name_nocase ON code_refs_def(name COLLATE NOCASE)",
    "CREATE INDEX IF NOT EXISTS idx_refs_use_name_nocase ON code_refs_use(name COLLATE NOCASE)",
];

impl Store {
    pub fn open_in_memory() -> Result<Self> { Self::open(Connection::open_in_memory()?) }
    pub fn open_at(path: &str) -> Result<Self> {
        if let Some(parent) = std::path::Path::new(path).parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        Self::open(Connection::open(path)?)
    }
    fn open(conn: Connection) -> Result<Self> {
        conn.busy_timeout(std::time::Duration::from_millis(5000))?;
        conn.execute_batch(SCHEMA)?;
        for m in MIGRATIONS {
            // "duplicate column name" just means the migration already ran.
            let _ = conn.execute_batch(m);
        }
        Ok(Self { conn })
    }
    pub fn now() -> i64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0)
    }

    pub fn add_note(&self, sid: &str, content: &str, kind: &str, tags: &[&str]) -> Result<i64> {
        Self::note_insert(&self.conn, sid, content, kind, tags)
    }
    /// Connection-taking variant so callers running inside a transaction (see
    /// Session::write) reuse the exact same SQL instead of duplicating it.
    pub(crate) fn note_insert(conn: &Connection, sid: &str, content: &str,
                              kind: &str, tags: &[&str]) -> Result<i64> {
        conn.execute(
            "INSERT INTO notes(session_id,content,kind,tags,created_at) VALUES(?1,?2,?3,?4,?5)",
            params![sid, content, kind, serde_json::to_string(tags)?, Self::now()])?;
        Ok(conn.last_insert_rowid())
    }
    pub fn search_notes(&self, q: &str, n: usize) -> Result<Vec<(i64, String)>> {
        if q.trim().is_empty() {
            return Ok(vec![]);
        }
        let quoted = q
            .split_whitespace()
            .map(|t| format!("\"{}\"", t.replace('"', "\"\"")))
            .collect::<Vec<_>>()
            .join(" ");
        let mut st = self.conn.prepare(
            "SELECT n.id, snippet(notes_fts,0,'','','…',12)
             FROM notes_fts f JOIN notes n ON n.id=f.rowid
             WHERE notes_fts MATCH ?1 ORDER BY rank LIMIT ?2")?;
        let rows = st.query_map(params![quoted, n as i64], |r| Ok((r.get(0)?, r.get(1)?)))?;
        Ok(rows.collect::<std::result::Result<_, _>>()?)
    }
    pub fn upsert_entity(&self, name: &str, kind: &str) -> Result<i64> {
        Self::entity_upsert(&self.conn, name, kind)
    }
    pub(crate) fn entity_upsert(conn: &Connection, name: &str, kind: &str) -> Result<i64> {
        conn.execute(
            "INSERT INTO entities(name,kind,weight,last_seen) VALUES(?1,?2,1.0,?3)
             ON CONFLICT(name) DO UPDATE SET weight=weight+1.0, last_seen=excluded.last_seen",
            params![name, kind, Self::now()])?;
        Ok(conn.query_row("SELECT id FROM entities WHERE name=?1", [name], |r| r.get(0))?)
    }
    pub fn link(&self, src: i64, dst: i64, kind: &str) -> Result<()> {
        Self::edge_link(&self.conn, src, dst, kind)
    }
    pub fn get_meta(&self, key: &str) -> Result<Option<String>> {
        let mut st = self.conn.prepare("SELECT value FROM meta WHERE key=?1")?;
        let mut rows = st.query([key])?;
        if let Some(r) = rows.next()? { Ok(Some(r.get(0)?)) } else { Ok(None) }
    }
    pub fn set_meta(&self, key: &str, value: &str) -> Result<()> {
        self.conn.execute("INSERT OR REPLACE INTO meta(key,value) VALUES(?1,?2)", rusqlite::params![key, value])?;
        Ok(())
    }

    /// Code reference index (Phase 2): per-file rebuild — delete the file's
    /// rows from def/use (+ FTS rowid cleanup) then insert fresh. Idempotent
    /// by construction; safe to call on every index pass.
    pub fn index_code_refs(
        &self,
        path: &str,
        defs: &[(String, String, i64, i64, String)],
        uses: &[(String, String, i64, String, Option<String>)],
    ) -> Result<(usize, usize)> {
        self.conn.execute("DELETE FROM code_refs_def WHERE path=?1", [path])?;
        let use_ids: Vec<i64> = self
            .conn
            .prepare("SELECT id FROM code_refs_use WHERE path=?1")?
            .query_map([path], |r| r.get(0))?
            .flatten()
            .collect();
        for id in &use_ids {
            // FTS5 external-content table: delete-by-rowid keeps the index in
            // sync even though no triggers are registered for code_refs_use.
            let _ = self.conn.execute(
                "INSERT INTO code_refs_fts(code_refs_fts, rowid) VALUES('delete', ?1)",
                [id],
            );
        }
        self.conn.execute("DELETE FROM code_refs_use WHERE path=?1", [path])?;
        let mut n_def = 0usize;
        for (name, kind, ls, le, sig) in defs {
            self.conn.execute(
                "INSERT OR IGNORE INTO code_refs_def(path,name,kind,line_start,line_end,signature)
                 VALUES(?1,?2,?3,?4,?5,?6)",
                rusqlite::params![path, name, kind, ls, le, sig],
            )?;
            n_def += 1;
        }
        let mut n_use = 0usize;
        for (name, kind, line, ctx, target) in uses {
            self.conn.execute(
                "INSERT INTO code_refs_use(path,name,kind,line,context,target)
                 VALUES(?1,?2,?3,?4,?5,?6)",
                rusqlite::params![path, name, kind, line, ctx, target],
            )?;
            let rowid = self.conn.last_insert_rowid();
            self.conn.execute(
                "INSERT INTO code_refs_fts(rowid,name,context) VALUES(?1,?2,?3)",
                rusqlite::params![rowid, name, ctx],
            )?;
            n_use += 1;
        }
        Ok((n_def, n_use))
    }

    /// Reference lookup: exact identifier hit → (path, line, kind, context),
    /// defs first then uses, each capped. Plus FTS5 fuzzy fallback over
    /// name+context for prose queries. Returns (defs, uses, fts_rows).
    pub fn find_refs(
        &self,
        name: &str,
        limit: usize,
    ) -> Result<(RefDefs, RefUses, RefFts)> {
        let lim = limit.max(1) as i64;
        let defs: RefDefs = self
            .conn
            .prepare(
                "SELECT path, line_start, kind FROM code_refs_def
                 WHERE name = ?1 COLLATE NOCASE ORDER BY path, line_start LIMIT ?2",
            )?
            .query_map(rusqlite::params![name, lim], |r| {
                Ok((r.get(0)?, r.get(1)?, r.get(2)?))
            })?
            .flatten()
            .collect();
        let uses: RefUses = self
            .conn
            .prepare(
                "SELECT path, line, kind, context FROM code_refs_use
                 WHERE name = ?1 COLLATE NOCASE ORDER BY path, line LIMIT ?2",
            )?
            .query_map(rusqlite::params![name, lim], |r| {
                Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?))
            })?
            .flatten()
            .collect();
        // FTS5 fuzzy: quote each token like search_notes; hostile queries
        // yield zero rows instead of an error.
        let quoted = name
            .split_whitespace()
            .map(|t| format!("\"{}\"", t.replace('"', "\"\"")))
            .collect::<Vec<_>>()
            .join(" ");
        let fts: RefFts = if quoted.trim().is_empty() {
            Vec::new()
        } else {
            match self.conn.prepare(
                "SELECT u.path, u.line, u.context FROM code_refs_fts f
                 JOIN code_refs_use u ON u.id=f.rowid
                 WHERE code_refs_fts MATCH ?1 ORDER BY rank LIMIT ?2",
            ) {
                Ok(mut st) => st
                    .query_map(rusqlite::params![quoted, lim], |r| {
                        Ok((r.get(0)?, r.get(1)?, r.get(2)?))
                    })
                    .map(|m| m.flatten().collect())
                    .unwrap_or_default(),
                Err(_) => Vec::new(),
            }
        };
        Ok((defs, uses, fts))
    }

    /// IDF weight for a refs-leg token: `ln(1 + total_files / df_files)`.
    /// `df_files` counts distinct files matching the token by exact name
    /// OR FTS (name+context) — i.e. the same retrieval universe the refs
    /// leg itself searches, so dotted-only tokens (`auth.zeta`, present
    /// only inside FTS rows) still get a positive weight instead of df=0.
    /// Rare identifiers (the navigation signal) weigh far more than
    /// boilerplate (`description`, `check`) spread across many files. The
    /// +1 smoothing keeps the weight positive on tiny indexes (where
    /// total == df would otherwise silence the whole leg) while preserving
    /// the rare-over-common order at any scale. Returns 0.0 for names with
    /// no match anywhere.
    pub fn total_indexed_files(&self) -> Result<i64> {
        let n: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM indexed_files",
            [],
            |r| r.get(0),
        ).unwrap_or(0);
        if n > 0 {
            return Ok(n);
        }
        self.conn.query_row(
            "SELECT COUNT(*) FROM (SELECT DISTINCT path FROM code_refs_def
              UNION SELECT DISTINCT path FROM code_refs_use)",
            [],
            |r| r.get(0),
        ).map_err(Into::into)
    }

    pub fn refs_idf(&self, name: &str) -> Result<f64> {
        let total = self.total_indexed_files()?;
        self.refs_idf_with_total(name, total)
    }

    pub fn refs_idf_with_total(&self, name: &str, total: i64) -> Result<f64> {
        if total <= 0 {
            return Ok(0.0);
        }
        let quoted = name
            .split_whitespace()
            .map(|t| format!("\"{}\"", t.replace('"', "\"\"")))
            .collect::<Vec<_>>()
            .join(" ");
        // Exact-name files first (never fails); FTS files second with
        // independent fallback — a MATCH error must not zero exact hits.
        let mut files: std::collections::HashSet<String> = self
            .conn
            .prepare(
                "SELECT DISTINCT path FROM code_refs_def WHERE name = ?1 COLLATE NOCASE
                 UNION
                 SELECT DISTINCT path FROM code_refs_use WHERE name = ?1 COLLATE NOCASE",
            )?
            .query_map(rusqlite::params![name], |r| r.get(0))?
            .flatten()
            .collect();
        if !quoted.trim().is_empty() {
            if let Ok(mut st) = self.conn.prepare(
                "SELECT DISTINCT u.path FROM code_refs_fts f
                 JOIN code_refs_use u ON u.id = f.rowid
                 WHERE code_refs_fts MATCH ?1",
            ) {
                if let Ok(rows) = st.query_map([quoted.as_str()], |r| r.get::<_, String>(0)) {
                    files.extend(rows.flatten());
                }
            }
        }
        let n = files.len() as f64;
        if n <= 0.0 {
            return Ok(0.0);
        }
        Ok((1.0 + total as f64 / n).ln())
    }

    /// Repo-agnostic version of the generic gate: true when the fraction of
    /// indexed files defining/using `name` meets `ratio` (e.g. 0.02 = 2%).
    /// An absolute file count (the old `max_files`) is strict on huge repos
    /// and useless on small ones; a ratio scales with the DB itself, so the
    /// same code works on every repo with no per-repo tuning.
    ///
    /// Degeneracy floor: on a tiny index (total < ~50 files) any ratio
    /// under 1/total flags *every* name generic and silences the refs leg
    /// entirely. A name in 1–2 files is a definition (+1 use site), never
    /// shared vocabulary, so require n >= 3 regardless of ratio.
    pub fn refs_name_is_generic_ratio(&self, name: &str, ratio: f64) -> Result<bool> {
        let total = self.total_indexed_files()?;
        self.refs_name_is_generic_ratio_with_total(name, ratio, total)
    }

    pub fn refs_name_is_generic_ratio_with_total(&self, name: &str, ratio: f64, total: i64) -> Result<bool> {
        if total <= 0 {
            return Ok(false);
        }
        let n: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM (SELECT DISTINCT path FROM code_refs_def WHERE name = ?1 COLLATE NOCASE
              UNION SELECT DISTINCT path FROM code_refs_use WHERE name = ?1 COLLATE NOCASE)",
            rusqlite::params![name],
            |r| r.get(0),
        )?;
        Ok(n >= 3 && n as f64 / total as f64 >= ratio)
    }

    /// Out-of-file connections for a file: which other files it imports
    /// (via import/import_from refs) and which files reference its symbols
    /// by name. Both directions capped; powers the `map` out-links.
    pub fn file_connections(&self, path: &str, limit: usize) -> Result<(Vec<String>, Vec<String>)> {
        let lim = limit.max(1) as i64;
        // Outgoing: modules this file imports. Normalise dotted names to a
        // top-level segment for file matching (best-effort).
        let imported: Vec<String> = self
            .conn
            .prepare(
                "SELECT DISTINCT name FROM code_refs_use
                 WHERE path=?1 AND kind IN ('import','import_from')",
            )?
            .query_map([path], |r| r.get::<_, String>(0))?
            .flatten()
            .collect();
        // Incoming: files whose use-refs name a symbol defined in `path`.
        let mut incoming: Vec<String> = Vec::new();
        {
            let def_names: Vec<String> = self
                .conn
                .prepare("SELECT DISTINCT name FROM code_refs_def WHERE path=?1")?
                .query_map([path], |r| r.get::<_, String>(0))?
                .flatten()
                .collect();
            if !def_names.is_empty() {
                let placeholders = def_names.iter().map(|_| "?").collect::<Vec<_>>().join(",");
                let sql = format!(
                    "SELECT DISTINCT path FROM code_refs_use
                     WHERE path <> ? AND name COLLATE NOCASE IN ({placeholders}) LIMIT ?"
                );
                let mut params: Vec<String> = Vec::new();
                params.push(path.to_string());
                for n in &def_names {
                    params.push(n.clone());
                }
                params.push(lim.to_string());
                let mut st = self.conn.prepare(&sql)?;
                let rows = st.query_map(
                    rusqlite::params_from_iter(params.iter()),
                    |r| r.get::<_, String>(0),
                )?;
                incoming = rows.flatten().collect();
            }
        } // close incoming scope block
        let mut outgoing: Vec<String> = imported
            .into_iter()
            .map(|m| m.split('.').next().unwrap_or("").to_string())
            .filter(|s| !s.is_empty())
            .collect();
        outgoing.sort();
        outgoing.dedup();
        outgoing.truncate(limit);
        incoming.sort();
        incoming.dedup();
        Ok((outgoing, incoming))
    }

    pub(crate) fn edge_link(conn: &Connection, src: i64, dst: i64, kind: &str) -> Result<()> {
        conn.execute(
            "INSERT INTO edges(src,dst,kind,weight,co_mentions,last_seen) VALUES(?1,?2,?3,1.0,1,?4)
              ON CONFLICT(src,dst,kind) DO UPDATE SET weight=weight+1.0,
                co_mentions=co_mentions+1, last_seen=excluded.last_seen",
            params![src, dst, kind, Self::now()])?;
        Ok(())
    }

    /// Recall feedback (plan item 1): bump recall_count/last_recalled for the
    /// given note ids. Hebbian: entities mentioned in recalled notes and the
    /// edges among them get +1.0 weight. Best-effort per note; failures are
    /// logged, not propagated (recall must never fail because bookkeeping did).
    pub fn reinforce_recalled(&self, note_ids: &[i64]) -> Result<()> {
        let now = Self::now();
        for id in note_ids {
            self.conn.execute(
                "UPDATE notes SET recall_count=recall_count+1, last_recalled=?2 WHERE id=?1",
                params![id, now])?;
        }
        // Entity mentions: notes don't store their entity links directly —
        // resolve via name overlap is expensive; instead use note->entity
        // co-mention graph built at remember-time: entities whose name tokens
        // appear in the recalled note's content. Simple, language-independent.
        for &id in note_ids {
            let content: Option<String> = self
                .conn
                .query_row("SELECT content FROM notes WHERE id=?1", [id], |r| r.get(0))
                .ok();
            let content = match content {
                Some(c) => c.to_lowercase(),
                None => continue,
            };
            let ents: Vec<(i64, String)> = {
                let mut st = match self.conn.prepare("SELECT id, name FROM entities") {
                    Ok(s) => s,
                    Err(_) => continue,
                };
                let rows: Vec<(i64, String)> = match st
                    .query_map([], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?)))
                {
                    Ok(m) => m.flatten().collect(),
                    Err(_) => Vec::new(),
                };
                rows
            };
            let mut hit_ids: Vec<i64> = Vec::new();
            for (eid, name) in ents {
                if content.contains(&name.to_lowercase()) {
                    self.conn.execute(
                        "UPDATE entities SET weight=weight+1.0, last_seen=?2 WHERE id=?1",
                        params![eid, now]).ok();
                    hit_ids.push(eid);
                }
            }
            // Hebbian edge reinforcement: pairs of co-mentioned entities.
            for i in 0..hit_ids.len() {
                for j in (i + 1)..hit_ids.len() {
                    let (a, b) = (hit_ids[i], hit_ids[j]);
                    self.conn.execute(
                        "UPDATE edges SET weight=weight+1.0, co_mentions=co_mentions+1, last_seen=?3
                         WHERE (src=?1 AND dst=?2) OR (src=?2 AND dst=?1)",
                        params![a, b, now]).ok();
                }
            }
        }
        Ok(())
    }

    /// Explicit agent feedback (usefulness signal): a `useless` verdict
    /// decrements recall_count (floor 0), reversibly distrusting a note
    /// that misled — no stickiness, future recalls re-promote. `useful`
    /// reuses `reinforce_recalled`. Returns affected rows (0 = unknown id).
    pub fn demote_feedback(&self, note_ids: &[i64]) -> Result<usize> {
        let mut n = 0usize;
        for id in note_ids {
            n += self.conn.execute(
                "UPDATE notes SET recall_count = CASE WHEN recall_count > 0 THEN recall_count - 1 ELSE 0 END WHERE id=?1",
                params![id],
            )?;
        }
        Ok(n)
    }

    /// Query-time reinforcement factor for a note (plan item 2):
    /// `f(recall_count) = 1 + ln(1 + recall_count)`. Logarithmic so repeated
    /// recall has diminishing returns (a note recalled 10× isn't 10× better).
    pub fn reinforcement(recall_count: i64) -> f64 {
        1.0 + (1.0 + recall_count.max(0) as f64).ln()
    }

    /// Temporal decay factor (plan item 2): `e^(−Δt/τ)` with τ = 30 days in
    /// seconds. Notes never recalled use created_at. Range (0, 1].
    pub fn decay(last_or_created: i64, now: i64) -> f64 {
        const TAU: f64 = 30.0 * 24.0 * 3600.0;
        let dt = (now - last_or_created).max(0) as f64;
        (-(dt / TAU)).exp()
    }

    /// Tier lifecycle (plan item 3): promote session notes that proved useful
    /// (recall_count >= threshold — the cross-session proxy until per-session
    /// recall tracking lands) to project tier. Returns (id, from, to) rows.
    pub fn promote_tiers(&self, min_recalls: i64) -> Result<Vec<(i64, String, String)>> {
        let ids: Vec<i64> = {
            let mut st = self.conn.prepare(
                "SELECT id FROM notes WHERE tier='session' AND recall_count>=?1")?;
            let rows = st.query_map([min_recalls], |r| r.get::<_, i64>(0))?;
            rows.flatten().collect()
        };
        let mut out = Vec::new();
        for id in ids {
            self.conn.execute(
                "UPDATE notes SET tier='project' WHERE id=?1", [id])?;
            out.push((id, "session".into(), "project".into()));
        }
        Ok(out)
    }

    /// Tier lifecycle (plan item 3): soft-archive untouched notes. Only
    /// session/project notes with zero recalls older than `max_age_days` are
    /// archived — recall data always wins over age. Returns archived ids.
    pub fn archive_stale(&self, max_age_days: i64) -> Result<Vec<i64>> {
        let cutoff = Self::now() - max_age_days * 24 * 3600;
        let mut st = self.conn.prepare(
            "SELECT id FROM notes WHERE tier IN ('session','project')
             AND recall_count=0 AND created_at < ?1")?;
        let ids: Vec<i64> = st
            .query_map([cutoff], |r| r.get::<_, i64>(0))?
            .flatten()
            .collect();
        drop(st);
        for id in &ids {
            self.conn.execute(
                "UPDATE notes SET tier='archived' WHERE id=?1", [id])?;
        }
        Ok(ids)
    }

    /// Query-time tier prior: proven project knowledge outranks unproven
    /// session notes (session text is LLM-written and hallucination-prone;
    /// project tier means recalled 3+ times). Archived stays heavily demoted
    /// but findable on strong lexical/semantic matches.
    pub fn tier_prior(tier: &str) -> f64 {
        match tier {
            "project" => 1.25,
            "session" => 1.0,
            "archived" => 0.25,
            _ => 1.0,
        }
    }
}

/// Checkpoint WAL back into the main DB file so `.cortex/memory.db` stays
/// git-shareable with no `-wal`/`-shm` sidecars left behind.
///
/// Deliberately `PRAGMA wal_checkpoint(TRUNCATE)` only — NOT a full `VACUUM`.
/// VACUUM rewrites the whole file including 150MB+ embedding BLOBs, which is
/// far too expensive to pay on every digest/index finish. TRUNCATE checkpoints
/// pending WAL frames and truncates the WAL file to zero bytes, which is all
/// hygiene needs; space reclaim inside the main file (if ever needed) is left
/// to an explicit, opt-in operation.
pub fn checkpoint(s: &Store) -> Result<()> {
    s.conn.execute_batch("PRAGMA wal_checkpoint(TRUNCATE);")?;
    Ok(())
}

pub fn serialize_f32s(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
pub fn deserialize_f32s(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}
pub(crate) fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    if is_zero_vec(a) || is_zero_vec(b) {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a < 1e-9 || norm_b < 1e-9 {
        return 0.0;
    }
    (dot / (norm_a * norm_b)).clamp(-1.0, 1.0)
}
pub(crate) fn is_zero_vec(v: &[f32]) -> bool {
    v.iter().all(|x| x.abs() < 1e-9)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mem() -> Store { Store::open_in_memory().unwrap() }

    #[test]
    fn schema_creates_all_tables() {
        let s = mem();
        for t in ["notes", "entities", "edges", "sessions", "symbols"] {
            let n: i64 = s.conn.query_row(
                "SELECT count(*) FROM sqlite_master WHERE type='table' AND name=?1",
                [t], |r| r.get(0)).unwrap();
            assert_eq!(n, 1, "missing table {t}");
        }
    }
    #[test]
    fn note_roundtrip_and_fts() {
        let s = mem();
        let id = s.add_note("ses1", "jwt expiry lives in validate_token", "decision", &["auth"]).unwrap();
        let hits = s.search_notes("jwt expiry", 5).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].0, id);
        assert!(hits[0].1.contains("validate_token"));
    }
    #[test]
    fn fts_survives_hostile_queries() {
        let s = mem();
        let hits = s.search_notes("foo-bar junk\"quote", 5).unwrap();
        assert!(hits.is_empty());
        assert!(s.search_notes("", 5).unwrap().is_empty());
    }
    #[test]
    fn find_refs_is_case_insensitive() {
        let s = mem();
        s.index_code_refs(
            "a.py",
            &[("FITSDiff".to_string(), "class".to_string(), 1, 10, "class FITSDiff:".to_string())],
            &[],
        ).unwrap();
        // Query tokens are lowercased at tokenize time; the index stores
        // raw source case. Exact arm must still hit.
        let (d, _, _) = s.find_refs("fitsdiff", 5).unwrap();
        assert_eq!(d.len(), 1, "lowercase query must hit CamelCase def");
    }
    #[test]
    fn generic_ratio_scales_with_repo_size() {
        let s = mem();
        for i in 0..50 {
            let p = format!("f{i}.py");
            s.conn
                .execute(
                    "INSERT INTO code_refs_use(path,name,kind,line,context,target)
                     VALUES(?1,'data','call',1,'ctx',NULL)",
                    rusqlite::params![p],
                )
                .unwrap();
        }
        s.conn
            .execute(
                "INSERT INTO code_refs_use(path,name,kind,line,context,target)
                 VALUES('g.py','rare_name','call',1,'ctx',NULL)",
                [],
            )
            .unwrap();
        // 50/51 files >> 2% -> generic under any repo size.
        assert!(s.refs_name_is_generic_ratio("data", 0.02).unwrap());
        // 1/51 files < 2% -> specific.
        assert!(!s.refs_name_is_generic_ratio("rare_name", 0.02).unwrap());
        // Empty index never generic.
        let e = mem();
        assert!(!e.refs_name_is_generic_ratio("anything", 0.02).unwrap());
    }
    #[test]
    fn generic_ratio_floor_protects_small_repos() {
        let s = mem();
        // 5-file repo: a name in 2 files is def+use, never generic even
        // though 2/5 = 40% >> 2%.
        for i in 0..2 {
            s.conn.execute(
                "INSERT INTO code_refs_use(path,name,kind,line,context,target)
                 VALUES(?1,'helper','call',1,'ctx',NULL)",
                rusqlite::params![format!("f{i}.py")],
            ).unwrap();
        }
        assert!(!s.refs_name_is_generic_ratio("helper", 0.02).unwrap());
        // Same repo, name in 3 files -> generic (floor met, ratio met).
        for i in 0..3 {
            s.conn.execute(
                "INSERT INTO code_refs_use(path,name,kind,line,context,target)
                 VALUES(?1,'data','call',1,'ctx',NULL)",
                rusqlite::params![format!("g{i}.py")],
            ).unwrap();
        }
        assert!(s.refs_name_is_generic_ratio("data", 0.02).unwrap());
    }
    #[test]
    fn refs_idf_orders_rare_above_common() {
        let s = mem();
        for i in 0..10 {
            let p = format!("f{i}.py");
            s.conn.execute(
                "INSERT INTO code_refs_use(path,name,kind,line,context,target)
                 VALUES(?1,'common','call',1,'ctx',NULL)",
                rusqlite::params![p],
            ).unwrap();
        }
        s.conn.execute(
            "INSERT INTO code_refs_use(path,name,kind,line,context,target)
             VALUES('g.py','rare','call',1,'ctx',NULL)",
            [],
        ).unwrap();
        let rare = s.refs_idf("rare").unwrap();
        let common = s.refs_idf("common").unwrap();
        assert!(rare > common, "rare {rare} must outweigh common {common}");
        assert_eq!(s.refs_idf("absent").unwrap(), 0.0);
        assert_eq!(mem().refs_idf("anything").unwrap(), 0.0);
    }
    #[test]
    fn entity_upsert_reinforces_weight() {
        let s = mem();
        let e1 = s.upsert_entity("validate_jwt", "function").unwrap();
        s.upsert_entity("validate_jwt", "function").unwrap();
        let w: f64 = s.conn.query_row("SELECT weight FROM entities WHERE id=?1",
            [e1], |r| r.get(0)).unwrap();
        assert!((w - 2.0).abs() < 1e-9);
    }
    #[test]
    fn checkpoint_truncates_wal_sidecar() {
        let dir = std::env::temp_dir().join(format!(
            "cx_checkpoint_{}_{}",
            std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let db_path = dir.join("memory.db");
        let db_str = db_path.to_string_lossy().to_string();
        let wal_path = dir.join("memory.db-wal");
        {
            let s = Store::open_at(&db_str).unwrap();
            s.add_note("ses-ckpt", "checkpoint hygiene probe", "note", &["ckpt"]).unwrap();
            super::checkpoint(&s).unwrap();
            let (busy, _log, _ckpted): (i32, i32, i32) = s.conn
                .query_row("PRAGMA wal_checkpoint(TRUNCATE);", [], |r| {
                    Ok((r.get(0)?, r.get(1)?, r.get(2)?))
                })
                .unwrap();
            assert_eq!(busy, 0, "checkpoint busy must be 0");
        }
        let wal_gone_or_empty = !wal_path.exists()
            || wal_path.metadata().map(|m| m.len() == 0).unwrap_or(true);
        assert!(wal_gone_or_empty, "WAL sidecar must be truncated away: {}", wal_path.display());
        let _ = std::fs::remove_dir_all(&dir);
    }
    #[test]
    fn edge_upsert_increments_co_mentions() {
        let s = mem();
        let a = s.upsert_entity("a", "fn").unwrap();
        let b = s.upsert_entity("b", "fn").unwrap();
        s.link(a, b, "co").unwrap();
        s.link(a, b, "co").unwrap();
        let (w, c): (f64, i64) = s.conn.query_row(
            "SELECT weight, co_mentions FROM edges WHERE src=?1 AND dst=?2",
            [a, b], |r| Ok((r.get(0)?, r.get(1)?))).unwrap();
        assert_eq!(c, 2);
        assert!((w - 2.0).abs() < 1e-9);
    }

    #[test]
    fn reinforce_bumps_recall_count_and_hebbian_weights() {
        let s = mem();
        let id = s.add_note("ses1", "the validate_jwt token check", "note", &[]).unwrap();
        let e1 = s.upsert_entity("validate_jwt", "function").unwrap();
        s.link(e1, e1, "self").ok(); // ensure edges table non-trivial
        s.reinforce_recalled(&[id]).unwrap();
        let (rc, lr): (i64, i64) = s.conn.query_row(
            "SELECT recall_count, last_recalled FROM notes WHERE id=?1",
            [id], |r| Ok((r.get(0)?, r.get(1).unwrap_or(0)))).unwrap();
        assert_eq!(rc, 1);
        assert!(lr > 0, "last_recalled should be set");
        let w: f64 = s.conn.query_row("SELECT weight FROM entities WHERE id=?1",
            [e1], |r| r.get(0)).unwrap();
        assert!((w - 2.0).abs() < 1e-9, "entity weight bumped 1→2, got {w}");
        // Second recall compounds.
        s.reinforce_recalled(&[id]).unwrap();
        let rc2: i64 = s.conn.query_row(
            "SELECT recall_count FROM notes WHERE id=?1", [id], |r| r.get(0)).unwrap();
        assert_eq!(rc2, 2);
    }

    #[test]
    fn feedback_demote_floors_and_counts() {
        let s = mem();
        let id = s.add_note("ses1", "flaky claim here", "note", &[]).unwrap();
        assert_eq!(s.demote_feedback(&[id, 999999]).unwrap(), 1, "only real rows count");
        assert_eq!(s.demote_feedback(&[id]).unwrap(), 1, "floor 0 still matches row");
        let rc: i64 = s.conn.query_row(
            "SELECT recall_count FROM notes WHERE id=?1", [id], |r| r.get(0)).unwrap();
        assert_eq!(rc, 0, "never below zero");
        s.reinforce_recalled(&[id]).unwrap();
        s.reinforce_recalled(&[id]).unwrap();
        s.demote_feedback(&[id]).unwrap();
        let rc2: i64 = s.conn.query_row(
            "SELECT recall_count FROM notes WHERE id=?1", [id], |r| r.get(0)).unwrap();
        assert_eq!(rc2, 1, "2 up, 1 down");
    }

    #[test]
    fn decay_and_reinforcement_shapes() {
        let now = Store::now();
        // Fresh note: decay ≈ 1. 30 days old: e^-1 ≈ 0.368.
        assert!(Store::decay(now, now) > 0.999);
        let d30 = Store::decay(now - 30 * 24 * 3600, now);
        assert!((d30 - (-1.0f64).exp()).abs() < 1e-6);
        assert!(Store::decay(now - 10_000 * 24 * 3600, now) > 0.0);
        // Never recalled: factor 1. Recalled once: 1+ln2 ≈ 1.693.
        assert!((Store::reinforcement(0) - 1.0).abs() < 1e-12);
        assert!((Store::reinforcement(1) - (1.0 + 2.0f64.ln())).abs() < 1e-12);
        // Negative counts clamped, diminishing returns.
        assert_eq!(Store::reinforcement(-5), Store::reinforcement(0));
        assert!(Store::reinforcement(100) < Store::reinforcement(10) * 2.0);
    }

    #[test]
    fn migration_adds_recall_columns_to_legacy_db() {
        // Simulate a pre-migration DB: create notes without the new columns,
        // then open via Store::open_at and confirm columns were added.
        let dir = std::env::temp_dir().join(format!(
            "cx_mig_{}_{}",
            std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
        ));
        let _ = std::fs::create_dir_all(&dir);
        let db = dir.join("legacy.db");
        let conn = rusqlite::Connection::open(&db).unwrap();
        conn.execute_batch(
            "CREATE TABLE notes(id INTEGER PRIMARY KEY, session_id TEXT NOT NULL,
             content TEXT NOT NULL, kind TEXT NOT NULL, tags TEXT NOT NULL DEFAULT '[]',
             created_at INTEGER NOT NULL, embedding BLOB);",
        )
        .unwrap();
        conn.execute(
            "INSERT INTO notes(session_id,content,kind,created_at) VALUES('s','legacy note','note',1)",
            [],
        )
        .unwrap();
        drop(conn);
        let s = Store::open_at(&db.to_string_lossy()).unwrap();
        let (rc, lr): (i64, Option<i64>) = s.conn.query_row(
            "SELECT recall_count, last_recalled FROM notes LIMIT 1",
            [], |r| Ok((r.get(0)?, r.get(1)?))).unwrap();
        assert_eq!(rc, 0);
        assert_eq!(lr, None);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn migration_adds_embed_text_to_legacy_symbols() {
        // Pre-migration DB: symbols without embed_text; open must add the
        // column (NULL = legacy) without touching existing rows.
        let dir = std::env::temp_dir().join(format!(
            "cx_mig_sym_{}_{}",
            std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
        ));
        let _ = std::fs::create_dir_all(&dir);
        let db = dir.join("legacy.db");
        let conn = rusqlite::Connection::open(&db).unwrap();
        conn.execute_batch(
            "CREATE TABLE symbols(id INTEGER PRIMARY KEY, path TEXT NOT NULL,
             name TEXT NOT NULL, kind TEXT NOT NULL, line_start INTEGER NOT NULL,
             line_end INTEGER NOT NULL, signature TEXT, embedding BLOB);",
        )
        .unwrap();
        conn.execute(
            "INSERT INTO symbols(path,name,kind,line_start,line_end,signature)
             VALUES('a.py','f','function',1,5,'def f():')",
            [],
        )
        .unwrap();
        drop(conn);
        let s = Store::open_at(&db.to_string_lossy()).unwrap();
        let t: Option<String> = s.conn.query_row(
            "SELECT embed_text FROM symbols LIMIT 1",
            [], |r| r.get(0)).unwrap();
        assert_eq!(t, None, "legacy rows stay NULL (signature fallback)");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn tier_lifecycle_promote_and_archive() {
        let s = mem();
        let proven = s.add_note("s1", "proven note about caching", "note", &[]).unwrap();
        let stale = s.add_note("s1", "stale never-used note", "note", &[]).unwrap();
        let fresh = s.add_note("s1", "fresh never-used note", "note", &[]).unwrap();
        // proven: 3 recalls; stale: created 60 days ago, 0 recalls; fresh: new.
        let old = Store::now() - 60 * 24 * 3600;
        s.conn.execute("UPDATE notes SET recall_count=3 WHERE id=?1", [proven]).unwrap();
        s.conn.execute("UPDATE notes SET created_at=?1 WHERE id=?2", [old, stale]).unwrap();
        let promoted = s.promote_tiers(3).unwrap();
        assert_eq!(promoted, vec![(proven, "session".to_string(), "project".to_string())]);
        let archived = s.archive_stale(30).unwrap();
        assert_eq!(archived, vec![stale]);
        let tiers: Vec<(i64, String)> = s.conn.prepare(
            "SELECT id, tier FROM notes ORDER BY id")
            .unwrap().query_map([], |r| Ok((r.get(0)?, r.get(1)?)))
            .unwrap().flatten().collect();
        let get = |id: i64| tiers.iter().find(|(i, _)| *i == id).map(|(_, t)| t.clone()).unwrap();
        assert_eq!(get(proven), "project");
        assert_eq!(get(stale), "archived");
        assert_eq!(get(fresh), "session");
    }

    #[test]
    fn tier_prior_ordering() {
        assert!(Store::tier_prior("project") > Store::tier_prior("session"));
        assert!(Store::tier_prior("session") > Store::tier_prior("archived"));
        assert_eq!(Store::tier_prior("unknown"), 1.0);
    }

    #[test]
    fn code_refs_schema_present() {
        let s = mem();
        for t in ["code_refs_def", "code_refs_use", "code_refs_fts"] {
            let n: i64 = s.conn.query_row(
                "SELECT count(*) FROM sqlite_master WHERE name=?1",
                [t], |r| r.get(0)).unwrap();
            assert_eq!(n, 1, "missing {t}");
        }
    }

    #[test]
    fn code_refs_index_find_roundtrip() {
        let s = mem();
        let defs = vec![("validate_token".to_string(), "function".to_string(), 10, 40,
            "def validate_token(x):".to_string())];
        let uses = vec![
            ("validate_token".to_string(), "call".to_string(), 55,
                "y = validate_token(x)".to_string(), None),
            ("os".to_string(), "import".to_string(), 1,
                "import os".to_string(), None),
        ];
        let (nd, nu) = s.index_code_refs("pkg/a.py", &defs, &uses).unwrap();
        assert_eq!((nd, nu), (1, 2));
        let (defs, uses, _) = s.find_refs("validate_token", 10).unwrap();
        assert_eq!(defs.len(), 1);
        assert_eq!(defs[0], ("pkg/a.py".to_string(), 10, "function".to_string()));
        assert_eq!(uses.len(), 1);
        assert_eq!(uses[0].0, "pkg/a.py".to_string());
        assert_eq!(uses[0].1, 55);
        assert!(uses[0].3.contains("validate_token(x)"));
        // Per-file rebuild is idempotent: re-index same file, no dupes.
        s.index_code_refs("pkg/a.py", &defs_iter(), &uses_iter()).unwrap();
        let (defs2, uses2, _) = s.find_refs("validate_token", 10).unwrap();
        assert_eq!(defs2.len(), 1);
        assert_eq!(uses2.len(), 1);

        fn defs_iter() -> Vec<(String, String, i64, i64, String)> {
            vec![("validate_token".to_string(), "function".to_string(), 10, 40,
                "def validate_token(x):".to_string())]
        }
        fn uses_iter() -> Vec<(String, String, i64, String, Option<String>)> {
            vec![
                ("validate_token".to_string(), "call".to_string(), 55,
                    "y = validate_token(x)".to_string(), None),
                ("os".to_string(), "import".to_string(), 1,
                    "import os".to_string(), None),
            ]
        }
    }

    #[test]
    fn code_refs_file_connections() {
        let s = mem();
        s.index_code_refs("pkg/a.py",
            &[("serve".to_string(), "function".to_string(), 5, 20, "def serve():".to_string())],
            &[("os".to_string(), "import".to_string(), 1, "import os".to_string(), None)],
        ).unwrap();
        s.index_code_refs("pkg/b.py", &[],
            &[("serve".to_string(), "call".to_string(), 9, "serve()".to_string(), None)],
        ).unwrap();
        let (out, inc) = s.file_connections("pkg/a.py", 10).unwrap();
        assert!(out.contains(&"os".to_string()), "outgoing: {out:?}");
        assert!(inc.contains(&"pkg/b.py".to_string()), "incoming: {inc:?}");
        // Hostile/empty queries never error.
        let (d, u, f) = s.find_refs("", 5).unwrap();
        assert!(d.is_empty() && u.is_empty() && f.is_empty());
        let (d2, u2, _) = s.find_refs("foo-bar junk\"quote", 5).unwrap();
        assert!(d2.is_empty() && u2.is_empty());
    }

    #[test]
    fn indexed_files_table_present_on_legacy_open() {
        // Pre-ledger DB (notes only): open must create `indexed_files`
        // empty — same CREATE-IF-NOT-EXISTS precedent as code_refs_*.
        let dir = std::env::temp_dir().join(format!(
            "cx_idxfmig_{}_{}",
            std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
        ));
        let _ = std::fs::create_dir_all(&dir);
        let db = dir.join("legacy.db");
        let conn = rusqlite::Connection::open(&db).unwrap();
        conn.execute_batch(
            "CREATE TABLE notes(id INTEGER PRIMARY KEY, session_id TEXT NOT NULL,
             content TEXT NOT NULL, kind TEXT NOT NULL, tags TEXT NOT NULL DEFAULT '[]',
             created_at INTEGER NOT NULL, embedding BLOB);",
        ).unwrap();
        drop(conn);
        let s = Store::open_at(&db.to_string_lossy()).unwrap();
        let n: i64 = s.conn.query_row(
            "SELECT count(*) FROM indexed_files",
            [], |r| r.get(0)).unwrap();
        assert_eq!(n, 0);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn code_refs_migrate_legacy_db() {
        // Simulate a pre-Phase-2 DB (no code_refs_* tables): open via
        // Store::open_at must create them via SCHEMA on open.
        let dir = std::env::temp_dir().join(format!(
            "cx_refsmig_{}_{}",
            std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
        ));
        let _ = std::fs::create_dir_all(&dir);
        let db = dir.join("legacy.db");
        let conn = rusqlite::Connection::open(&db).unwrap();
        conn.execute_batch(
            "CREATE TABLE notes(id INTEGER PRIMARY KEY, session_id TEXT NOT NULL,
             content TEXT NOT NULL, kind TEXT NOT NULL, tags TEXT NOT NULL DEFAULT '[]',
             created_at INTEGER NOT NULL, embedding BLOB);",
        ).unwrap();
        drop(conn);
        let s = Store::open_at(&db.to_string_lossy()).unwrap();
        let n: i64 = s.conn.query_row(
            "SELECT count(*) FROM sqlite_master WHERE name='code_refs_use'",
            [], |r| r.get(0)).unwrap();
        assert_eq!(n, 1);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
