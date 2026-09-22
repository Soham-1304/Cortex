use crate::store::Store;
use crate::Result;
use std::collections::HashMap;

pub struct Session<'a> { db: &'a Store, pub id: String }

impl<'a> Session<'a> {
    pub fn start(db: &'a Store, id: &str) -> Result<Self> {
        db.conn.execute(
            "INSERT OR IGNORE INTO sessions(id,started_at) VALUES(?1,?2)",
            rusqlite::params![id, Store::now()])?;
        Ok(Self { db, id: id.to_string() })
    }

    /// Live merge: note row + entity upserts + edges, immediately visible to recalls.
    /// LLM supplies semantics (entities/links); store supplies deterministic weights.
    /// Each distinct name bumps weight once per write: link endpoints already
    /// present among `entities` reuse their id instead of re-upserting, and
    /// duplicate names collapse to one id before co_mention pairing (no self-edges,
    /// no double-counted pairs).
    ///
    /// Atomicity: the entire write runs in one transaction — a failure partway
    /// rolls back everything, leaving no note with a partial graph behind.
    ///
    /// Embeddings are best-effort: if `emb.embed` fails (e.g. Ollama down),
    /// the note + graph still persist with NULL embeddings; vectors are an
    /// enhancement, not a correctness gate. The embed call runs BEFORE the
    /// tx opens — providers may do blocking HTTP, which must never hold the
    /// SQLite write lock.
    pub fn write(&mut self, content: &str, kind: &str,
                 entities: &[String], links: &[(&str, &str)],
                 emb: &dyn crate::embed::EmbeddingProvider) -> Result<i64> {
        let mut distinct: Vec<String> = Vec::with_capacity(entities.len());
        for e in entities {
            if !distinct.iter().any(|d| d == e) {
                distinct.push(e.clone());
            }
        }
        let texts: Vec<String> = std::iter::once(content.to_string()).chain(distinct.iter().cloned()).collect();
        let n_texts = texts.len();
        // Strict length check: a short vector list must skip ALL updates, never
        // partially apply (a shifted zip would attach the wrong vector to a name).
        let vecs: Option<Vec<Vec<f32>>> = emb.embed(texts).ok().filter(|v| v.len() == n_texts);
        let tx = self.db.conn.unchecked_transaction()?;
        let tags: Vec<&str> = entities.iter().map(|s| s.as_str()).collect();
        let nid = Store::note_insert(&tx, &self.id, content, kind, &tags)?;
        let mut by_name: HashMap<String, i64> = HashMap::new();
        let mut ids: Vec<i64> = Vec::with_capacity(entities.len());
        for e in entities {
            let name = e.as_str();
            let id = match by_name.get(name) {
                Some(&id) => id,
                None => {
                    let id = Store::entity_upsert(&tx, name, kind)?;
                    by_name.insert(name.to_string(), id);
                    id
                }
            };
            ids.push(id);
        }
        let mut resolve = |name: &str| -> Result<i64> {
            match by_name.get(name) {
                Some(&id) => Ok(id),
                None => {
                    let id = Store::entity_upsert(&tx, name, "concept")?;
                    by_name.insert(name.to_string(), id);
                    Ok(id)
                }
            }
        };
        for (a, b) in links {
            let ia = resolve(a)?;
            let ib = resolve(b)?;
            Store::edge_link(&tx, ia, ib, "relates")?;
        }
        ids.sort_unstable();
        ids.dedup();
        for (i, a) in ids.iter().enumerate() {
            for b in ids.iter().skip(i + 1) {
                if a != b {
                    Store::edge_link(&tx, *a, *b, "co_mention")?;
                }
            }
        }
        if let Some(v) = vecs {
            if let Some((first, rest)) = v.split_first() {
                tx.execute("UPDATE notes SET embedding=?1 WHERE id=?2", rusqlite::params![crate::store::serialize_f32s(first), nid])?;
                for (name, e) in distinct.iter().zip(rest.iter()) {
                    tx.execute("UPDATE entities SET embedding=?1 WHERE name=?2", rusqlite::params![crate::store::serialize_f32s(e), name])?;
                }
            }
        }
        tx.commit()?;
        Ok(nid)
    }

    /// Merge into project tier: entity/edge rows are already global (weights live there),
    /// so merge = sweep orphans + close this session with summary.
    pub fn digest(&mut self, summary: &str, tokens_used: i64) -> Result<()> {
        self.sweep_orphans()?;
        self.db.conn.execute(
            "UPDATE sessions SET ended_at=?1, summary=?2, tokens_used=?3 WHERE id=?4",
            rusqlite::params![Store::now(), summary, tokens_used, self.id])?;
        Ok(())
    }

    /// Sweeps sessions left open by earlier processes. Assumption: a single writer
    /// process owns the DB at a time (one stdio MCP server or CLI instance). WAL
    /// allows concurrent multi-process *readers*, but any digest sweeps ALL open
    /// sessions DB-wide — two concurrent writers would close each other's live
    /// sessions. Future hardening: only sweep sessions older than a `started_at`
    /// staleness threshold instead of every open session.
    fn sweep_orphans(&self) -> Result<()> {
        self.db.conn.execute(
            "UPDATE sessions SET ended_at=?1, summary='[orphaned: no digest]'
             WHERE ended_at IS NULL AND id != ?2",
            rusqlite::params![Store::now(), self.id])?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::Store;

    #[test]
    fn write_fills_entity_embeddings() {
        let s = Store::open_in_memory().unwrap();
        let mut ses = Session::start(&s, "e1").unwrap();
        let emb = crate::embed::HashEmbedder::new(64);
        ses.write("sqlite beats postgres", "decision", &["sqlite".into()], &[], &emb).unwrap();
        let blob: Option<Vec<u8>> = s.conn.query_row("SELECT embedding FROM entities WHERE name='sqlite'", [], |r| r.get(0)).unwrap();
        assert!(blob.is_some(), "entity embedding must be filled");
        let nblob: Option<Vec<u8>> = s.conn.query_row("SELECT embedding FROM notes LIMIT 1", [], |r| r.get(0)).unwrap();
        assert!(nblob.is_some(), "note embedding must be filled");
    }

    struct FailEmbed;
    impl crate::embed::EmbeddingProvider for FailEmbed {
        fn name(&self) -> &str { "fail" }
        fn dim(&self) -> usize { 64 }
        fn is_local(&self) -> bool { true }
        fn embed(&self, _texts: Vec<String>) -> crate::Result<Vec<Vec<f32>>> {
            Err(crate::Error::Cli("boom".into()))
        }
        fn health_check(&self) -> crate::Result<()> { Ok(()) }
    }

    #[test]
    fn write_persists_despite_embed_failure() {
        let s = Store::open_in_memory().unwrap();
        let mut ses = Session::start(&s, "e-fail").unwrap();
        let nid = ses.write("content here", "note",
            &["a".to_string(), "b".to_string()], &[("a", "b")], &FailEmbed).unwrap();
        let n: i64 = s.conn.query_row("SELECT count(*) FROM notes WHERE id=?1", [nid], |r| r.get(0)).unwrap();
        assert_eq!(n, 1, "note must persist despite embed failure");
        let e: i64 = s.conn.query_row("SELECT count(*) FROM entities WHERE name IN ('a','b')", [], |r| r.get(0)).unwrap();
        assert_eq!(e, 2, "entities must persist despite embed failure");
        let edges: i64 = s.conn.query_row("SELECT count(*) FROM edges", [], |r| r.get(0)).unwrap();
        assert!(edges >= 1, "edges must persist despite embed failure");
        let nb: Option<Vec<u8>> = s.conn.query_row("SELECT embedding FROM notes WHERE id=?1", [nid], |r| r.get(0)).unwrap();
        assert!(nb.is_none(), "note embedding must stay NULL on embed failure");
        let nulls: i64 = s.conn.query_row("SELECT count(*) FROM entities WHERE name IN ('a','b') AND embedding IS NULL", [], |r| r.get(0)).unwrap();
        assert_eq!(nulls, 2, "entity embeddings must stay NULL on embed failure");
    }

    #[test]
    fn digest_merges_graph_into_project_and_closes_session() {
        let s = Store::open_in_memory().unwrap();
        let mut ses = Session::start(&s, "s1").unwrap();
        let emb = crate::embed::HashEmbedder::new(64);
        ses.write("decided: sqlite over postgres", "decision",
                  &["sqlite".to_string(), "postgres".to_string()],
                  &[("sqlite", "postgres")], &emb).unwrap();
        ses.write("second note about sqlite", "note",
                  &["sqlite".to_string()], &[], &emb).unwrap();
        ses.digest("used sqlite; chose it over postgres", 120).unwrap();

        let w_sqlite: f64 = s.conn.query_row(
            "SELECT weight FROM entities WHERE name='sqlite'", [], |r| r.get(0)).unwrap();
        assert!((w_sqlite - 2.0).abs() < 1e-9, "w_sqlite={w_sqlite}");
        let (ended, sum): (Option<i64>, String) = s.conn.query_row(
            "SELECT ended_at, summary FROM sessions WHERE id='s1'", [],
            |r| Ok((r.get(0)?, r.get(1)?))).unwrap();
        assert!(ended.is_some());
        assert!(sum.contains("sqlite"));
        let n: i64 = s.conn.query_row("SELECT count(*) FROM notes", [], |r| r.get(0)).unwrap();
        assert_eq!(n, 2);
    }

    #[test]
    fn duplicate_entities_no_self_edge_no_double_count() {
        let s = Store::open_in_memory().unwrap();
        let mut ses = Session::start(&s, "s-dup").unwrap();
        let emb = crate::embed::HashEmbedder::new(64);
        ses.write("dup mentioned twice alongside other", "note",
                  &["dup".to_string(), "dup".to_string(), "other".to_string()],
                  &[], &emb).unwrap();
        let self_edges: i64 = s.conn.query_row(
            "SELECT count(*) FROM edges WHERE src=dst", [], |r| r.get(0)).unwrap();
        assert_eq!(self_edges, 0, "self-edges must never be created");
        let (dup_id, other_id): (i64, i64) = s.conn.query_row(
            "SELECT (SELECT id FROM entities WHERE name='dup'),
                    (SELECT id FROM entities WHERE name='other')", [],
            |r| Ok((r.get(0)?, r.get(1)?))).unwrap();
        let co: i64 = s.conn.query_row(
            "SELECT co_mentions FROM edges
             WHERE kind='co_mention'
               AND ((src=?1 AND dst=?2) OR (src=?2 AND dst=?1))",
            [dup_id, other_id], |r| r.get(0)).unwrap();
        assert_eq!(co, 1, "pair must be counted exactly once despite duplicate name");
    }

    #[test]
    fn orphaned_session_swept_by_later_digest() {
        let s = Store::open_in_memory().unwrap();
        let emb = crate::embed::HashEmbedder::new(64);
        {
            let mut x = Session::start(&s, "ghost").unwrap();
            x.write("lost note", "note", &["g".to_string()], &[], &emb).unwrap();
        }
        let mut y = Session::start(&s, "real").unwrap();
        y.write("kept", "note", &["k".to_string()], &[], &emb).unwrap();
        y.digest("ok", 1).unwrap();
        let swept: i64 = s.conn.query_row(
            "SELECT count(*) FROM sessions WHERE id='ghost' AND ended_at IS NOT NULL",
            [], |r| r.get(0)).unwrap();
        assert_eq!(swept, 1);
    }
}
