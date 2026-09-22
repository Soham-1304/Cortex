//! Structured event log (plan item 4): `.cortex/events.jsonl`.
//!
//! One JSON object per line. Every recall/compile event records ts, kind,
//! query_hash, hits (id, tier, score), tokens, took_ms and per-stage timings.
//! This is the raw data needed to fit rho/ECE on memory scores without
//! rerunning benchmarks — the log IS the calibration dataset.
//!
//! Append-only, never fatal: logging must not break recall if the filesystem
//! is read-only or the dir vanished.

use serde_json::json;
use std::io::Write;
use std::path::Path;

/// A hit recorded in a recall event. `tier` is "note" or "symbol" for now;
/// session-first/project promotion (plan item 3) will extend this vocabulary.
#[derive(Debug, Clone, serde::Serialize)]
pub struct EventHit {
    pub id: String,
    pub tier: String,
    pub score: f64,
}

/// Write one event line to `<base>/.cortex/events.jsonl`. Never panics, never
/// aborts the caller: all errors swallowed (log is advisory).
pub fn log_event(base: &Path, kind: &str, query: &str, hits: &[EventHit],
                 tokens: i64, took_ms: u64, stages: &serde_json::Value) {
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    // query_hash: first 16 hex of a simple FNV-1a — stable across runs,
    // never stores raw query text (privacy: notes content is sensitive).
    let mut h: u64 = 0xcbf29ce484222325;
    for b in query.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    let rec = json!({
        "ts": ts,
        "kind": kind,
        "query_hash": format!("{h:016x}"),
        "hits": hits,
        "tokens": tokens,
        "took_ms": took_ms,
        "stages": stages,
    });
    let dir = base.join(".cortex");
    if dir.exists() {
        let path = dir.join("events.jsonl");
        if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&path) {
            let _ = writeln!(f, "{rec}");
        }
    }
}

/// Stage timings from compile() (plan item 5).
#[derive(Debug, Clone, Copy, Default)]
pub struct StageTimes {
    pub fts_ms: u64,
    pub vec_ms: u64,
    pub graph_ms: u64,
    pub pack_ms: u64,
}

impl StageTimes {
    pub fn to_json(&self) -> serde_json::Value {
        json!({
            "fts_ms": self.fts_ms,
            "vec_ms": self.vec_ms,
            "graph_ms": self.graph_ms,
            "pack_ms": self.pack_ms,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn event_log_roundtrip_and_fnv_stability() {
        let dir = std::env::temp_dir().join(format!(
            "cx_ev_{}_{}", std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
        ));
        std::fs::create_dir_all(dir.join(".cortex")).unwrap();
        let hits = vec![EventHit { id: "n1".into(), tier: "note".into(), score: 0.5 }];
        log_event(&dir, "recall", "jwt bug", &hits, 42, 3, &StageTimes::default().to_json());
        let text = std::fs::read_to_string(dir.join(".cortex/events.jsonl")).unwrap();
        let line = text.lines().next().unwrap();
        let v: serde_json::Value = serde_json::from_str(line).unwrap();
        assert_eq!(v["kind"], "recall");
        assert_eq!(v["tokens"], 42);
        assert_eq!(v["hits"][0]["tier"], "note");
        assert_eq!(v["stages"]["fts_ms"], 0);
        assert!(!v["query_hash"].as_str().unwrap().contains("jwt"));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn missing_dir_is_not_fatal() {
        let dir = std::env::temp_dir().join(format!(
            "cx_ev_none_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        // No .cortex dir — must silently do nothing.
        log_event(&dir, "recall", "q", &[], 0, 0, &serde_json::json!({}));
    }
}
