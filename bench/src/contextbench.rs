use serde::Deserialize;
use std::path::Path;

use crate::corpus::{Corpus, Doc, Query};

#[derive(Deserialize)]
struct CorpusRaw {
    docs: Vec<Doc>,
}

#[derive(Deserialize)]
struct TaskRow {
    instance_id: String,
    problem_statement: String,
    gold_context: String,
}

#[derive(Deserialize)]
struct GoldSpan {
    file: String,
    #[serde(default)]
    start_line: usize,
    #[serde(default)]
    end_line: usize,
}

pub fn spans_overlap(a0: usize, a1: usize, b0: usize, b1: usize) -> bool {
    if a0 == 0 && a1 == 0 {
        return true;
    }
    if b0 == 0 && b1 == 0 {
        return false;
    }
    a0 <= b1 && b0 <= a1
}

fn parse_gold_spans(raw: &str) -> Vec<GoldSpan> {
    serde_json::from_str::<Vec<GoldSpan>>(raw).unwrap_or_default()
}

pub fn load_dir(dir: &Path) -> Result<Corpus, String> {
    let raw_path = dir.join("corpus_raw.json");
    let tasks_path = dir.join("tasks.jsonl");

    let raw_text = std::fs::read_to_string(&raw_path)
        .map_err(|e| format!("read {}: {}", raw_path.display(), e))?;
    let parsed: CorpusRaw =
        serde_json::from_str(&raw_text).map_err(|e| format!("parse corpus_raw.json: {}", e))?;
    let docs = parsed.docs;

    let tasks_text = std::fs::read_to_string(&tasks_path)
        .map_err(|e| format!("read {}: {}", tasks_path.display(), e))?;

    let mut queries = Vec::new();
    let mut empty_tasks = 0usize;
    for line in tasks_text.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let task: TaskRow = serde_json::from_str(line)
            .map_err(|e| format!("parse tasks.jsonl line: {}", e))?;
        let spans = parse_gold_spans(&task.gold_context);
        let mut gold: Vec<String> = Vec::new();
        for span in &spans {
            let span_clean = span.file.trim_start_matches('/');
            for doc in &docs {
                let doc_clean = doc.path.trim_start_matches('/');
                let matches_path = doc_clean == span_clean
                    || span_clean.ends_with(doc_clean)
                    || doc_clean.ends_with(span_clean);
                if matches_path
                    && spans_overlap(
                        doc.line_start,
                        doc.line_end,
                        span.start_line,
                        span.end_line,
                    )
                    && !gold.contains(&doc.id)
                {
                    gold.push(doc.id.clone());
                }
            }
        }
        if gold.is_empty() {
            empty_tasks += 1;
            continue;
        }
        queries.push(Query {
            id: task.instance_id,
            text: task.problem_statement,
            gold,
        });
    }

    if queries.is_empty() {
        return Err("no task mapped to any symbol doc; check exporter output".to_string());
    }

    Ok(Corpus { docs, queries, _empty_tasks: empty_tasks })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overlap_basics() {
        assert!(spans_overlap(10, 20, 15, 25));
        assert!(spans_overlap(10, 20, 1, 12));
        assert!(spans_overlap(10, 20, 20, 30));
        assert!(!spans_overlap(10, 20, 21, 30));
        assert!(spans_overlap(0, 0, 5, 9));
        assert!(!spans_overlap(5, 9, 0, 0));
    }

    #[test]
    fn parses_gold_context_string() {
        let raw = r#"[{"file":"a.py","start_line":1,"end_line":10},{"file":"b.py","start_line":3,"end_line":4}]"#;
        let spans = parse_gold_spans(raw);
        assert_eq!(spans.len(), 2);
        assert_eq!(spans[0].file, "a.py");
        assert!(parse_gold_spans("not json").is_empty());
    }

    #[test]
    fn sample_export_end_to_end() {
        let dir = std::path::Path::new("fixtures/contextbench_sample");
        if !dir.exists() {
            return;
        }
        let c = load_dir(dir).expect("sample loads");
        assert!(!c.docs.is_empty());
        assert!(!c.queries.is_empty());
        for q in &c.queries {
            assert!(!q.gold.is_empty(), "query {} has no gold", q.id);
        }
    }
}
