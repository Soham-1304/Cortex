use std::io::{BufRead, BufReader, Write};
use serde_json::{json, Value};
use crate::Result;
use crate::corefile::CoreFile;
use crate::session::Session;
use crate::store::Store;
use std::path::Path;

fn tool_defs() -> Value {
    // CodeGraph pattern: one strong tool beats a menu. All tools stay
    // functional, but only the allow-listed ones are advertised — set
    // CORTEX_MCP_TOOLS=compile_context (or a csv subset) to steer the
    // agent to a single entry point. Default lists everything (no
    // behavior change for existing agents).
    let allow: Option<Vec<String>> = std::env::var("CORTEX_MCP_TOOLS").ok().map(|v| {
        v.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect()
    });
    let all = json!([
        {
            "name": "compile_context",
            "description": "Answer structural/code questions directly with this — it IS the pre-built index (relevant memories + symbols with source, callers/callees, related files) within a token budget. Treat returned source as already read; do NOT re-verify with grep. If the response carries a staleness notice, Read the named files directly.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "task": {"type": "string"},
                    "token_budget": {"type": "number"}
                },
                "required": ["task"]
            }
        },
        {
            "name": "memory_write",
            "description": "Write a memory note with optional entities and links.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "content": {"type": "string"},
                    "kind": {"type": "string"},
                    "entities": {"type": "array", "items": {"type": "string"}},
                    "links": {"type": "array", "items": {"type": "object", "properties": {"src": {"type": "string"}, "dst": {"type": "string"}, "kind": {"type": "string"}}, "required": ["src","dst"]}}
                },
                "required": ["content"]
            }
        },
        {
            "name": "memory_search",
            "description": "Search memory notes by keyword query.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "n": {"type": "number"}
                },
                "required": ["query"]
            }
        },
        {
            "name": "session_digest",
            "description": "Digest the current session with an optional summary and token count.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string"},
                    "tokens_used": {"type": "number"}
                },
                "required": []
            }
        },
        {
            "name": "core_read",
            "description": "Read the CORTEX.md core file.",
            "inputSchema": {
                "type": "object",
                "properties": {},
                "required": []
            }
        },
        {
            "name": "core_write",
            "description": "Write a section to the CORTEX.md core file.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "section": {"type": "string"},
                    "content": {"type": "string"}
                },
                "required": ["section","content"]
            }
        },
        {
            "name": "feedback",
            "description": "Rate a recalled note: useful strengthens it, useless distrusts it (reversible).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "note_id": {"type": "number"},
                    "verdict": {"type": "string", "enum": ["useful", "useless"]}
                },
                "required": ["note_id", "verdict"]
            }
        },
        {
            "name": "code_find",
            "description": "Exact identifier lookup: symbol -> (file, line, defs, call sites, usages) with out-of-file connections. Fast path that skips embeddings. Use for pinpointing definitions and callers.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "identifier or symbol name to search"},
                    "n": {"type": "number", "description": "max results (default 10)"}
                },
                "required": ["query"]
            }
        },
        {
            "name": "code_map",
            "description": "File roadmap: entry points, imports, and inbound references for files around query keywords (<=1k tokens).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "file path or identifier keyword"}
                },
                "required": ["query"]
            }
        },
        {
            "name": "reindex",
            "description": "Re-embed all notes and symbols after changing embedding provider/dim.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "embed": {"type": "string"}
                },
                "required": []
            }
        }
    ]);
    match allow {
        None => all,
        Some(list) => {
            let arr = all.as_array().cloned().unwrap_or_default();
            Value::Array(arr.into_iter().filter(|t| {
                t.get("name").and_then(|n| n.as_str()).map(|n| list.iter().any(|a| a == n)).unwrap_or(false)
            }).collect())
        }
    }
}

fn open_store() -> Result<Store> {
    let base = std::env::current_dir().unwrap_or_else(|_| Path::new(".").to_path_buf());
    let db_path = base.join(".cortex/memory.db");
    if let Some(parent) = db_path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    Store::open_at(&db_path.to_string_lossy())
}

fn core_path() -> String {
    let base = std::env::current_dir().unwrap_or_else(|_| Path::new(".").to_path_buf());
    base.join("CORTEX.md").to_string_lossy().to_string()
}

fn session_id() -> String {
    if let Ok(s) = std::env::var("CORTEX_SESSION") {
        if !s.trim().is_empty() {
            return s;
        }
    }
    format!("mcp-{}", std::process::id())
}

fn ok_result(id: Value, text: String) -> Value {
    json!({"jsonrpc":"2.0","id":id,"result":{"content":[{"type":"text","text":text}]}})
}

fn err_result(id: Value, msg: String) -> Value {
    let text = json!({"error": msg}).to_string();
    json!({"jsonrpc":"2.0","id":id,"result":{"content":[{"type":"text","text":text}],"isError":true}})
}

fn dispatch(name: &str, args: &Value, id: Value) -> Value {
    let res: std::result::Result<String, String> = (|| {
        match name {
            "compile_context" => {
                let task = args.get("task").and_then(|v| v.as_str()).ok_or("missing task")?.to_string();
                let budget = args.get("token_budget").and_then(|v| v.as_i64()).unwrap_or(4000);
                let store = open_store().map_err(|e| e.to_string())?;
                let cp = core_path();
                let core = CoreFile::load(&cp).unwrap_or_else(|_| CoreFile { path: cp.clone(), preamble: None, sections: Default::default() });
                let base = std::env::current_dir().unwrap_or_else(|_| Path::new(".").to_path_buf());
                // No silent hash fallback: mixed dims corrupt ranking. Fail loud.
                let provider: Box<dyn crate::embed::EmbeddingProvider> = crate::embed::registry::load_for_base(&base).map_err(|e| format!("bad embedding config ({e}): run `cortex init --reconfigure`"))?;
                if let Some(db_dim) = store.get_meta("embedding_dim").map_err(|e| e.to_string())? {
                    if db_dim != provider.dim().to_string() {
                        return Err(format!("embedding dim mismatch (db={}, config={}): run cortex reindex", db_dim, provider.dim()));
                    }
                }
                let mut payload = crate::compile::compile(&store, &core, provider.as_ref(), &task, budget).map_err(|e| e.to_string())?;
                // Event log + recall feedback (items 1 & 4), best-effort.
                let hits: Vec<crate::events::EventHit> = payload
                    .memories
                    .iter()
                    .map(|m| crate::events::EventHit {
                        id: format!("note:{}", m.id),
                        tier: "note".into(),
                        score: m.score,
                    })
                    .collect();
                // Stale-index notice (B-2): fulfills the tool description's
                // staleness contract; best-effort, capped.
                payload.stale = crate::cli::stale_quick(&store, &crate::cli::index_root_for(&store, &base));
                crate::events::log_event(
                    &base, "mcp_compile", &task, &hits,
                    payload.tokens_used, payload.took_ms,
                    &serde_json::to_value(&payload.stages).unwrap_or_default(),
                );
                let note_ids: Vec<i64> = payload.memories.iter().map(|m| m.id).collect();
                if !note_ids.is_empty() {
                    let _ = store.reinforce_recalled(&note_ids);
                }
                Ok(serde_json::to_string(&payload).map_err(|e| e.to_string())?)
            }
            "memory_write" => {
                let content = args.get("content").and_then(|v| v.as_str()).ok_or("missing content")?.to_string();
                let kind = args.get("kind").and_then(|v| v.as_str()).unwrap_or("note").to_string();
                let entities: Vec<String> = args.get("entities").and_then(|v| v.as_array()).map(|a| a.iter().filter_map(|x| x.as_str().map(|s| s.to_string())).collect()).unwrap_or_default();
                let links_raw = args.get("links").and_then(|v| v.as_array()).cloned().unwrap_or_default();
                let links_owned: Vec<(String,String)> = links_raw.iter().filter_map(|v| {
                    let src = v.get("src")?.as_str()?.to_string();
                    let dst = v.get("dst")?.as_str()?.to_string();
                    Some((src,dst))
                }).collect();
                let link_refs: Vec<(&str,&str)> = links_owned.iter().map(|(a,b)| (a.as_str(), b.as_str())).collect();
                let store = open_store().map_err(|e| e.to_string())?;
                let sid = session_id();
                let mut sess = Session::start(&store, &sid).map_err(|e| e.to_string())?;
                let base = std::env::current_dir().unwrap_or_else(|_| Path::new(".").to_path_buf());
                // No silent hash fallback (see compile_context above).
                let provider: Box<dyn crate::embed::EmbeddingProvider> = crate::embed::registry::load_for_base(&base).map_err(|e| format!("bad embedding config ({e}): run `cortex init --reconfigure`"))?;
                let nid = sess.write(&content, &kind, &entities, &link_refs, provider.as_ref()).map_err(|e| e.to_string())?;
                Ok(json!({"id": nid}).to_string())
            }
            "memory_search" => {
                let query = args.get("query").and_then(|v| v.as_str()).ok_or("missing query")?.to_string();
                let n = args.get("n").and_then(|v| v.as_u64()).unwrap_or(10) as usize;
                let store = open_store().map_err(|e| e.to_string())?;
                let hits = store.search_notes(&query, n).map_err(|e| e.to_string())?;
                let arr: Vec<Value> = hits.into_iter().map(|(id, snippet)| json!({"id": id, "snippet": snippet})).collect();
                Ok(json!(arr).to_string())
            }
            "session_digest" => {
                let summary = args.get("summary").and_then(|v| v.as_str()).unwrap_or("").to_string();
                let tokens = args.get("tokens_used").and_then(|v| v.as_i64()).unwrap_or(0);
                let store = open_store().map_err(|e| e.to_string())?;
                let sid = session_id();
                let mut sess = Session::start(&store, &sid).map_err(|e| e.to_string())?;
                sess.digest(&summary, tokens).map_err(|e| e.to_string())?;
                Ok(json!({"ok": true}).to_string())
            }
            "core_read" => {
                let cp = core_path();
                let core = CoreFile::load(&cp).unwrap_or_else(|_| CoreFile { path: cp.clone(), preamble: None, sections: Default::default() });
                let content = core.render();
                let full = std::fs::read_to_string(&cp).unwrap_or(content.clone());
                let text = if full.trim().is_empty() { content } else { full };
                Ok(json!({"content": text}).to_string())
            }
            "core_write" => {
                let section = args.get("section").and_then(|v| v.as_str()).ok_or("missing section")?.to_string();
                let content = args.get("content").and_then(|v| v.as_str()).ok_or("missing content")?.to_string();
                let cp = core_path();
                let mut core = CoreFile::load(&cp).unwrap_or_else(|_| CoreFile { path: cp.clone(), preamble: None, sections: Default::default() });
                let body = if content.ends_with('\n') { content } else { content + "\n" };
                core.set_section(&section, &body);
                core.save().map_err(|e| e.to_string())?;
                Ok(json!({"ok": true, "requires_approval": true}).to_string())
            }
            "reindex" => {
                let embed = args.get("embed").and_then(|v| v.as_str()).map(|s| s.to_string());
                let base = std::env::current_dir().unwrap_or_else(|_| Path::new(".").to_path_buf());
                let db_path = base.join(".cortex/memory.db");
                let store = Store::open_at(&db_path.to_string_lossy()).map_err(|e| e.to_string())?;
                let provider: Box<dyn crate::embed::EmbeddingProvider> = if let Some(e) = embed {
                    let dim = match e.as_str() {
                        "local-nomic" | "ollama" | "google" => 768,
                        "mistral" => 1024,
                        "openai" => 3072,
                        _ => 64,
                    };
                    let model = e.clone();
                    let cfg = crate::config::EmbeddingConfig { provider: e.clone(), dim, model, ollama_url: None, api_key_env: None };
                    crate::embed::registry::from_config(&cfg).map_err(|e| e.to_string())?
                } else {
                    crate::embed::registry::load_for_base(&base).map_err(|e| format!("bad embedding config ({e}): run `cortex init --reconfigure`"))?
                };
                provider.health_check().map_err(|e| e.to_string())?;
                let mut note_rows: Vec<(i64, String)> = Vec::new();
                {
                    let mut st = store.conn.prepare("SELECT id, content FROM notes").map_err(|e| e.to_string())?;
                    let rows = st.query_map([], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?))).map_err(|e| e.to_string())?;
                    for r in rows { note_rows.push(r.map_err(|e| e.to_string())?); }
                }
                let mut sym_rows: Vec<(i64, String)> = Vec::new();
                {
                    let mut st = store.conn.prepare("SELECT id, COALESCE(embed_text, signature, name) FROM symbols").map_err(|e| e.to_string())?;
                    let rows = st.query_map([], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?))).map_err(|e| e.to_string())?;
                    for r in rows { sym_rows.push(r.map_err(|e| e.to_string())?); }
                }
                if !note_rows.is_empty() {
                    let texts: Vec<String> = note_rows.iter().map(|(_, c)| c.clone()).collect();
                    let vecs = provider.embed(texts).map_err(|e| e.to_string())?;
                    for ((id, _), vec) in note_rows.iter().zip(vecs.iter()) {
                        let blob = crate::store::serialize_f32s(vec);
                        store.conn.execute("UPDATE notes SET embedding=?1 WHERE id=?2", rusqlite::params![blob, id]).map_err(|e| e.to_string())?;
                    }
                }
                if !sym_rows.is_empty() {
                    let texts: Vec<String> = sym_rows.iter().map(|(_, s)| s.clone()).collect();
                    let vecs = provider.embed(texts).map_err(|e| e.to_string())?;
                    for ((id, _), vec) in sym_rows.iter().zip(vecs.iter()) {
                        let blob = crate::store::serialize_f32s(vec);
                        store.conn.execute("UPDATE symbols SET embedding=?1 WHERE id=?2", rusqlite::params![blob, id]).map_err(|e| e.to_string())?;
                    }
                }
                store.set_meta("embedding_dim", &provider.dim().to_string()).map_err(|e| e.to_string())?;
                Ok(json!({"reindexed": true, "notes": note_rows.len(), "symbols": sym_rows.len()}).to_string())
            }
            "feedback" => {
                let id = args.get("note_id").and_then(|v| v.as_i64()).ok_or("missing note_id")?;
                let verdict = args.get("verdict").and_then(|v| v.as_str()).ok_or("missing verdict")?;
                if verdict != "useful" && verdict != "useless" {
                    return Err(format!("verdict must be useful|useless, got {verdict}"));
                }
                let store = open_store().map_err(|e| e.to_string())?;
                let exists: i64 = store.conn.query_row("SELECT count(*) FROM notes WHERE id=?1", [id], |r| r.get(0)).map_err(|e| e.to_string())?;
                if exists == 0 {
                    return Err(format!("unknown note id {id}"));
                }
                if verdict == "useful" {
                    store.reinforce_recalled(&[id]).map_err(|e| e.to_string())?;
                } else {
                    store.demote_feedback(&[id]).map_err(|e| e.to_string())?;
                }
                Ok(json!({"id": id, "verdict": verdict}).to_string())
            }
            "code_find" => {
                let query = args.get("query").and_then(|v| v.as_str()).ok_or("missing query")?;
                let n = args.get("n").and_then(|v| v.as_u64()).unwrap_or(10) as usize;
                let store = open_store().map_err(|e| e.to_string())?;
                let mut tokens = vec![query.to_string()];
                tokens.extend(query.split_whitespace().map(|s| s.to_string()));
                tokens.sort();
                tokens.dedup();
                let mut seen: std::collections::HashSet<(String, i64)> = std::collections::HashSet::new();
                let mut defs = Vec::new();
                let mut uses = Vec::new();
                let mut fts = Vec::new();
                for tok in &tokens {
                    let (d, u, f) = store.find_refs(tok, n).map_err(|e| e.to_string())?;
                    for row in d {
                        if seen.insert((row.0.clone(), row.1)) {
                            defs.push(json!({"path": row.0, "line": row.1, "kind": row.2}));
                        }
                    }
                    for row in u {
                        if seen.insert((row.0.clone(), row.1)) {
                            uses.push(json!({"path": row.0, "line": row.1, "kind": row.2, "context": row.3}));
                        }
                    }
                    for row in f {
                        if seen.insert((row.0.clone(), row.1)) {
                            fts.push(json!({"path": row.0, "line": row.1, "context": row.2}));
                        }
                    }
                    if defs.len() + uses.len() >= n * 2 {
                        break;
                    }
                }
                let mut files: Vec<String> = defs.iter().chain(uses.iter()).filter_map(|x| x["path"].as_str().map(|s| s.to_string())).collect();
                files.sort();
                files.dedup();
                let mut conns = Vec::new();
                for f in files.iter().take(5) {
                    if let Ok((out, inc)) = store.file_connections(f, 5) {
                        conns.push(json!({"file": f, "imports": out, "referenced_by": inc}));
                    }
                }
                Ok(json!({"defs": defs, "uses": uses, "fts": fts, "connections": conns}).to_string())
            }
            "code_map" => {
                let query = args.get("query").and_then(|v| v.as_str()).ok_or("missing query")?;
                let store = open_store().map_err(|e| e.to_string())?;
                let toks: Vec<String> = query
                    .split(|c: char| !c.is_alphanumeric() && c != '_' && c != '.')
                    .filter(|s| s.len() > 1)
                    .map(|s| s.to_string())
                    .collect();
                let mut files: Vec<String> = Vec::new();
                for tok in &toks {
                    let (d, u, _) = store.find_refs(tok, 10).unwrap_or_default();
                    for (p, _, _) in d {
                        if !files.contains(&p) { files.push(p); }
                    }
                    for (p, _, _, _) in u {
                        if !files.contains(&p) { files.push(p); }
                    }
                    if files.len() >= 12 { break; }
                }
                if files.is_empty() {
                    let (_, _, fts) = store.find_refs(query, 10).unwrap_or_default();
                    for (p, _, _) in fts {
                        if !files.contains(&p) { files.push(p); }
                    }
                }
                let mut roadmap = Vec::new();
                for f in files.iter().take(10) {
                    let (out, inc) = store.file_connections(f, 6).unwrap_or_default();
                    roadmap.push(json!({"file": f, "imports": out, "referenced_by": inc}));
                }
                Ok(json!({"files": roadmap}).to_string())
            }
            _ => Err(format!("unknown tool {name}")),
        }
    })();
    match res {
        Ok(text) => ok_result(id, text),
        Err(msg) => err_result(id, msg),
    }
}

pub fn run_stdio() -> Result<()> {
    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout();
    let reader = BufReader::new(stdin.lock());
    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() { continue; }
        let req: Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(e) => {
                let resp = json!({"jsonrpc":"2.0","id":null,"error":{"code":-32700,"message":format!("parse error: {e}")}});
                writeln!(stdout, "{}", serde_json::to_string(&resp)?)?;
                stdout.flush()?;
                continue;
            }
        };
        let id = req.get("id").cloned();
        let method = req.get("method").and_then(|m| m.as_str()).unwrap_or("");
        let is_notification = id.is_none() || id == Some(Value::Null);
        if is_notification && method == "notifications/initialized" {
            continue;
        }
        if is_notification && !method.is_empty() {
            continue;
        }
        let response = match method {
            "initialize" => {
                let uid = id.clone().unwrap_or(Value::Null);
                json!({"jsonrpc":"2.0","id":uid,"result":{"protocolVersion":"2024-11-05","capabilities":{"tools":{}},"serverInfo":{"name":"cortex","version":"0.1.0"}}})
            }
            "tools/list" => {
                let uid = id.clone().unwrap_or(Value::Null);
                json!({"jsonrpc":"2.0","id":uid,"result":{"tools":tool_defs()}})
            }
            "tools/call" => {
                let uid = id.clone().unwrap_or(Value::Null);
                let params = req.get("params").cloned().unwrap_or(Value::Null);
                let name = params.get("name").and_then(|v| v.as_str()).unwrap_or("").to_string();
                let args = params.get("arguments").cloned().unwrap_or(json!({}));
                let known = [
                    "compile_context",
                    "memory_write",
                    "memory_search",
                    "session_digest",
                    "core_read",
                    "core_write",
                    "feedback",
                    "reindex",
                    "code_find",
                    "code_map",
                ];
                if !known.contains(&name.as_str()) {
                    let text = json!({"error": format!("unknown tool {name}")}).to_string();
                    json!({"jsonrpc":"2.0","id":uid,"result":{"content":[{"type":"text","text":text}],"isError":true}})
                } else {
                    dispatch(&name, &args, uid)
                }
            }
            _ if is_notification => continue,
            _ => {
                let uid = id.clone().unwrap_or(Value::Null);
                json!({"jsonrpc":"2.0","id":uid,"error":{"code":-32601,"message":format!("unknown method {method}")}})
            }
        };
        writeln!(stdout, "{}", serde_json::to_string(&response)?)?;
        stdout.flush()?;
    }
    Ok(())
}
