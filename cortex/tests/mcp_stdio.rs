//! End-to-end MCP stdio tests against the built `cortex` binary.

use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};

fn bin() -> &'static str { env!("CARGO_BIN_EXE_cortex") }

#[test]
fn serve_respects_initialize_tools_list_and_call() {
    let dir = std::env::temp_dir().join(format!("cx_mcp_test_{}", std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
    std::fs::create_dir_all(&dir).unwrap();
    let _ = Command::new(bin()).arg("init").current_dir(&dir).output();
    let mut child = Command::new(bin())
        .args(["serve", "--stdio"])
        .stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::null())
        .current_dir(&dir)
        .spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let stdout = child.stdout.take().unwrap();
    let mut lines = BufReader::new(stdout).lines();
    writeln!(stdin, r#"{{"jsonrpc":"2.0","id":1,"method":"initialize","params":{{}}}}"#).unwrap();
    let r1: serde_json::Value = serde_json::from_str(&lines.next().unwrap().unwrap()).unwrap();
    assert_eq!(r1["id"], 1);
    assert!(r1["result"]["capabilities"]["tools"].is_object());
    writeln!(stdin, r#"{{"jsonrpc":"2.0","method":"notifications/initialized"}}"#).unwrap();
    std::thread::sleep(std::time::Duration::from_millis(50));
    writeln!(stdin, r#"{{"jsonrpc":"2.0","id":2,"method":"tools/list"}}"#).unwrap();
    let r2: serde_json::Value = serde_json::from_str(&lines.next().unwrap().unwrap()).unwrap();
    assert_eq!(r2["id"], 2);
    let tools = r2["result"]["tools"].as_array().unwrap();
    assert!(tools.iter().any(|t| t["name"] == "compile_context"));
    assert!(tools.iter().any(|t| t["name"] == "memory_write"));
    assert!(tools.iter().any(|t| t["name"] == "code_find"));
    assert!(tools.iter().any(|t| t["name"] == "code_map"));
    writeln!(stdin, r#"{{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{{"name":"memory_search","arguments":{{"query":"anything","n":3}}}}}}"#).unwrap();
    let r3: serde_json::Value = serde_json::from_str(&lines.next().unwrap().unwrap()).unwrap();
    assert_eq!(r3["id"], 3);
    assert!(r3["error"].is_null());
    writeln!(stdin, r#"{{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{{"name":"code_find","arguments":{{"query":"anything","n":2}}}}}}"#).unwrap();
    let r4: serde_json::Value = serde_json::from_str(&lines.next().unwrap().unwrap()).unwrap();
    assert_eq!(r4["id"], 4);
    assert!(r4["error"].is_null());
    drop(stdin);
    let _ = child.wait();
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn unknown_method_returns_error() {
    let dir = std::env::temp_dir().join(format!("cx_mcp_err_{}", std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
    std::fs::create_dir_all(&dir).unwrap();
    let _ = Command::new(bin()).arg("init").current_dir(&dir).output();
    let mut child = Command::new(bin())
        .args(["serve", "--stdio"]).stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::null())
        .current_dir(&dir).spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let stdout = child.stdout.take().unwrap();
    let mut lines = BufReader::new(stdout).lines();
    writeln!(stdin, r#"{{"jsonrpc":"2.0","id":99,"method":"no_such_method"}}"#).unwrap();
    let r: serde_json::Value = serde_json::from_str(&lines.next().unwrap().unwrap()).unwrap();
    assert_eq!(r["id"], 99);
    assert!(r["error"]["code"].as_i64().unwrap() == -32601);
    drop(stdin);
    let _ = child.wait();
    let _ = std::fs::remove_dir_all(&dir);
}
