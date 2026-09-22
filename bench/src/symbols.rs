use std::sync::OnceLock;
use tree_sitter::{Parser, Query, QueryCursor, StreamingIterator};

#[derive(Debug, Clone, PartialEq)]
pub struct RawSymbol {
    pub path: String,
    pub name: String,
    pub kind: String,
    pub line_start: usize,
    pub line_end: usize,
    pub text: String,
    pub parent: Option<String>,
}

impl RawSymbol {
    pub fn full_text(&self) -> String {
        match &self.parent {
            Some(p) => format!("class {}:\n{}", p, self.text),
            None => self.text.clone(),
        }
    }
}

const PY_QUERY: &str = r#"
(class_definition name: (identifier) @cls.name body: (block) @cls.body) @cls.def
(function_definition name: (identifier) @fn.name parameters: (parameters) @fn.params body: (block) @fn.body) @fn.def
(call function: (identifier) @call.name) @call.expr
(call function: (attribute object: (_) attribute: (identifier) @call.attr)) @call.expr
(import_statement name: (dotted_name) @imp.name) @imp.stmt
(import_statement name: (aliased_import name: (dotted_name) @imp.name)) @imp.stmt
(import_from_statement module_name: (dotted_name) @from.mod) @from.stmt
(import_from_statement module_name: (relative_import) @from.mod) @from.stmt
"#;

const MAX_LINES: usize = 120;
const MIN_CHARS: usize = 30;

static QUERY: OnceLock<Option<Query>> = OnceLock::new();

fn compiled_query() -> Option<&'static Query> {
    QUERY
        .get_or_init(|| {
            let lang: tree_sitter::Language = tree_sitter_python::LANGUAGE.into();
            Query::new(&lang, PY_QUERY).ok()
        })
        .as_ref()
}

pub fn extract_file_symbols(path: &str, src: &str) -> Vec<RawSymbol> {
    let mut out = Vec::new();
    let mut parser = Parser::new();
    let lang: tree_sitter::Language = tree_sitter_python::LANGUAGE.into();
    if parser.set_language(&lang).is_err() {
        return regex_fallback(path, src);
    }
    let tree = match parser.parse(src, None) {
        Some(t) => t,
        None => return regex_fallback(path, src),
    };
    if tree.root_node().has_error() {
        return regex_fallback(path, src);
    }
    let Some(query) = compiled_query() else {
        return regex_fallback(path, src);
    };
    let lines: Vec<&str> = src.lines().collect();
    let mut cursor = QueryCursor::new();
    let mut matches = cursor.matches(query, tree.root_node(), src.as_bytes());
    let mut pending: Vec<RawSymbol> = Vec::new();
    while let Some(m) = matches.next() {
        let mut name = String::new();
        let mut is_cls = false;
        let mut body_node = None;
        let mut def_node = None;
        for cap in m.captures {
            let cname = query.capture_names()[cap.index as usize];
            if cname.ends_with(".name") {
                name = cap.node.utf8_text(src.as_bytes()).unwrap_or("").to_string();
            }
            if cname == "cls.body" || cname == "fn.body" {
                is_cls = cname == "cls.body";
                body_node = Some(cap.node);
            }
            if cname == "cls.def" || cname == "fn.def" {
                is_cls = cname == "cls.def";
                def_node = Some(cap.node);
            }
        }
        let n = match def_node.or(body_node) {
            Some(n) => n,
            None => continue,
        };
        if name.is_empty() {
            continue;
        }
        let ls = n.start_position().row + 1;
        let le = n.end_position().row + 1;
        let text: String = lines.get(ls - 1..le.min(lines.len())).unwrap_or(&[]).join("\n");
        pending.push(RawSymbol {
            path: path.to_string(),
            name,
            kind: if is_cls { "class".into() } else { "function".into() },
            line_start: ls,
            line_end: le,
            text,
            parent: None,
        });
    }
    let classes: Vec<(String, usize, usize)> = pending.iter()
        .filter(|s| s.kind == "class")
        .map(|s| (s.name.clone(), s.line_start, s.line_end)).collect();
    for s in pending.iter_mut() {
        if s.kind == "function" {
            if let Some((cls, _, _)) = classes.iter().find(|(_, a, b)| s.line_start >= *a && s.line_end <= *b) {
                s.kind = "method".into();
                s.parent = Some(cls.clone());
            }
        }
        let ls = s.line_start;
        s.line_end = s.line_end.min(ls + MAX_LINES - 1);
        out.push(s.clone());
    }
    out.retain(|s| s.text.len() >= MIN_CHARS);
    if out.is_empty() {
        return regex_fallback(path, src);
    }
    out
}

/// Reference extraction over the same parse as symbols: call sites
/// (`foo()`, `mod.foo()`), plain + aliased imports, and from-imports.
/// Attribute calls emit one ref for the full `recv.attr` name AND one for the
/// short `attr` name so exact-identifier lookup hits either spelling.
pub fn extract_file_refs(path: &str, src: &str) -> Vec<RawRef> {
    let mut parser = Parser::new();
    let lang: tree_sitter::Language = tree_sitter_python::LANGUAGE.into();
    if parser.set_language(&lang).is_err() {
        return regex_refs_fallback(path, src);
    }
    let tree = match parser.parse(src, None) {
        Some(t) => t,
        None => return regex_refs_fallback(path, src),
    };
    if tree.root_node().has_error() {
        return regex_refs_fallback(path, src);
    }
    let Some(query) = compiled_query() else {
        return regex_refs_fallback(path, src);
    };
    let lines: Vec<&str> = src.lines().collect();
    let mut out = Vec::new();
    let mut cursor = QueryCursor::new();
    let mut matches = cursor.matches(query, tree.root_node(), src.as_bytes());
    while let Some(m) = matches.next() {
        let mut call_name: Option<String> = None;
        let mut call_attr: Option<String> = None;
        let mut imp_name: Option<String> = None;
        let mut from_mod: Option<String> = None;
        let mut line = 0usize;
        for cap in m.captures {
            let cname = query.capture_names()[cap.index as usize];
            let text = cap.node.utf8_text(src.as_bytes()).unwrap_or("").to_string();
            match cname {
                "call.name" => {
                    call_name = Some(text);
                    line = cap.node.start_position().row + 1;
                }
                "call.attr" => {
                    call_attr = Some(text);
                    line = cap.node.start_position().row + 1;
                }
                "imp.name" => {
                    imp_name = Some(text);
                    line = cap.node.start_position().row + 1;
                }
                "from.mod" => {
                    from_mod = Some(text);
                    line = cap.node.start_position().row + 1;
                }
                _ => {}
            }
        }
        if line == 0 {
            continue;
        }
        let ctx = line_context(&lines, line);
        if let Some(n) = call_name {
            if !n.is_empty() {
                out.push(RawRef {
                    path: path.to_string(),
                    name: n,
                    kind: "call".into(),
                    line,
                    context: ctx,
                    target: None,
                });
            }
            continue;
        }
        if let Some(attr) = call_attr {
            if !attr.is_empty() {
                // Full-name ref from the source line: `<recv>.<attr>(`.
                let full = match ctx.find(&format!("{attr}(")) {
                    Some(pos) => {
                        let before = ctx[..pos].trim_end();
                        let start = before
                            .rfind(|c: char| {
                                !(c.is_alphanumeric() || c == '_' || c == '.')
                            })
                            .map(|i| i + 1)
                            .unwrap_or(0);
                        let recv = before[start..]
                            .trim_matches(|c: char| c == '.' || c == ' ');
                        format!("{recv}.{attr}")
                    }
                    None => attr.clone(),
                };
                let full = full.trim_start_matches('.').to_string();
                out.push(RawRef {
                    path: path.to_string(),
                    name: full,
                    kind: "call".into(),
                    line,
                    context: ctx.clone(),
                    target: None,
                });
                out.push(RawRef {
                    path: path.to_string(),
                    name: attr,
                    kind: "call".into(),
                    line,
                    context: ctx,
                    target: None,
                });
            }
            continue;
        }
        if let Some(n) = imp_name {
            if !n.is_empty() {
                // `import a.b.c` — index full path and top segment.
                out.push(RawRef {
                    path: path.to_string(),
                    name: n.clone(),
                    kind: "import".into(),
                    line,
                    context: ctx.clone(),
                    target: None,
                });
                if let Some(top) = n.split('.').next() {
                    if top != n.as_str() {
                        out.push(RawRef {
                            path: path.to_string(),
                            name: top.to_string(),
                            kind: "import".into(),
                            line,
                            context: ctx,
                            target: Some(n),
                        });
                    }
                }
            }
            continue;
        }
        if let Some(mo) = from_mod {
            if !mo.is_empty() {
                out.push(RawRef {
                    path: path.to_string(),
                    name: mo,
                    kind: "import_from".into(),
                    line,
                    context: ctx,
                    target: None,
                });
            }
        }
    }
    if out.is_empty() {
        return regex_refs_fallback(path, src);
    }
    out
}

fn regex_fallback(path: &str, src: &str) -> Vec<RawSymbol> {
    let mut out = Vec::new();
    let lines: Vec<&str> = src.lines().collect();
    let mut marks: Vec<(usize, String, String)> = Vec::new();
    for (i, line) in lines.iter().enumerate() {
        let t = line.trim_start();
        let name = if let Some(r) = t.strip_prefix("def ") {
            r.chars().take_while(|c| c.is_alphanumeric() || *c == '_').collect::<String>()
        } else if let Some(r) = t.strip_prefix("class ") {
            r.chars().take_while(|c| c.is_alphanumeric() || *c == '_').collect::<String>()
        } else {
            String::new()
        };
        if !name.is_empty() {
            marks.push((i + 1, name, if t.starts_with("class") { "class".into() } else { "function".into() }));
        }
    }
    for (idx, (ln, name, kind)) in marks.iter().enumerate() {
        let end = if idx + 1 < marks.len() { marks[idx + 1].0 - 1 } else { lines.len() };
        let end = (*ln + MAX_LINES - 1).min(end);
        let text: String = lines.get(ln - 1..end.min(lines.len())).unwrap_or(&[]).join("\n");
        if text.len() < MIN_CHARS {
            continue;
        }
        out.push(RawSymbol { path: path.into(), name: name.clone(), kind: kind.clone(), line_start: *ln, line_end: end, text, parent: None });
    }
    out
}

#[derive(Debug, Clone, PartialEq)]
pub struct RawRef {
    pub path: String,
    pub name: String,
    /// "def" | "call" | "import" | "import_from"
    pub kind: String,
    pub line: usize,
    /// trimmed single source line (max 200 chars)
    pub context: String,
    /// for import_from: source module; for call `a.b()`: receiver text
    pub target: Option<String>,
}

fn line_context(lines: &[&str], row_1based: usize) -> String {
    let mut s = lines.get(row_1based.saturating_sub(1)).unwrap_or(&"").trim().to_string();
    if s.len() > 200 {
        s.truncate(200);
    }
    s
}

fn regex_refs_fallback(path: &str, src: &str) -> Vec<RawRef> {
    let mut out = Vec::new();
    for (i, line) in src.lines().enumerate() {
        let ln = i + 1;
        let t = line.trim();
        let mut ctx = t.to_string();
        if ctx.len() > 200 {
            ctx.truncate(200);
        }
        if let Some(rest) = t.strip_prefix("import ") {
            for part in rest.split(',') {
                let seg = part.split_whitespace().next().unwrap_or("");
                let name = seg.split('.').next().unwrap_or("").to_string();
                if !name.is_empty()
                    && name.chars().all(|c| c.is_alphanumeric() || c == '_')
                {
                    out.push(RawRef {
                        path: path.to_string(),
                        name,
                        kind: "import".into(),
                        line: ln,
                        context: ctx.clone(),
                        target: None,
                    });
                }
            }
            continue;
        }
        if let Some(rest) = t.strip_prefix("from ") {
            if let Some((mo, _)) = rest.split_once(" import ") {
                let mo = mo.trim().to_string();
                if !mo.is_empty() {
                    out.push(RawRef {
                        path: path.to_string(),
                        name: mo,
                        kind: "import_from".into(),
                        line: ln,
                        context: ctx.clone(),
                        target: None,
                    });
                }
            }
            continue;
        }
        if t.starts_with("def ") || t.starts_with("class ") {
            continue;
        }
        // Best-effort call scan: identifier directly followed by `(`.
        let bytes = t.as_bytes();
        let mut j = 0;
        while j < bytes.len() {
            if bytes[j].is_ascii_alphabetic() || bytes[j] == b'_' {
                let s = j;
                while j < bytes.len()
                    && (bytes[j].is_ascii_alphanumeric() || bytes[j] == b'_')
                {
                    j += 1;
                }
                let name = &t[s..j];
                let mut k = j;
                while k < bytes.len() && bytes[k].is_ascii_whitespace() {
                    k += 1;
                }
                if k < bytes.len() && bytes[k] == b'('
                    && !matches!(
                        name,
                        "if" | "for" | "while" | "with" | "return" | "assert"
                            | "raise" | "yield" | "print" | "len" | "isinstance"
                    )
                {
                    out.push(RawRef {
                        path: path.to_string(),
                        name: name.to_string(),
                        kind: "call".into(),
                        line: ln,
                        context: ctx.clone(),
                        target: None,
                    });
                }
            } else {
                j += 1;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn extracts_fn_with_docstring_and_class_context() {
        let src = "class Foo:\n    \"\"\"Does things.\"\"\"\n    def bar(self, x):\n        \"\"\"Adds one.\"\"\"\n        return x + 1\n";
        let syms = extract_file_symbols("pkg/a.py", src);
        assert_eq!(syms.len(), 2);
        let bar = syms.iter().find(|s| s.name == "bar").unwrap();
        assert_eq!(bar.kind, "method");
        assert_eq!(bar.parent, Some("Foo".to_string()));
        assert_eq!((bar.line_start, bar.line_end), (3, 5));
        let src_lines: Vec<&str> = src.lines().collect();
        assert_eq!(bar.text.lines().collect::<Vec<_>>(), src_lines[bar.line_start - 1..bar.line_end]);
        assert!(bar.full_text().contains("class Foo:"));
        assert!(bar.full_text().contains("def bar(self, x):"));
        assert!(bar.full_text().contains("Adds one."));
        let foo = syms.iter().find(|s| s.name == "Foo").unwrap();
        assert_eq!(foo.kind, "class");
        assert_eq!(foo.parent, None);
        assert_eq!((foo.line_start, foo.line_end), (1, 5));
        assert_eq!(foo.text.lines().collect::<Vec<_>>(), src_lines[foo.line_start - 1..foo.line_end]);
    }

    #[test]
    fn fallback_on_garbage_and_empty() {
        assert!(extract_file_symbols("a.py", "").is_empty());
        let garbage = "def foo(a):\n    \"\"\"Docstring padding here.\"\"\"\n    return a + 1\nx = (1 +\n";
        let syms = extract_file_symbols("a.py", garbage);
        assert_eq!(syms.len(), 1);
        assert_eq!(syms[0].name, "foo");
        assert_eq!(syms[0].kind, "function");
        assert!(syms[0].text.contains("return a + 1"));
    }

    #[test]
    fn refs_capture_calls_and_imports() {
        let src = "import os\nimport a.b.c\nfrom x.y import z\n\ndef f():\n    y = validate_token(1)\n    mod.helper(y)\n    return len(y)\n";
        let refs = extract_file_refs("pkg/a.py", src);
        let has = |name: &str, kind: &str| {
            refs.iter().any(|r| r.name == name && r.kind == kind)
        };
        assert!(has("os", "import"), "refs: {refs:?}");
        assert!(has("a.b.c", "import"));
        assert!(has("a", "import"));
        assert!(has("x.y", "import_from"));
        assert!(has("validate_token", "call"));
        assert!(has("mod.helper", "call"));
        assert!(has("helper", "call"));
        assert!(has("len", "call"));
        // No def-line leakage: `f` itself must not appear as a call.
        assert!(!has("f", "call"));
        let call = refs.iter().find(|r| r.name == "validate_token").unwrap();
        assert_eq!(call.line, 6);
        assert!(call.context.contains("validate_token(1)"));
        let top = refs.iter().find(|r| r.name == "a" && r.target.is_some()).unwrap();
        assert_eq!(top.target.as_deref(), Some("a.b.c"));
    }

    #[test]
    fn refs_fallback_on_garbage_and_empty() {
        assert!(extract_file_refs("a.py", "").is_empty());
        // Garbage with an unbalanced paren has a tree-sitter error node AND
        // no valid call/import pattern — honestly empty, not a failure.
        let garbage = "def foo(a):\n    return a + 1\nx = (1 +\n";
        let refs = extract_file_refs("a.py", garbage);
        assert!(refs.is_empty(), "garbage refs: {refs:?}");
        let src = "import sys\nfrom m import n\nval = compute(x)\n";
        let refs = extract_file_refs("a.py", src);
        assert!(refs.iter().any(|r| r.name == "sys" && r.kind == "import"));
        assert!(refs.iter().any(|r| r.name == "m" && r.kind == "import_from"));
        assert!(refs.iter().any(|r| r.name == "compute" && r.kind == "call"));
    }
}
