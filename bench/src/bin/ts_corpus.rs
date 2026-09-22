use std::path::{Path, PathBuf};

fn usage() -> ! {
    eprintln!("usage: ts_corpus --repo <dir> --tasks <tasks.jsonl> --out <dir>");
    std::process::exit(2);
}

fn main() {
    let mut repo: Option<String> = None;
    let mut tasks: Option<String> = None;
    let mut out: Option<String> = None;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--repo" => repo = args.next(),
            "--tasks" => tasks = args.next(),
            "--out" => out = args.next(),
            _ => usage(),
        }
    }
    let (Some(repo), Some(tasks), Some(out)) = (repo, tasks, out) else {
        usage()
    };
    let repo_root = PathBuf::from(&repo);
    if !repo_root.is_dir() {
        eprintln!("error: --repo '{repo}' not a directory");
        std::process::exit(2);
    }
    let mut files: Vec<PathBuf> = Vec::new();
    collect_py(&repo_root, &mut files);
    if files.is_empty() {
        eprintln!("error: no .py files found under '{repo}'");
        std::process::exit(1);
    }
    files.sort();
    let mut docs: Vec<serde_json::Value> = Vec::new();
    for f in &files {
        let src = match std::fs::read_to_string(f) {
            Ok(s) => s,
            Err(_) => continue,
        };
        let rel = f
            .strip_prefix(&repo_root)
            .unwrap_or(f)
            .to_string_lossy()
            .replace('\\', "/");
        for s in bench::symbols::extract_file_symbols(&rel, &src) {
            docs.push(serde_json::json!({
                "id": format!("{}:{}:{}", s.path, s.name, s.line_start),
                "path": s.path,
                "text": s.full_text(),
                "line_start": s.line_start,
                "line_end": s.line_end,
            }));
        }
    }
    let out_dir = Path::new(&out);
    if let Err(e) = std::fs::create_dir_all(out_dir) {
        eprintln!("error: create out dir '{out}': {e}");
        std::process::exit(1);
    }
    if let Err(e) = std::fs::copy(&tasks, out_dir.join("tasks.jsonl")) {
        eprintln!("error: copy '{tasks}' to '{out}/tasks.jsonl': {e}");
        std::process::exit(1);
    }
    let corpus = match serde_json::to_string(&serde_json::json!({ "docs": docs })) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("error: serialize corpus: {e}");
            std::process::exit(1);
        }
    };
    if let Err(e) = std::fs::write(out_dir.join("corpus_raw.json"), corpus) {
        eprintln!("error: write '{out}/corpus_raw.json': {e}");
        std::process::exit(1);
    }
    eprintln!("wrote {} docs", docs.len());
}

const SKIP_DIRS: &[&str] = &[
    "test",
    "tests",
    "docs",
    "examples",
    "node_modules",
    ".git",
];

fn collect_py(dir: &Path, out: &mut Vec<PathBuf>) {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };
    for e in entries.flatten() {
        let p = e.path();
        let name = e.file_name().to_string_lossy().into_owned();
        if p.is_dir() {
            if SKIP_DIRS.contains(&name.as_str()) {
                continue;
            }
            collect_py(&p, out);
        } else if p.extension().map(|x| x == "py").unwrap_or(false) {
            out.push(p);
        }
    }
}
