#!/usr/bin/env python3
"""Fetch ContextBench tasks + build a symbol-doc corpus for the bench harness.

Usage:
  python3 scripts/fetch_contextbench.py --repo astropy/astropy --limit 20 \
      --out ../fixtures/contextbench [--clone]

Outputs into --out dir:
  corpus_raw.json  {"docs":[{id,path,text,line_start,line_end}...]}   (symbol docs)
  tasks.jsonl      {"instance_id","problem_statement","gold_context"} (ContextBench rows)
Stdlib only. Repo clone is optional; without it, symbol extraction runs on
whatever is already checked out in <out>/repos/<name>.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import urllib.request

ROWS_URL = ("https://datasets-server.huggingface.co/rows"
            "?dataset=Contextbench%2FContextBench&config=default&split=train")

DEF_PATTERNS = [
    re.compile(r"^(\s*)def\s+([A-Za-z_]\w*)\s*\("),
    re.compile(r"^(\s*)class\s+([A-Za-z_]\w*)\s*[:(]"),
    re.compile(r"^\s*(?:pub\s+)?(?:async\s+)?fn\s+([A-Za-z_]\w*)"),
    re.compile(r"^\s*(?:pub\s+)?(?:struct|enum|trait|interface)\s+([A-Za-z_]\w*)"),
    re.compile(r"^\s*func\s+(?:\([^)]*\)\s*)?([A-Za-z_]\w*)\s*\("),
    re.compile(r"^(\s*)(?:export\s+)?(?:async\s+)?function\s+([A-Za-z_]\w*)\s*\("),
    re.compile(r"^\s*(?:export\s+)?(const|let)\s+([A-Za-z_]\w*)\s*=\s*(?:async\s*)?\("),
]
EXTS = {".py", ".rs", ".go", ".java", ".js", ".ts", ".tsx"}
SKIP_DIRS = {"test", "tests", "docs", "examples", "node_modules", ".git"}
MAX_DOCS = 50000
MAX_LINES_PER_SPAN = 40


def fetch_rows(repo_filter, limit):
    rows, offset = [], 0
    while True:
        url = f"{ROWS_URL}&offset={offset}&length=100"
        with urllib.request.urlopen(url, timeout=60) as r:
            page = json.load(r)
        for row in page.get("rows", []):
            rr = row["row"]
            if repo_filter and repo_filter not in rr.get("repo", ""):
                continue
            rows.append(rr)
            if limit and len(rows) >= limit:
                return rows
        if not page.get("num_rows_per_page") or len(page.get("rows", [])) < 100:
            break
        offset += 100
    return rows


def ensure_repo(out_dir, repo_url, name, base_commit, do_clone):
    path = os.path.join(out_dir, "repos", name.replace("/", "__"))
    if not os.path.isdir(path):
        if not do_clone:
            sys.exit(f"repo not checked out at {path}; rerun with --clone")
        subprocess.run(["git", "clone", "--quiet", "--filter=blob:none",
                        repo_url, path], check=True)
    subprocess.run(["git", "-C", path, "checkout", "--quiet", base_commit], check=True)
    return path


def extract_docs(root):
    docs = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        rel = os.path.relpath(dirpath, root)
        for fn in sorted(filenames):
            ext = os.path.splitext(fn)[1].lower()
            if ext not in EXTS:
                continue
            fpath = os.path.join(dirpath, fn)
            rel_posix = os.path.normpath(os.path.join(rel, fn)).replace(os.sep, "/")
            try:
                with open(fpath, encoding="utf-8", errors="ignore") as fh:
                    lines = fh.read().splitlines()
            except OSError:
                continue
            marks = [(i + 1, m.groups()[-1])
                     for i, line in enumerate(lines)
                     for pat in DEF_PATTERNS
                     if (m := pat.match(line))]
            for idx, (lineno, name) in enumerate(marks):
                end = marks[idx + 1][0] - 1 if idx + 1 < len(marks) else len(lines)
                end = min(end, lineno + MAX_LINES_PER_SPAN)
                text = "\n".join(lines[lineno - 1:end]).strip()
                if len(text) < 30:
                    continue
                docs.append({
                    "id": f"{rel_posix}:{name}:{lineno}",
                    "path": rel_posix,
                    "text": text,
                    "line_start": lineno,
                    "line_end": end,
                })
                if len(docs) >= MAX_DOCS:
                    return docs
    return docs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--out", default="fixtures/contextbench")
    ap.add_argument("--clone", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print(f"fetching ContextBench rows for repo~'{args.repo}' ...")
    rows = fetch_rows(args.repo, args.limit)
    if not rows:
        sys.exit("no matching rows")
    with open(os.path.join(args.out, "tasks.jsonl"), "w") as fh:
        for rr in rows:
            fh.write(json.dumps({
                "instance_id": rr["instance_id"],
                "problem_statement": rr["problem_statement"],
                "gold_context": rr["gold_context"],
            }) + "\n")

    first = rows[0]
    root = ensure_repo(args.out, first["repo_url"], first["repo"],
                       first["base_commit"], args.clone)
    docs = extract_docs(root)
    with open(os.path.join(args.out, "corpus_raw.json"), "w") as fh:
        json.dump({"docs": docs}, fh)

    print(f"wrote {len(rows)} tasks and {len(docs)} symbol docs to {args.out}/")
    print("run: cargo run -q -- --fixtures", args.out, "--budget 200")


if __name__ == "__main__":
    main()
