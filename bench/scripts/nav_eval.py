#!/usr/bin/env python3
"""Cortex-side navigation eval (A/B refs on vs off) for one repo.

Usage:
  python3 nav_eval.py <fixtures_dir> <work_dir> <out_json>
    fixtures_dir: has tasks.jsonl ([{instance_id, problem_statement, gold_context}])
    work_dir:     cwd for `cortex recall` (its .cortex/memory.db is the index)
    out_json:     where to write {refs_on, refs_off} results

Index first (same provider for every arm under comparison):
  cd <work_dir> && cortex index <repo_checkout> --embed hash

Method (frozen): per issue, `cortex recall "<problem>" --json --budget 2000`.
Span hit = any recalled symbol overlapping a gold span; file hit = same file.
Run twice (CORTEX_REFS=1 vs 0), diff per task. Same binary, same gates on
every repo — no per-repo constants anywhere in this loop.

Stdlib only.
"""
import json
import os
import subprocess
import sys
import time

CORTEX = os.environ.get(
    "CORTEX_BIN",
    "/Users/sohamkarandikar/Documents/Cortex/target/release/cortex",
)


def spans_overlap(a0, a1, b0, b1):
    if a0 == 0 and a1 == 0:
        return True
    if b0 == 0 and b1 == 0:
        return False
    return a0 <= b1 and b0 <= a1


def run_recall(work_dir, problem, refs_on):
    env = dict(os.environ)
    env["CORTEX_REFS"] = "1" if refs_on else "0"
    r = subprocess.run(
        [CORTEX, "recall", problem, "--budget", "2000", "--json"],
        capture_output=True, text=True, env=env, cwd=work_dir, timeout=180,
    )
    out = r.stdout
    return json.loads(out[out.find("{"):  ])


def score(payload, task):
    syms = payload.get("symbols", [])
    spans = json.loads(task["gold_context"])
    span_hit = False
    file_hit = False
    for sp in spans:
        for s in syms:
            if s["path"] == sp["file"]:
                file_hit = True
                if spans_overlap(s["line_start"], s["line_end"],
                                 sp.get("start_line", 0), sp.get("end_line", 0)):
                    span_hit = True
    return span_hit, file_hit, len(syms)


def provenance(workdir):
    """Best-effort run tags: timestamp, binary identity, cortex version."""
    try:
        r = subprocess.run(
            [CORTEX, "--version"], capture_output=True, text=True, timeout=30,
        )
        raw = (r.stdout.strip() or r.stderr.strip() or "unknown").splitlines()[0]
        # clap version goes through our Error::Cli wrapper ("error: cli: ...").
        ver = raw.removeprefix("error: cli: ").strip() or "unknown"
    except Exception:
        ver = "unknown"
    try:
        st = os.stat(CORTEX)
        binary = {"path": CORTEX, "mtime": int(st.st_mtime), "size": st.st_size}
    except Exception:
        binary = {"path": CORTEX}
    return {
        "generated_at_unix": int(time.time()),
        "cortex_version": ver,
        "binary": binary,
        "budget": 2000,
        "work_dir": workdir,
    }


def main():
    fixdir, workdir, outpath = sys.argv[1], sys.argv[2], sys.argv[3]
    tasks = [json.loads(l) for l in open(f"{fixdir}/tasks.jsonl") if l.strip()]
    results = {"provenance": provenance(workdir), "fixtures_dir": fixdir}
    for refs_on in (True, False):
        tag = "refs_on" if refs_on else "refs_off"
        span_hits = 0
        file_hits = 0
        per_task = []
        for t in tasks:
            p = run_recall(workdir, t["problem_statement"], refs_on)
            sh, fh, nsym = score(p, t)
            span_hits += sh
            file_hits += fh
            per_task.append({"id": t["instance_id"], "span_hit": sh,
                             "file_hit": fh, "n_syms": nsym})
        n = len(tasks)
        results[tag] = {"nav_span": span_hits / n, "nav_file": file_hits / n,
                        "per_task": per_task}
        print(f"{tag}: nav-success(span) {span_hits}/{n} = {span_hits/n:.3f} "
              f"| nav(file) {file_hits}/{n} = {file_hits/n:.3f}", flush=True)
    json.dump(results, open(outpath, "w"), indent=1)
    for a, b in zip(results["refs_off"]["per_task"], results["refs_on"]["per_task"]):
        if a["span_hit"] != b["span_hit"] or a["file_hit"] != b["file_hit"]:
            print(f"FLIP {a['id']}: off(span={a['span_hit']},file={a['file_hit']})"
                  f" -> on(span={b['span_hit']},file={b['file_hit']})")


if __name__ == "__main__":
    main()
