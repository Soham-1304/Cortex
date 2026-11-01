#!/usr/bin/env python3
"""Decisive 72-Task Suite Runner for Cortex vs Baseline on Google Gemma 4 26B Instruct.

Orchestrates the head-to-head evaluation across 72 real tasks (12 per language across 6 languages):
- Evaluates Cortex Arm (cortex_search tool) vs Baseline Arm (grep / read_file)
- Full transparent audit logging:
  - bench/results/logs/episodes/<task_id>_<arm>.json
  - bench/results/logs/markdown/<task_id>_<arm>.md
  - bench/results/logs/index.html (live interactive viewer)
- Proactive Token Bucket Limiter to strictly respect Google AI Studio 16K TPM free quota
"""

import argparse
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
FIXTURES_FILE = ROOT / "bench" / "fixtures" / "decisive_72_tasks.jsonl"
LOGS_DIR = ROOT / "bench" / "results" / "logs"
EPISODES_DIR = LOGS_DIR / "episodes"
MARKDOWN_DIR = LOGS_DIR / "markdown"
INDEX_HTML = LOGS_DIR / "index.html"
OUT_RESULTS = ROOT / "bench" / "results" / "decisive_suite_results.jsonl"
CORTEX_BIN = os.environ.get("CORTEX_BIN", str(Path.home() / ".cargo" / "bin" / "cortex"))

# Add bench/scripts to sys.path so we can import agent_eval and agent_logger
sys.path.insert(0, str(ROOT / "bench" / "scripts"))
from agent_eval import Llm, run_episode, score
from agent_logger import EpisodeLogger, generate_interactive_html

# Local fixture directory map for pre-indexed repositories
REPO_MAP = {
    "astropy/astropy": (ROOT / "bench" / "fixtures" / "contextbench_astropy" / "repos" / "astropy__astropy",
                        ROOT / "bench" / "fixtures" / "contextbench_astropy" / "repos" / "astropy__astropy"),
    "django/django": (ROOT / "bench" / "fixtures" / "contextbench_django" / "repos" / "django__django",
                      ROOT / "bench" / "fixtures" / "contextbench_django" / "repos" / "django__django"),
    "sympy/sympy": (ROOT / "bench" / "fixtures" / "contextbench_sympy" / "repos" / "sympy__sympy",
                    ROOT / "bench" / "fixtures" / "contextbench_sympy" / "repos" / "sympy__sympy"),
    "clap-rs/clap": (ROOT / "bench" / "fixtures" / "contextbench_clap" / "repos" / "clap-rs__clap",
                     ROOT / "bench" / "fixtures" / "contextbench_clap" / "repos" / "clap-rs__clap"),
    "cli/cli": (ROOT / "bench" / "fixtures" / "contextbench_clicli" / "repos" / "cli__cli",
                ROOT / "bench" / "fixtures" / "contextbench_clicli" / "repos" / "cli__cli"),
    "iamkun/dayjs": (ROOT / "bench" / "fixtures" / "contextbench_dayjs" / "repos" / "iamkun__dayjs",
                     ROOT / "bench" / "fixtures" / "contextbench_dayjs" / "repos" / "iamkun__dayjs"),
    "google/gson": (ROOT / "bench" / "fixtures" / "contextbench_gson" / "repos" / "google__gson",
                    ROOT / "bench" / "fixtures" / "contextbench_gson" / "repos" / "google__gson"),
    "facebook/zstd": (ROOT / "bench" / "fixtures" / "contextbench_zstd" / "repos" / "facebook__zstd",
                      ROOT / "bench" / "fixtures" / "contextbench_zstd" / "repos" / "facebook__zstd"),
}

CACHE_REPOS_DIR = ROOT / "bench" / "fixtures" / "repos_cache"


def ensure_repo_and_index(repo_name, repo_url=None):
    """Ensures repository is checked out and has a .cortex index."""
    if repo_name in REPO_MAP:
        repo_dir, work_dir = REPO_MAP[repo_name]
    else:
        owner, rname = repo_name.split("/")
        repo_dir = CACHE_REPOS_DIR / f"{owner}__{rname}"
        work_dir = repo_dir
        if not repo_dir.is_dir():
            CACHE_REPOS_DIR.mkdir(parents=True, exist_ok=True)
            url = repo_url or f"https://github.com/{repo_name}.git"
            print(f"[CLONE] Cloning {repo_name} into {repo_dir}...")
            subprocess.run(["git", "clone", "--depth", "1", "--quiet", url, str(repo_dir)], check=True)

    db_path = work_dir / ".cortex" / "memory.db"
    cfg_path = work_dir / ".cortex" / "config.toml"
    if not db_path.is_file() or db_path.stat().st_size == 0 or not cfg_path.is_file():
        print(f"[INDEX] Building Cortex index for {repo_name} at {work_dir}...")
        (work_dir / ".cortex").mkdir(parents=True, exist_ok=True)
        with open(cfg_path, "w") as f:
            f.write('provider = "hash"\ndim = 64\nmodel = "hash"\n')
        subprocess.run([CORTEX_BIN, "index", ".", "--embed", "hash"], cwd=str(work_dir), check=True)

    return str(repo_dir), str(work_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/gemma-4-26b-a4b-it")
    ap.add_argument("--provider", default="gemini")
    ap.add_argument("--arms", default="tool,base")
    ap.add_argument("--language", default="ALL", help="Filter by language (e.g. Python, Rust, Go, or ALL)")
    ap.add_argument("--limit-per-lang", type=int, default=12)
    ap.add_argument("--pacing-delay", type=float, default=4.0, help="Seconds to sleep between calls to respect TPM quota")
    ap.add_argument("--dry-run", action="store_true", help="Print plan and exit without making API calls")
    args = ap.parse_args()

    if not FIXTURES_FILE.is_file():
        sys.exit(f"Fixtures file not found: {FIXTURES_FILE}")

    with open(FIXTURES_FILE, "r", encoding="utf-8") as f:
        all_tasks = [json.loads(line) for line in f if line.strip()]

    # Filter tasks by language
    if args.language != "ALL":
        all_tasks = [t for t in all_tasks if t["language"].lower() == args.language.lower()]

    if args.limit_per_lang and args.limit_per_lang < 12:
        import collections
        by_lang = collections.defaultdict(list)
        for t in all_tasks:
            if len(by_lang[t["language"]]) < args.limit_per_lang:
                by_lang[t["language"]].append(t)
        all_tasks = [t for tasks in by_lang.values() for t in tasks]

    arms = [a.strip() for a in args.arms.split(",")]
    EPISODES_DIR.mkdir(parents=True, exist_ok=True)
    MARKDOWN_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"CORTEX DECISIVE EVALUATION SUITE")
    print(f"Model: {args.model} ({args.provider}) | Arms: {arms}")
    print(f"Fixtures: {len(all_tasks)} tasks selected (Language filter: {args.language})")
    print(f"Logs: {LOGS_DIR}")
    print("=" * 80)

    if args.dry_run:
        print("\n[DRY RUN] Plan displayed. Exiting without execution.")
        return

    # Load existing done records
    done_records = {}
    if OUT_RESULTS.is_file():
        with open(OUT_RESULTS, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        rec = json.loads(line)
                        done_records[rec["id"]] = rec
                    except Exception:
                        pass
        print(f"[RESUME] Found {len(done_records)} previously completed tasks in {OUT_RESULTS}")

    llm = Llm(args.provider, args.model)
    completed = 0
    total = len(all_tasks)

    with open(OUT_RESULTS, "a", encoding="utf-8") as out_f:
        for idx, t in enumerate(all_tasks, 1):
            t_id = t["instance_id"]
            repo_name = t["repo"]
            lang = t["language"]
            domain = t.get("domain", "")

            if t_id in done_records:
                print(f"[{idx}/{total}] SKIP (already done): {t_id[-12:]} ({repo_name})", flush=True)
                continue

            print(f"\n[{idx}/{total}] TASK: {t_id[-12:]} | {lang} | {repo_name} ({domain})", flush=True)
            print(f"  Issue: {t['problem_statement'][:100].replace(chr(10), ' ')}...", flush=True)

            try:
                repo_path, work_path = ensure_repo_and_index(repo_name, t.get("repo_url"))
            except Exception as ex:
                print(f"  [ERROR] Failed to prepare repo {repo_name}: {ex}", flush=True)
                continue

            gold = t.get("gold_context", [])
            row = {
                "id": t_id,
                "repo": repo_name,
                "language": lang,
                "domain": domain,
                "model": args.model,
                "gold_files": len(t.get("gold_files", [])),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }

            # Evaluate arms in randomized order per task
            eval_order = arms[:]
            random.Random(t_id).shuffle(eval_order)

            for arm in eval_order:
                print(f"  --> Running Arm: {arm}...", end=" ", flush=True)
                clean_id = t_id.replace("/", "__")
                logger = EpisodeLogger(
                    task_id=t_id,
                    repo=repo_name,
                    language=lang,
                    arm=arm,
                    model=args.model,
                    problem_statement=t["problem_statement"],
                    gold_context=gold,
                )

                t_start = time.time()
                r = run_episode(llm, repo_path, work_path, t["problem_statement"], arm, max_steps=15, logger=logger)
                sc = score(r["final"], gold)
                r.update(sc)
                row[arm] = r

                logger.finish(r["final"], sc, r["error"])
                logger.save_json(EPISODES_DIR / f"{clean_id}_{arm}.json")
                logger.save_markdown(MARKDOWN_DIR / f"{clean_id}_{arm}.md")
                generate_interactive_html(EPISODES_DIR, INDEX_HTML)

                f1 = sc.get("file_f1", 0.0)
                sub = sc.get("submitted", False)
                retries = r.get("retries", 0)
                print(f"done in {r['wall_s']}s (cogn={logger.cognitive_s}s, 429_wait={logger.rate_limit_wait_s}s) | "
                      f"turns={r['turns']} tok={r['prompt_tokens']+r['completion_tokens']} | F1={f1:.2f} "
                      f"{'[SOLVED]' if f1 == 1.0 else '[PARTIAL]' if f1 > 0 else '[FAIL]'} "
                      f"{f'⚠️ {retries} retries' if retries else ''}", flush=True)

                # Pacing sleep between arms to prevent quota spikes
                if args.pacing_delay > 0:
                    time.sleep(args.pacing_delay)

            out_f.write(json.dumps(row) + "\n")
            out_f.flush()
            completed += 1

            # Pacing sleep between tasks
            if args.pacing_delay > 0:
                time.sleep(args.pacing_delay)

    print("\n" + "=" * 80, flush=True)
    print(f"SUITE COMPLETE: Evaluated {completed} tasks.", flush=True)
    print(f"Results saved to: {OUT_RESULTS}", flush=True)
    print(f"Interactive Dashboard updated: {INDEX_HTML}", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
