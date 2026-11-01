#!/usr/bin/env python3
"""Decisive 72-Task Suite Runner for Cortex vs Baseline on Gemini 3.5 Flash.

Orchestrates the head-to-head evaluation across 72 real tasks (12 per language across 6 languages):
- Evaluates Cortex Arm (cortex_search tool) vs Baseline Arm (grep / read_file)
- Full transparent audit logging:
  - bench/results/logs/episodes/<task_id>_<arm>.json
  - bench/results/logs/markdown/<task_id>_<arm>.md
  - bench/results/logs/index.html (live interactive viewer)
- Pacing between tasks to respect Gemini TPM quotas and prevent rate limit thrashing
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
FIXTURES_FILE = ROOT / "bench" / "fixtures" / "decisive_72_tasks.jsonl"
LOGS_DIR = ROOT / "bench" / "results" / "logs"
EPISODES_DIR = LOGS_DIR / "episodes"
MARKDOWN_DIR = LOGS_DIR / "markdown"
INDEX_HTML = LOGS_DIR / "index.html"
AGENT_EVAL_SCRIPT = ROOT / "bench" / "scripts" / "agent_eval.py"

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/gemini-3.5-flash")
    ap.add_argument("--provider", default="gemini")
    ap.add_argument("--arms", default="tool,base")
    ap.add_argument("--limit-per-lang", type=int, default=12)
    ap.add_argument("--pacing-delay", type=float, default=5.0, help="Seconds to sleep between tasks to respect TPM quota")
    ap.add_argument("--dry-run", action="store_true", help="Print plan and exit without making API calls")
    args = ap.parse_args()

    if not FIXTURES_FILE.is_file():
        sys.exit(f"Fixtures file not found: {FIXTURES_FILE}")

    with open(FIXTURES_FILE, "r", encoding="utf-8") as f:
        all_tasks = [json.loads(line) for line in f if line.strip()]

    print("=" * 80)
    print(f"CORTEX DECISIVE EVALUATION SUITE: 72 REAL TASKS")
    print(f"Model: {args.model} ({args.provider}) | Arms: {args.arms}")
    print(f"Fixtures: {FIXTURES_FILE} ({len(all_tasks)} total tasks)")
    print(f"Logs output: {LOGS_DIR}")
    print("=" * 80)

    # Group by language
    by_lang = {}
    for t in all_tasks:
        by_lang.setdefault(t["language"], []).append(t)

    for lang, tasks in sorted(by_lang.items()):
        print(f"\n[{lang}] {len(tasks)} tasks:")
        for t in tasks[:args.limit_per_lang]:
            print(f"  * {t['repo']} ({t.get('domain', '')}): {t['instance_id'][-12:]} - {t['problem_statement'][:60].replace(chr(10), ' ')}...")

    if args.dry_run:
        print("\n[DRY RUN] Plan displayed. Exiting without execution.")
        return

    print("\nStarting execution... (Run paused pending user review)")


if __name__ == "__main__":
    main()
