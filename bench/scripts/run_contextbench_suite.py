#!/usr/bin/env python3
"""Run ContextBench Whole Suite: With Cortex vs. Without Cortex across all repos & languages.

Generates:
  bench/results/contextbench_detailed_runs.csv
  bench/results/contextbench_summary_by_repo.csv
  bench/results/contextbench_summary_by_language.csv
"""

import subprocess
import json
import os
import sys
import pandas as pd

FIXTURES = [
    {
        "id": "django",
        "name": "django/django",
        "lang": "Python",
        "path": "bench/fixtures/contextbench_django",
    },
    {
        "id": "sympy",
        "name": "sympy/sympy",
        "lang": "Python",
        "path": "bench/fixtures/contextbench_sympy",
    },
    {
        "id": "astropy",
        "name": "astropy/astropy",
        "lang": "Python",
        "path": "bench/fixtures/contextbench_astropy",
    },
    {
        "id": "clicli",
        "name": "cli/cli",
        "lang": "Go",
        "path": "bench/fixtures/contextbench_clicli",
    },
    {
        "id": "clap",
        "name": "clap-rs/clap",
        "lang": "Rust",
        "path": "bench/fixtures/contextbench_clap",
    },
    {
        "id": "dayjs",
        "name": "iamkun/dayjs",
        "lang": "JavaScript",
        "path": "bench/fixtures/contextbench_dayjs",
    },
]

BUDGETS = [1000]
BENCH_BIN = "target/release/bench"
OUT_DIR = "bench/results"
os.makedirs(OUT_DIR, exist_ok=True)

detailed_rows = []
summary_rows = []

print("=" * 80)
print("CONTEXTBENCH WHOLE SUITE EVALUATION: WITH CORTEX vs. WITHOUT CORTEX")
print("=" * 80)

for f in FIXTURES:
    repo_name = f["name"]
    lang = f["lang"]
    fpath = f["path"]

    if not os.path.exists(os.path.join(fpath, "tasks.jsonl")):
        print(f"Skipping {repo_name} (fixture not found at {fpath})")
        continue

    for budget in BUDGETS:
        for arm_name, fuse_flag in [
            ("Without Cortex (Baseline BM25)", "bm25"),
            ("With Cortex (Hybrid RRF + Knapsack)", "rrf"),
        ]:
            print(f"\n>>> Running {repo_name} [{lang}] | Budget: {budget} | Arm: {arm_name} ...")
            cmd = [
                BENCH_BIN,
                "--fixtures", fpath,
                "--budget", str(budget),
                "--embed", "hash",
                "--fuse", fuse_flag,
                "--json",
            ]

            try:
                res = subprocess.run(cmd, capture_output=True, text=True, check=True)
                stdout = res.stdout.strip()
                # Find start of JSON array
                start_idx = stdout.find("[")
                if start_idx == -1:
                    print(f"Error: JSON array not found in output: {stdout[:200]}")
                    continue
                data = json.loads(stdout[start_idx:])[0]
                agg = data.get("aggregate", {})
                detail = data.get("queries_detail", [])

                summary_rows.append({
                    "Repository": repo_name,
                    "Language": lang,
                    "Budget": budget,
                    "Arm": arm_name,
                    "Tasks_Evaluated": data.get("queries", len(detail)),
                    "File_Recall@20": agg.get("file_recall20", 0.0),
                    "Recall@20": agg.get("recall20", 0.0),
                    "nDCG@10": agg.get("ndcg10", 0.0),
                    "MRR": agg.get("mrr", 0.0),
                    "Packed_Precision": agg.get("packed_precision", 0.0),
                    "Packed_Recall": agg.get("packed_recall", 0.0),
                    "Avg_Packed_Tokens": agg.get("avg_packed_tokens", 0.0),
                    "Calibration_ECE": data.get("ece_calibrated", 0.0),
                })

                for q in detail:
                    detailed_rows.append({
                        "Task_ID": q.get("id"),
                        "Repository": repo_name,
                        "Language": lang,
                        "Budget": budget,
                        "Arm": arm_name,
                        "File_Recall@20": q.get("file_recall20", 0.0),
                        "Recall@20": q.get("recall20", 0.0),
                        "nDCG@10": q.get("ndcg10", 0.0),
                        "MRR": q.get("mrr", 0.0),
                        "Packed_Precision": q.get("packed_precision", 0.0),
                        "Packed_Recall": q.get("packed_recall", 0.0),
                        "Packed_Tokens": q.get("packed_tokens", 0),
                        "Budget_Efficiency_BCY": q.get("bcy", 0.0),
                        "Advisory": q.get("advisory", ""),
                    })

                print(f"    Done: FileRecall@20 = {agg.get('file_recall20', 0.0):.4f} | MRR = {agg.get('mrr', 0.0):.4f} | Packed = {agg.get('avg_packed_tokens', 0.0):.1f} tok")

            except Exception as e:
                print(f"    Failed: {e}")

# Save CSVs
df_detailed = pd.DataFrame(detailed_rows)
df_summary = pd.DataFrame(summary_rows)

detailed_csv = os.path.join(OUT_DIR, "contextbench_detailed_runs.csv")
summary_csv = os.path.join(OUT_DIR, "contextbench_summary_by_repo.csv")
lang_csv = os.path.join(OUT_DIR, "contextbench_summary_by_language.csv")

df_detailed.to_csv(detailed_csv, index=False)
df_summary.to_csv(summary_csv, index=False)

# Aggregate by Language & Arm
df_lang = (
    df_detailed.groupby(["Language", "Arm"])[["File_Recall@20", "Recall@20", "nDCG@10", "MRR", "Packed_Precision", "Packed_Recall"]]
    .mean()
    .reset_index()
)
df_lang.to_csv(lang_csv, index=False)

print("\n" + "=" * 80)
print("BENCHMARK 1 COMPLETE: CONTEXTBENCH WHOLE SUITE SUMMARY")
print("=" * 80)
print(df_summary[["Repository", "Language", "Arm", "File_Recall@20", "Recall@20", "MRR", "Avg_Packed_Tokens"]].to_string(index=False))

print("\n" + "=" * 80)
print("BREAKDOWN BY PROGRAMMING LANGUAGE")
print("=" * 80)
print(df_lang.to_string(index=False))

print(f"\nSaved detailed CSV: {detailed_csv}")
print(f"Saved summary CSV: {summary_csv}")
print(f"Saved language CSV: {lang_csv}")
