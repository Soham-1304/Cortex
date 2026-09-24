# Changelog

## Unreleased (v2 branch)

- Bench: exact-identifier refs leg as third RRF leg (`--fuse rrf3`).
  Astropy span R@20 0.356 → 0.493, django 0.280 → 0.310, sympy 0.215 → 0.290.
- Bench: `file_recall20` column (files 0.7–0.8 vs spans 0.2–0.36).
- Bench + cortex: file-prioritized serving order (`--pool F` / `pool_by_file`,
  top-10 files first). Astropy +0.063 span, A/B success 46/48 → 48/48.
- Bench: `--embed ollama` with dead server exits non-zero (was silent
  hash-query corruption).
- All harnesses tag output (timestamp, version, provider/config).
- Refuted and removed: CRRF (≡RRF), GAK-pack (eff R@20 0.03), Task-4 graph
  spread (−0.048 file-level), heuristic + Qwen rerank.

## v0.1.0

- Initial release: 3-tier memory engine, tree-sitter symbol index,
  RRF + knapsack context compiler, MCP stdio server, CLI.
- ρ-context estimator with LOQO holdout + bootstrap CI (|Δ| ≤ 0.02).
- Sim A/B: 70.5% default / 75–80% heavy token reduction at parity-or-better success.
