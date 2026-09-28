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
  spread (−0.048 file-level), heuristic + Qwen rerank, raises leg (error-name
  →raise-site votes fire but spread across dozens of sites; aggregates flat).

## v0.1.0

- Initial release: 3-tier memory engine, tree-sitter symbol index,
  RRF + knapsack context compiler, MCP stdio server, CLI.
- ρ-context estimator with LOQO holdout + bootstrap CI (|Δ| ≤ 0.02).
- Sim A/B: 70.5% default / 75–80% heavy token reduction at parity-or-better success.
- Real-LLM A/B (12 tasks, seed 42): bunny 68.8% [63.9–73.6], nemotron-free
  78.2%, both arms 100% success, turns ~2.2 → 1.0. (qwen3.8 run byte-identical
  to bunny — stealth-model alias suspected, counted once.)
- Fix-location eval (15 real astropy issues, paired WITH/WITHOUT cortex):
  four runs: (0.233→0.380), (0.491→0.367), (0.358→0.402), (0.244→0.258).
  Pooled means tied (0.344 vs 0.352). No accuracy effect at N=15 with this
  agent. WITH-arm range 0.035 vs WITHOUT 0.258 across all four: stability
  finding holds. Blocker is agent strength (8 steps, truncated reads),
  not retrieval — bench R@20 0.500 on the same index.
- Direct single-shot eval (no tool loop, bunny, 15 real issues): WITHOUT
  file-F1 0.000 (0/74 spans) vs WITH 0.133 (2 files, 2 spans). Bare model
  gets nothing; cortex context is the only signal. Small N, clean split.
- Memory-strengthening run (15 issues × 4 replay rounds, same DB): nav span
  0.667 / file 0.867 FLAT all rounds while recall_count 109→~270, entity
  weights →551. Reinforcement rewards exposure, not usefulness — no accuracy
  slope. Pitch unproven; loop needs a usefulness signal (agent feedback or
  success labels), not just recall counts.
- Dogfood agent loop (edit+test tools, red→green hidden tests, this repo):
  health-ping solved 1-1 (both arms, one edit each); secret redaction 0-0
  (neither arm attempted an edit in 14 reads). Instrument valid (tree
  restores clean); task set brackets agent ability without separating arms.
- Fixed two failing tests on v2 (both specced-but-unimplemented, failing on
  clean tree): AWS-key redaction at serve time (`AKIA`+16 scrubbed from
  recalled memory payloads, store keeps raw for FTS) and Ollama
  `health_check` live probe (`GET /api/tags`, 3s cap) so dead servers report
  missing instead of silently degrading to hash vectors.
- Refuted and removed: leading intent comments in indexed text (no-gap +
  cap-10 rule, Python `full_text` + non-Python signature twin). Django gate:
  file 0.822→0.8095, span 0.343→0.337, nDCG −0.026 — down on every metric.
  Comments share generic vocab / inflate costs; no signal. Fixture restored,
  code fully reverted.
- REMOVE bucket done: Cohere stub provider (menu offered it, always errored)
  deleted from registry/CLI/MCP; synthetic `cortex eval` CLI removed
  (vanity-task generator; `eval-llm` + bench crate stay the instruments);
  legacy absolute-count generic gate removed (ratio version won, zero
  callers). Net −157/+12 lines. Suites: 103 cortex / 54 bench green.
- IMPROVE bucket done: (1) tier prior flipped — proven project notes (1.25)
  now outrank hallucination-prone session notes (1.0), archived still 0.25;
  (2) reindex fidelity — new `symbols.embed_text` stores the exact indexed
  string (migration backfills NULL=legacy with signature fallback), so
  reindex no longer silently downgrades Python vectors to signature-only
  (byte-identical-blob test); (3) dead-provider errors point at recovery
  (`ollama serve` or `embeddings use` + `reindex`). Suites: 107/54 green.
- B-1 auto-index foundation: `indexed_files(path,mtime,hash,sym_count)` ledger
  written at index time (+walk-root meta, dead-row sweep), `status` reports
  clean/dirty/missing/untracked. Live-verified: edit+add+re-index cycle
  reports correctly. Triggers/watcher deferred to B-2.
- Refuted and removed: file-content leg (A-1: file BM25+cosine RRF added to
  member symbols). Django gate: file 0.822->0.736, span 0.343->0.351 (noise),
  nDCG -0.023. Mechanism matches the GAK-pack failure: aggregated file
  vocabulary lifts big wrong-file neighborhoods uniformly, burying the
  precise symbol legs. Pooling had nothing left to add (raw==pool).
- B-2 staleness triggers: recall-time mtime prescreen (`stale_quick`, capped,
  never fatal) attached to CLI + MCP payloads as `stale` with a stderr
  warning pointing at `cortex index .`; README documents the fresh-keeping
  flow incl. git-hook snippet. Fulfills the pre-existing MCP staleness
  contract in the tool description.
- Refuted and removed: intent/body cosine split (A-2: docstring+signature vs
  implementation as separate RRF arms). Django gate: file 0.822->0.793, span
  0.343->0.316, nDCG -0.052. Splitting halves the evidence per arm while
  double-counting noisy cosine against BM25. Track A bench work now 0-for-2.
- Weight thread closed per owner order (pool sweep kept as evidence, leg
  triplet killed mid-run and reverted): pool depth saturates at 10
  (15/20/30 == 0.8224 file) with a file<->span trade at pool=5 (file 0.72,
  span 0.38). Integrity warning: a default-weight rerun printed file 0.791
  vs 0.822 baseline with identical nDCG — suspected live query-embedding
  nondeterminism; fine-grained bench deltas need a repeatability probe
  before any weight conclusion.
- Ship bar re-anchored (Option A, owner call): file recall >0.90 all repos +
  agent task-success on real issues. Span >0.70 retired as a gate (0-for-2
  spikes + saturating weights say lexical+vector+graph won't 2x it); span
  stays a diagnostic. Remaining retrieval job: django -0.08, sympy -0.15.
- Usefulness signal (thread 3): `cortex feedback <id> useful|useless` + MCP
  `feedback` tool. Useful reuses recall reinforcement; useless decrements
  recall_count (floor 0, reversible — no sticky archive). Unknown ids error.
  Exposure is now tempered by outcomes. CLI+MCP smoke-verified.
- Silent 64-dim hash fallbacks removed everywhere (CLI remember/recall/
  index/reindex, MCP compile/write/reindex): bad config now fails loud with
  the `init --reconfigure` fix instead of poisoning rankings with mixed
  dims. Explicit `--embed hash` untouched. Correction: the "drift" alarm
  was a raw-vs-pooled metric mixup on my side — probe repeats agree
  exactly, harness is deterministic, no query-cache fix needed.
- Refuted and removed: hierarchical file->functions quotas (round-robin
  top-K files x top-M members over fused ranking). Django grid: best file
  +0.008 (noise), span down everywhere (10,2/10,4: -0.076). Round-robin
  swaps flat ranks ~11-20 (often gold same-file neighbors) for weak files'
  bests; quotas over wrong file order just re-exclude. Pool stays.
- Non-Python use-refs (plan point 3): generic import/call extraction for
  rs/go/js/ts/java into code_refs_use (Go blocks, require/from-forms,
  dotted full+short, def/comment/keyword filtering, decl lookbehind).
  find/map and the refs leg now see callers repo-wide. 115/54 green.
- Auto-index triggers: `cortex hook install` manages post-merge/post-checkout
  hooks (absolute binary, .cortex guard, hook.log failures, foreign-safe,
  worktree-aware, uninstall removes only ours). README points at the command.
- Harness gains Oriented@20 (span-hit AND file-hit in top-20): astropy
  0.867, django 0.741, sympy 0.682. Diagnostic only, never a gate.
