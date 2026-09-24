# Contributing to Cortex

## Setup

```bash
cargo build --workspace
cargo test --workspace
```

Ollama is optional (real 768d embeddings). Without it, `--embed hash`
everywhere still works — the bench smoke and all unit tests run offline.

## Rules

- **One task at a time.** Finish, measure, log before starting the next.
- **No per-repo tuning.** Same code, same constants on every repo. Gate
  comparisons at N≈100 minimum with CIs — N<100 is smoke, not a result.
- **Refuted levers get removed, not gated.** A lever with a negative
  CI-backed number is a result: delete the code, keep the record.
- **Claims discipline.** Quote only CI-backed numbers. Never claim novelty
  beyond the narrowed ρ-context framing; never make market comparisons.
- **No silent fallbacks.** Degraded modes must warn on stderr or fail fast.
  (See the 2026-09-24 mixed-embedding incident: hash queries against
  ollama docs with a log line still claiming ollama.)
- **Work on branches, merge via PR.** CI (`cargo test`, clippy `-D warnings`,
  bench smoke) must pass. `main` stays releasable.

## Tests

```bash
cargo test --workspace                    # 48 bench + 99 cortex
cargo clippy --workspace --all-targets -- -D warnings
cargo run -q -p bench --bin bench -- --fixtures fixtures/contextbench_sample --embed hash --budget 120,200
```

## Benchmarks (need ollama + fixtures, never in CI)

```bash
cargo run -q -p bench --bin bench -- --fixtures fixtures/ts_astropy --embed ollama --budget 4000 --fuse rrf3 --pool 10
cargo run -q -p cortex -- eval            # sim A/B, gate: paired reduction ≥60%
```
