# Cortex — Memory-First Context Engine

<p align="center">
  <strong>The agent brings the brain. Cortex brings the memory.</strong>
</p>

<p align="center">
  <a href="#quickstart">Quickstart</a> •
  <a href="#key-features">Key Features</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#mcp-integration">MCP Integration</a> •
  <a href="#cli-reference">CLI Reference</a> •
  <a href="#benchmarks">Benchmarks</a> •
  <a href="#license">License</a>
</p>

---

**Cortex** is a local, persistent memory engine for AI coding agents. Built as a single, lightweight Rust binary, Cortex equips agents with a continuous, structured understanding of your codebase across sessions without relying on cloud services, daemons, or telemetry.

One engine, two doors:
- **CLI door** for humans: inspect memory, index repos, record architectural decisions, and run evaluations.
- **MCP door** for AI agents: standard stdio JSON-RPC 2.0 interface compatible with Claude Code, Cursor, Windsurf, Cline, OpenCode, and any MCP-compatible client.

---

## Key Features

- **3-Tier Memory Model**:
  - **Tier 1: CORE** (`CORTEX.md` at repository root) — High-priority rules, architectural invariants, and project conventions that are always included in context.
  - **Tier 2: PROJECT** (`.cortex/memory.db`) — Local SQLite database storing project-wide decisions, file connections, tree-sitter symbols, code-reference knowledge graphs, and vector embeddings.
  - **Tier 3: SESSION** — Live, ephemeral knowledge graph built incrementally during an agent's task run, merged and decayed into project memory upon session digest.
- **Zero-Daemon Footprint**: 0 MB idle RAM. Cortex runs on demand via CLI commands or is spawned per-agent session via standard input/output.
- **Hybrid Fusion Retrieval (RRF)**: Combines dense vector similarity (cosine over 768d embeddings), lexical search (SQLite FTS5), and code-reference graphs using calibrated Reciprocal Rank Fusion (RRF).
- **Exact Token-Budget Knapsack**: Context compiler packs the most relevant symbols, memories, and core instructions into your model's exact token budget without overflow.
- **Local & Private by Default**:
  - Local quantized 768d model (`nomic-embed-text`) via fastembed.
  - Ollama integration for self-hosted embeddings.
  - Bring-Your-Own-Key (BYOK) providers for OpenAI (`text-embedding-3-large`) and Google (`text-embedding-004`).
  - Hash-based deterministic fallback for air-gapped CI environments.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           LOCAL MACHINE (100% Offline)                          │
│                                                                                 │
│  ┌─────────────────┐     ┌──────────────────────┐     ┌─────────────────────┐  │
│  │  Coding Agent   │────▶│    Cortex Engine     │     │  Project Repository │  │
│  │  (Claude Code / │     │    (Rust Binary)     │◀───▶│  (Source Files)     │  │
│  │   Cursor / MCP) │◀────│                      │     │                     │  │
│  └─────────────────┘     │  - Context Compiler  │     └─────────────────────┘  │
│                          │  - Symbol Indexer    │                                │
│                          │  - Hybrid Fusion/RRF │                                │
│                          │  - Knapsack Packer   │                                │
│                          └──────────┬───────────┘                                │
│                                     │                                            │
│                                     ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────────────────┐ │
│  │  Local Storage                                                              │ │
│  │                                                                              │ │
│  │  • Tier 1: CORTEX.md (Repo root, version-controlled)                        │ │
│  │  • Tier 2: .cortex/memory.db (SQLite + FTS5 + BLOB Vectors + Code Refs)     │ │
│  │  • Tier 3: Session Buffers (Ephemeral session graph, merged on digest)       │ │
│  │  • Global: ~/.cortex/models/ (Downloaded embedding models)                  │ │
│  └─────────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Quickstart

### 1. Build & Install

```bash
# Clone the repository
git clone https://github.com/Soham-1304/Cortex.git
cd Cortex

# Build and install locally via Cargo
cargo install --path cortex
```

### 2. Initialize in your Repository

Navigate to any software project and run the interactive setup wizard:

```bash
cortex init
```

The wizard will prompt you to select an embedding provider:
- **Local nomic-embed-text** (~110MB download, 768d, fully offline) *(Recommended)*
- **Ollama** (connect to an existing `nomic-embed-text` instance)
- **OpenAI / Google** (bring your own API keys)
- **Hash** (lightweight hash-based fallback for CI / testing)

You can also run non-interactively:
```bash
cortex init --embed local-nomic
# or
cortex init --embed ollama --ollama-url http://localhost:11434
```

This creates:
- `CORTEX.md` — Core instructions template for your repository.
- `.cortex/config.toml` — Local configuration and embedding provider preferences.
- `.cortex/memory.db` — SQLite database with schema, tables, and vector stores.

### 3. Index your Codebase

Extract AST symbols (via tree-sitter), calculate code-reference dependencies, and populate vector embeddings:

```bash
cortex index .
```

---

## MCP Integration

Cortex speaks Model Context Protocol (MCP) JSON-RPC over `stdio`. Any MCP-compliant client can use Cortex as its memory engine.

### Configuration

#### Claude Code
Add to your project's `.claude.json` or global configuration:
```json
{
  "mcpServers": {
    "cortex": {
      "command": "cortex",
      "args": ["serve", "--stdio"]
    }
  }
}
```

#### Cursor / Windsurf / Cline
In your MCP configuration file (`cursor_mcp.json` or equivalent):
```json
{
  "mcpServers": {
    "cortex": {
      "command": "cortex",
      "args": ["serve", "--stdio"]
    }
  }
}
```

### Exposed MCP Tools

| Tool | Description |
|---|---|
| `compile_context` | Compiles the optimal knapsack of Tier 1 (Core), Tier 2 (Memories & Symbols), and Tier 3 (Session) for a given query and token budget. |
| `memory_write` | Stores architectural decisions, entity notes, or resolved debugging insights into memory. |
| `memory_search` | Runs hybrid FTS5 and semantic search over stored notes. |
| `session_digest` | Summarizes session knowledge, merges episodic entities into project memory, and prunes transient nodes. |
| `core_read` | Retrieves the contents of `CORTEX.md`. |
| `core_write` | Updates `CORTEX.md` (permission-gated). |

---

## CLI Reference

### Daily Operations

```bash
# Record an architectural decision or fact
cortex remember "Use rusqlite bundled feature to ensure static link without system sqlite dependency" --entity sqlite --link sqlite:build

# Query memory with an explicit token budget
cortex recall "sqlite build issues" --budget 2000

# Search memory using lexical FTS5
cortex search "rusqlite"

# Record a session digest
cortex digest "Completed migration of tree-sitter symbols and verified offline embeddings"

# View or edit repository core rules
cortex core show
cortex core edit
```

### Embeddings & Index Management

```bash
# Inspect current provider status and vector dimensions
cortex embeddings list

# Switch embedding providers
cortex embeddings use ollama --ollama-url http://localhost:11434
cortex embeddings use openai --api-key-env OPENAI_API_KEY

# Re-index project following provider or dimension changes
cortex reindex
```

### Keeping the index fresh

`cortex status` reports index freshness (clean vs dirty/missing/untracked
files, tracked since first index). `recall` (CLI and MCP) carries a
stale-index notice when files changed under it — refresh with:

```bash
cortex index .
```

Hands-free freshness (opt-in, zero daemon): install git hooks that
re-index after pulls and branch switches (foreign hooks are never
overwritten without `--force`):

```bash
cortex hook install
# cortex hook uninstall  # removes only ours
```

### Agentic Evaluation

Retrieval is measured by the `bench` crate on real-issue benchmarks
(astropy/django/sympy). For a live LLM check with real tool calls:

```bash
export OPENROUTER_API_KEY=sk-or-...
cortex eval-llm --json --out eval_results.json
```

---

## Benchmarks & Performance

- **Memory & Latency**:
  - Idle RAM: **0 MB** (no resident background service).
  - Runtime RSS: **< 50 MB** during query compilation.
  - Compile Latency: Sub-millisecond context compilation using stored pre-computed embedding BLOBs.
- **Retrieval Precision**:
  - Tested on ContextBench benchmarks across production repositories (Astropy, Django, Sympy).
  - Hybrid RRF fusion delivers consistent token reduction and higher gold-symbol recall over pure lexical or naive cosine vector search.

---

## Repository Structure

```
.
├── Cargo.toml         # Rust workspace configuration
├── cortex/            # Cortex engine & CLI / MCP implementation
│   ├── Cargo.toml
│   └── src/
│       ├── main.rs    # Entrypoint (CLI router + MCP server)
│       ├── cli.rs     # CLI subcommands & interactive wizard
│       ├── mcp.rs     # JSON-RPC stdio server
│       ├── compile.rs # Knapsack context compiler & RRF fusion
│       ├── embed.rs   # Embedding providers (Nomic, Ollama, OpenAI, Google, Hash)
│       ├── graph.rs   # Spreading activation, Hebbian reinforcement & decay
│       ├── store.rs   # SQLite schema, FTS5 queries, and vector BLOBs
│       ├── session.rs # Session buffer lifecycle & digest logic
│       └── corefile.rs# CORTEX.md parser & manager
└── bench/             # Evaluation harness, ContextBench adapters, and calibration
    ├── Cargo.toml
    └── src/
```

---

## License

Licensed under the [Apache License, Version 2.0](LICENSE).
