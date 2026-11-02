#!/usr/bin/env python3
"""Real-agent A/B: does Cortex save tokens, turns and API calls while keeping
(or improving) localization accuracy?  Paired per task, same model, same budget.

Arms (all get grep / read_file / list_dir / submit; same truncation limits):
  base      no cortex.
  tool      + `cortex_search` tool the agent may call (MCP-style usage).
  prefetch  cortex context for the issue injected into the first prompt
            (hook / harness-style usage), no extra tool.

Usage:
  python3 agent_eval.py --list-models groq|gemini
  python3 agent_eval.py <fixtures_dir> <repo_checkout> <cortex_workdir> <out.json>
        --provider groq|gemini --model <id> [--limit N] [--arms base,tool,prefetch]
        [--max-steps 20] [--seed 0]

  fixtures_dir   has tasks.jsonl: {instance_id, problem_statement, gold_context}
  cortex_workdir dir whose .cortex/memory.db was built with `cortex index`.

Keys come from GROQ_API_KEY / GEMINI_API_KEY (never printed or logged).
Stdlib only. Results are appended per task, so an interrupted run resumes.
"""
import argparse
import json
import os
import random
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from agent_logger import EpisodeLogger, generate_interactive_html

ROOT = Path(__file__).resolve().parents[2]
CORTEX = os.environ.get("CORTEX_BIN", str(ROOT / "target" / "release" / "cortex"))

# Automatically load .env if present
env_file = ROOT / ".env"
if env_file.is_file():
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip("'\""))

PROVIDERS = {
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai", "GEMINI_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "mistral": ("https://api.mistral.ai/v1", "MISTRAL_API_KEY"),
}
SKIP_DIRS = {".git", "__pycache__", ".venv", "node_modules", "dist", "build", ".cortex", ".tox"}
TOOL_OUT_CHARS = 2500   # identical truncation in every arm
READ_MAX_LINES = 80


class RollingTokenLimiter:
    """Enforces strict rolling Tokens Per Minute (TPM) quota to prevent HTTP 429 stalls."""
    def __init__(self, max_tpm=12000, window_s=60.0):
        self.max_tpm = max_tpm
        self.window_s = window_s
        self.history = []  # list of (timestamp, token_count)

    def _prune(self, now):
        cutoff = now - self.window_s
        self.history = [(t, tok) for t, tok in self.history if t > cutoff]

    def current_usage(self):
        now = time.time()
        self._prune(now)
        return sum(tok for _, tok in self.history)

    def wait_if_needed(self, estimated_tokens=1800, logger=None):
        now = time.time()
        self._prune(now)
        current = sum(tok for _, tok in self.history)
        while current + estimated_tokens > self.max_tpm and self.history:
            oldest_t, _ = self.history[0]
            sleep_s = max(1.0, (oldest_t + self.window_s) - time.time() + 0.5)
            if logger:
                logger.record_attempt(1, 429, wait_s=round(sleep_s, 1),
                                      error_msg=f"Proactive TPM limit pacing ({current:,}/{self.max_tpm:,} tokens)")
            print(f" [TPM Pacer: {current:,}/{self.max_tpm:,} tok, cooling {sleep_s:.1f}s]...", end="", flush=True)
            time.sleep(sleep_s)
            now = time.time()
            self._prune(now)
            current = sum(tok for _, tok in self.history)

    def record_usage(self, tokens):
        now = time.time()
        self.history.append((now, tokens))
        self._prune(now)


GLOBAL_TOKEN_LIMITER = RollingTokenLimiter(max_tpm=int(os.environ.get("GEMINI_MAX_TPM", "240000")), window_s=60.0)


def prune_messages(msgs, max_chars=8000):
    # 1. Truncate older tool outputs
    tool_indices = [i for i, m in enumerate(msgs) if m.get("role") == "tool"]
    to_prune = tool_indices[:-1] if len(tool_indices) > 1 else []
    for idx in to_prune:
        c = str(msgs[idx].get("content", ""))
        if len(c) > 250:
            msgs[idx]["content"] = c[:180] + "\n...[trimmed to conserve tokens]"

    # 2. Strip verbose <thought> tags from historical assistant messages
    for m in msgs:
        if m.get("role") == "assistant" and m.get("content"):
            c = str(m["content"])
            if "<thought>" in c and "</thought>" in c:
                m["content"] = re.sub(r'<thought>.*?</thought>', '', c, flags=re.DOTALL).strip()

    # 3. If total characters still exceed max_chars, prune earlier messages
    total = sum(len(str(m.get("content", ""))) for m in msgs)
    if total > max_chars and len(tool_indices) > 0:
        for idx in tool_indices:
            c = str(msgs[idx].get("content", ""))
            if len(c) > 120:
                msgs[idx]["content"] = c[:100] + "\n...[trimmed]"
    return msgs


# ---------------------------------------------------------------- LLM client
class Llm:
    def __init__(self, provider, model):
        self.base, key_env = PROVIDERS[provider]
        self.key = os.environ.get(key_env, "")
        if not self.key:
            sys.exit(f"{key_env} not set")
        self.model = model
        self.calls = 0          # every HTTP attempt (incl. retries) = billed/rate-limited calls
        self.retries = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def _post(self, path, body, timeout=120):
        req = urllib.request.Request(
            self.base + path, data=json.dumps(body).encode(),
            headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json", "User-Agent": "Mozilla/5.0"})
        return json.load(urllib.request.urlopen(req, timeout=timeout))

    def chat(self, messages, tools, tool_choice="auto", max_attempts=8, logger=None):
        body = {"model": self.model, "messages": messages, "tools": tools,
                "tool_choice": tool_choice, "temperature": 0}
        delay = 4.0
        for attempt in range(max_attempts):
            self.calls += 1
            # Proactively enforce rolling TPM limits before sending HTTP request
            GLOBAL_TOKEN_LIMITER.wait_if_needed(estimated_tokens=1800, logger=logger)
            t_call_0 = time.time()
            try:
                r = self._post("/chat/completions", body)
                latency_ms = int((time.time() - t_call_0) * 1000)
                u = r.get("usage") or {}
                p_tok = u.get("prompt_tokens", 0)
                c_tok = u.get("completion_tokens", 0)
                self.prompt_tokens += p_tok
                self.completion_tokens += c_tok
                GLOBAL_TOKEN_LIMITER.record_usage(p_tok + c_tok)
                if logger:
                    logger.record_attempt(attempt + 1, 200, wait_s=0, prompt_tokens=p_tok,
                                          completion_tokens=c_tok, latency_ms=latency_ms)
                return r["choices"][0]["message"]
            except urllib.error.HTTPError as e:
                detail = e.read().decode("utf-8", "replace")[:300]
                if e.code == 413 and attempt < max_attempts - 1:
                    self.retries += 1
                    # Aggressively trim older tool observations to fit TPM limit
                    for m in messages:
                        if m.get("role") == "tool" and len(m.get("content", "")) > 150:
                            m["content"] = m["content"][:100] + "\n...[trimmed]"
                    if logger:
                        logger.record_attempt(attempt + 1, 413, wait_s=2.0, error_msg=detail)
                    time.sleep(2)
                    continue
                if e.code in (429, 500, 502, 503, 504) and attempt < max_attempts - 1:
                    self.retries += 1
                    wait = delay
                    if e.headers.get("retry-after"):
                        try:
                            wait = float(e.headers.get("retry-after"))
                        except ValueError:
                            pass
                    reset_tok = e.headers.get("x-ratelimit-reset-tokens")
                    if reset_tok:
                        try:
                            if "m" in reset_tok:
                                m_part, s_part = reset_tok.split("m")
                                wait = max(wait, float(m_part) * 60 + float(s_part.rstrip("s")))
                            else:
                                wait = max(wait, float(reset_tok.rstrip("s")))
                        except Exception:
                            pass
                    actual_wait = min(wait + 1.0, 60)
                    if logger:
                        logger.record_attempt(attempt + 1, e.code, wait_s=actual_wait,
                                              error_msg=detail, headers=dict(e.headers))
                    time.sleep(actual_wait)
                    delay *= 2
                    continue
                raise RuntimeError(f"HTTP {e.code}: {detail}")
            except (urllib.error.URLError, TimeoutError) as ex:
                if attempt < max_attempts - 1:
                    self.retries += 1
                    if logger:
                        logger.record_attempt(attempt + 1, 599, wait_s=delay, error_msg=str(ex))
                    time.sleep(delay)
                    delay *= 2
                    continue
                raise
        raise RuntimeError("exhausted retries")


def list_models(provider):
    base, key_env = PROVIDERS[provider]
    key = os.environ.get(key_env, "")
    if not key:
        sys.exit(f"{key_env} not set")
    req = urllib.request.Request(base + "/models", headers={"Authorization": f"Bearer {key}", "User-Agent": "Mozilla/5.0"})
    data = json.load(urllib.request.urlopen(req, timeout=60))
    for m in sorted(x["id"] for x in data.get("data", [])):
        print(m)


# --------------------------------------------------------------------- tools
def _safe(repo, rel):
    p = (Path(repo) / rel.lstrip("/")).resolve()
    if not str(p).startswith(str(Path(repo).resolve())):
        return None
    return p


def _clip(s):
    return s if len(s) <= TOOL_OUT_CHARS else s[:TOOL_OUT_CHARS] + "\n...[truncated]"


def t_list_dir(repo, path="."):
    p = _safe(repo, path or ".")
    if not p or not p.is_dir():
        return f"not a directory: {path}"
    names = sorted(n.name + ("/" if n.is_dir() else "") for n in p.iterdir()
                   if n.name not in SKIP_DIRS)
    return _clip("\n".join(names[:300]))


def t_read_file(repo, path, start_line=1, end_line=None):
    p = _safe(repo, path)
    if not p or not p.is_file():
        return f"no such file: {path}"
    lines = p.read_text(errors="replace").splitlines()
    s = max(1, int(start_line or 1))
    e = min(len(lines), int(end_line) if end_line else s + READ_MAX_LINES - 1, s + READ_MAX_LINES - 1)
    body = "\n".join(f"{i}: {lines[i-1]}" for i in range(s, e + 1))
    return _clip(f"[{path} lines {s}-{e} of {len(lines)}]\n{body}")


def t_grep(repo, pattern, path="."):
    try:
        rx = re.compile(pattern)
    except re.error as ex:
        return f"bad regex: {ex}"
    base = _safe(repo, path or ".")
    if not base:
        return "bad path"
    out = []
    files = [base] if base.is_file() else (
        f for f in base.rglob("*") if f.is_file() and not (set(f.parts) & SKIP_DIRS))
    for f in files:
        if f.suffix not in (".py", ".rs", ".js", ".ts", ".go", ".java", ".c", ".h", ".rst", ".md", ".txt", ""):
            continue
        try:
            for i, line in enumerate(f.read_text(errors="replace").splitlines(), 1):
                if rx.search(line):
                    out.append(f"{f.relative_to(Path(repo).resolve())}:{i}: {line.strip()[:160]}")
                    if len(out) >= 40:
                        return _clip("\n".join(out) + "\n[40 match cap]")
        except OSError:
            pass
    return _clip("\n".join(out) or "no matches")


def cortex_context(workdir, query, budget=2000):
    r = subprocess.run([CORTEX, "recall", query, "--budget", str(budget), "--json"],
                       capture_output=True, text=True, cwd=workdir, timeout=180)
    out = r.stdout
    s_idx = out.find("{")
    e_idx = out.rfind("}")
    if s_idx == -1 or e_idx == -1:
        return "cortex: no result"
    try:
        p = json.loads(out[s_idx:e_idx+1])
    except Exception:
        return "cortex: no result"
    lines = []
    for s in p.get("symbols", []):
        lines.append(f"{s['path']}:{s.get('line_start')}-{s.get('line_end')}  {s.get('signature') or s.get('name')}")
    for m in p.get("memories", []):
        lines.append(f"note: {m.get('content','')[:200]}")
    return _clip("\n".join(lines) or "cortex: nothing relevant indexed")


def cortex_find(workdir, query, n=10):
    r = subprocess.run([CORTEX, "find", str(query), "-n", str(n), "--json"],
                       capture_output=True, text=True, cwd=workdir, timeout=60)
    out = r.stdout
    s_idx, e_idx = out.find("{"), out.rfind("}")
    if s_idx == -1 or e_idx == -1:
        return "cortex_find: no definitions or connections found"
    try:
        p = json.loads(out[s_idx:e_idx+1])
    except Exception:
        return "cortex_find: error parsing output"
    lines = []
    for d in p.get("defs", []):
        lines.append(f"def: {d.get('path')}:{d.get('line')} ({d.get('kind', 'symbol')})")
    for f in p.get("fts", [])[:10]:
        lines.append(f"match: {f.get('path')}:{f.get('line')} -> {f.get('context', '').strip()[:100]}")
    for c in p.get("connections", [])[:5]:
        lines.append(f"conn: {c.get('file')} imports {c.get('imports', [])[:4]}")
    return _clip("\n".join(lines) or "cortex_find: no matches")


def cortex_map(workdir, query):
    r = subprocess.run([CORTEX, "map", str(query), "--json"],
                       capture_output=True, text=True, cwd=workdir, timeout=60)
    out = r.stdout
    s_idx, e_idx = out.find("{"), out.rfind("}")
    if s_idx == -1 or e_idx == -1:
        return "cortex_map: no roadmap found"
    try:
        p = json.loads(out[s_idx:e_idx+1])
    except Exception:
        return "cortex_map: error parsing output"
    lines = []
    for ep in p.get("entry_points", [])[:5]:
        lines.append(f"entry_point: {ep.get('file')} imports {ep.get('imports', [])[:5]}")
    for ref in p.get("referenced_by", [])[:5]:
        lines.append(f"referenced_by: {ref.get('file')}")
    return _clip("\n".join(lines) or "cortex_map: empty roadmap")


def cortex_remember(workdir, content, kind="note"):
    r = subprocess.run([CORTEX, "remember", str(content), "--kind", str(kind)],
                       capture_output=True, text=True, cwd=workdir, timeout=30)
    out = r.stdout.strip()
    return f"cortex_remember: saved memory note id={out}" if out else "cortex_remember: saved"


def cortex_feedback(workdir, note_id, verdict="useful"):
    r = subprocess.run([CORTEX, "feedback", str(note_id), str(verdict)],
                       capture_output=True, text=True, cwd=workdir, timeout=30)
    return f"cortex_feedback: reinforced note {note_id} as {verdict}"


def tool_specs(with_cortex):
    def fn(name, desc, props, req):
        return {"type": "function", "function": {"name": name, "description": desc,
                "parameters": {"type": "object", "properties": props, "required": req}}}
    s = lambda d: {"type": "string", "description": d}
    i = lambda d: {"type": "integer", "description": d}
    specs = [
        fn("list_dir", "List a directory in the repository.", {"path": s("relative dir")}, []),
        fn("read_file", f"Read up to {READ_MAX_LINES} numbered lines of a file.",
           {"path": s("relative path"), "start_line": i("1-based"), "end_line": i("inclusive")}, ["path"]),
        fn("grep", "Regex search over source files (40 match cap).",
           {"pattern": s("python regex"), "path": s("optional dir or file")}, ["pattern"]),
    ]
    if with_cortex:
        specs.append(fn("cortex_search",
                        "Ask the project memory index for the symbols, source snippets, and past memories most relevant to a "
                        "description. Returns file:line ranges + signatures. Fast, local.",
                        {"query": s("natural-language description or identifiers"), "budget": i("token budget, default 1500")}, ["query"]))
        specs.append(fn("cortex_find",
                        "Exact AST identifier lookup: find exact definitions, usages, and call sites of a symbol name across files. Skips embeddings for high precision.",
                        {"query": s("symbol or function name to locate"), "n": i("max matches, default 10")}, ["query"]))
        specs.append(fn("cortex_map",
                        "File roadmap: discover architectural entry points, imports, and inbound references for files around keywords (under 1k tokens).",
                        {"query": s("file path or module keyword")}, ["query"]))
        specs.append(fn("cortex_remember",
                        "Store a key architectural note, bug hypothesis, or verified fix in the persistent project memory graph so subsequent tasks can recall it.",
                        {"content": s("the key insight or discovery to persist"), "kind": s("note, bug_hypothesis, or architecture")}, ["content"]))
        specs.append(fn("cortex_feedback",
                        "Rate a recalled note as useful or useless to strengthen or deprioritize it in the Hebbian memory graph.",
                        {"note_id": i("ID of the note"), "verdict": s("useful or useless")}, ["note_id", "verdict"]))
    specs.append(fn("submit", "Submit your final answer and complete the task: the code locations that must change to fix the issue.",
                    {"locations": {"type": "array", "items": {"type": "object", "properties": {
                        "file": s("repo-relative path"), "start_line": i("first line"),
                        "end_line": i("last line")}, "required": ["file", "start_line", "end_line"]}}},
                    ["locations"]))
    return specs


SYSTEM = ("You are an expert code-localization agent. Given a bug report, find the exact code that must "
          "change to fix it in the repository. Inspect the relevant files using `read_file`. "
          "Be decisive and fast: as soon as you find or confirm the location, call `submit` immediately to complete the task. "
          "Do not perform extra searches once the location is identified. You must call `submit` to finish.")


# ----------------------------------------------------------------- one episode
def run_episode(llm, repo, workdir, issue, arm, max_steps, logger=None):
    use_tool = arm == "tool"
    msgs = [{"role": "system", "content": SYSTEM}]
    prompt = f"Bug report:\n{issue[:2500]}\n"
    t0 = time.time()
    cortex_ms = 0
    if arm == "prefetch":
        c0 = time.time()
        ctx = cortex_context(workdir, issue)
        cortex_ms += int((time.time() - c0) * 1000)
        prompt += (
            f"\nCortex code index pre-fetched these candidate locations:\n{ctx}\n\n"
            "Examine the top candidate location using read_file. As soon as you confirm the code, "
            "immediately call `submit` with the exact line ranges. Aim to finish in 1-2 turns.\n"
        )
    elif use_tool:
        prompt += (
            "\nTip: You have access to Cortex codebase intelligence tools:\n"
            "- `cortex_search`: Semantic search over symbols and past memories (returns file:line ranges + signatures).\n"
            "- `cortex_find`: Exact AST definition and call-site lookup (pinpoints where a function/class is defined and used).\n"
            "- `cortex_map`: File roadmaps (entry points, imports, and inbound references).\n"
            "- `cortex_remember`: Store key architectural discoveries or bug hypotheses in persistent memory for future recall.\n"
            "Use these tools to locate relevant code rapidly, inspect with `read_file`, and call `submit`.\n"
        )
    msgs.append({"role": "user", "content": prompt})
    specs = tool_specs(use_tool)
    c_start = (llm.calls, llm.prompt_tokens, llm.completion_tokens, llm.retries)
    final, steps, tool_calls, err = None, 0, 0, None
    read_history = []
    try:
        for step in range(max_steps):
            if logger:
                logger.start_turn(step + 1)
            last = step == max_steps - 1
            if last:
                msgs.append({"role": "user", "content": "This is your final step. You must call submit now with your best estimate."})
            msgs = prune_messages(msgs)
            msg = llm.chat(msgs, specs, tool_choice="auto", logger=logger)
            steps += 1
            if logger and msg.get("content"):
                logger.record_model_response(msg.get("content"))
            # Clean assistant message for ongoing history: strip thought blocks to conserve tokens
            raw_c = msg.get("content") or ""
            clean_c = re.sub(r'<thought>.*?</thought>', '', raw_c, flags=re.DOTALL).strip()
            asst_msg = {"role": "assistant"}
            if clean_c:
                asst_msg["content"] = clean_c
            if msg.get("tool_calls"):
                asst_msg["tool_calls"] = msg["tool_calls"]
            msgs.append(asst_msg)
            tcs = msg.get("tool_calls") or []
            if not tcs:
                msgs.append({"role": "user", "content": "Use the tools; call `submit` when done."})
                if logger:
                    logger.end_turn()
                continue
            for tc in tcs:
                name = tc["function"]["name"]
                try:
                    args = json.loads(tc["function"].get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                tool_calls += 1
                t_tool_0 = time.time()
                if name == "submit":
                    locs = args.get("locations")
                    if locs is None and "file" in args:
                        locs = [args]
                    elif isinstance(locs, dict):
                        locs = [locs]
                    final = locs if isinstance(locs, list) else []
                    out = "submitted"
                elif name == "list_dir":
                    out = t_list_dir(repo, args.get("path", "."))
                elif name == "read_file":
                    p = args.get("path", "")
                    sl = int(args.get("start_line") or 1)
                    el = int(args.get("end_line") or (sl + 50))
                    read_history.append({"file": p, "start_line": sl, "end_line": el})
                    out = t_read_file(repo, p, sl, el)
                elif name == "grep":
                    out = t_grep(repo, args.get("pattern", ""), args.get("path", "."))
                elif name == "cortex_search" and use_tool:
                    c0 = time.time()
                    out = cortex_context(workdir, args.get("query", ""), budget=int(args.get("budget", 2000)))
                    cortex_ms += int((time.time() - c0) * 1000)
                elif name == "cortex_find" and use_tool:
                    c0 = time.time()
                    out = cortex_find(workdir, args.get("query", ""), n=int(args.get("n", 10)))
                    cortex_ms += int((time.time() - c0) * 1000)
                elif name == "cortex_map" and use_tool:
                    c0 = time.time()
                    out = cortex_map(workdir, args.get("query", ""))
                    cortex_ms += int((time.time() - c0) * 1000)
                elif name == "cortex_remember" and use_tool:
                    c0 = time.time()
                    out = cortex_remember(workdir, args.get("content", ""), kind=args.get("kind", "note"))
                    cortex_ms += int((time.time() - c0) * 1000)
                elif name == "cortex_feedback" and use_tool:
                    c0 = time.time()
                    out = cortex_feedback(workdir, args.get("note_id", 0), verdict=args.get("verdict", "useful"))
                    cortex_ms += int((time.time() - c0) * 1000)
                else:
                    out = f"unknown tool {name}"
                dur_ms = int((time.time() - t_tool_0) * 1000)
                if logger:
                    logger.record_tool_call(name, args, out, dur_ms)
                msgs.append({"role": "tool", "tool_call_id": tc.get("id", name), "content": out})
            if logger:
                logger.end_turn()
            if final is not None:
                break
        if final is None and not err:
            # One explicit submission turn if loop ended without submit
            if logger:
                logger.start_turn(steps + 1)
            msgs.append({"role": "user", "content": "Investigation finished. Call `submit` now with the code locations to fix."})
            try:
                msg = llm.chat(prune_messages(msgs), specs, tool_choice={"type": "function", "function": {"name": "submit"}}, logger=logger)
            except Exception:
                msg = llm.chat(prune_messages(msgs), specs, tool_choice="auto", logger=logger)
            steps += 1
            if logger and msg.get("content"):
                logger.record_model_response(msg.get("content"))
            tcs = msg.get("tool_calls") or []
            for tc in tcs:
                if tc["function"]["name"] == "submit":
                    try:
                        args = json.loads(tc["function"].get("arguments") or "{}")
                    except json.JSONDecodeError:
                        args = {}
                    locs = args.get("locations")
                    if locs is None and "file" in args:
                        locs = [args]
                    elif isinstance(locs, dict):
                        locs = [locs]
                    final = locs if isinstance(locs, list) else []
                    if logger:
                        logger.record_tool_call("submit", args, "submitted", 0)
            if final is None and msg.get("content"):
                matches = re.findall(r'([\w/.-]+\.(?:py|rs|js|ts|c|h))\b(?:[^\n\d]*lines?\s*(\d+)(?:\s*-\s*(\d+))?)?', msg.get("content", ""))
                if matches:
                    recovered = []
                    for f, s_l, e_l in matches:
                        if Path(repo, f).is_file():
                            s_num = int(s_l) if s_l else 1
                            e_num = int(e_l) if e_l else s_num + 20
                            recovered.append({"file": f, "start_line": s_num, "end_line": e_num})
                    if recovered:
                        final = recovered
            if final is None and read_history:
                final = read_history[-3:]
            if logger:
                logger.end_turn()
    except Exception as e:  # keep the run going; record the failure
        err = str(e)[:300]
    return {
        "final": final, "error": err, "turns": steps, "tool_calls": tool_calls,
        "api_calls": llm.calls - c_start[0], "retries": llm.retries - c_start[3],
        "prompt_tokens": llm.prompt_tokens - c_start[1],
        "completion_tokens": llm.completion_tokens - c_start[2],
        "wall_s": round(time.time() - t0, 2), "cortex_ms": cortex_ms,
    }


# --------------------------------------------------------------------- scoring
def _norm_path(p):
    s = str(p or "").replace("\\", "/").strip()
    if s.startswith("./"):
        s = s[2:]
    return s


def score(final, gold):
    gold_files = {_norm_path(g.get("file", "")) for g in gold if isinstance(g, dict)}
    out = {"file_f1": 0.0, "file_recall": 0.0, "span_recall": 0.0,
           "line_recall": 0.0, "line_precision": 0.0, "submitted": bool(final)}
    if not final:
        return out
    if isinstance(final, dict):
        final = [final]
    elif not isinstance(final, list):
        return out
    preds = [p for p in final if isinstance(p, dict) and p.get("file")]
    pf = {_norm_path(p["file"]) for p in preds}
    inter = len(pf & gold_files)
    prec, rec = inter / max(1, len(pf)), inter / max(1, len(gold_files))
    out["file_recall"] = round(rec, 3)
    out["file_f1"] = round(2 * prec * rec / max(1e-9, prec + rec), 3)

    def lines(spans):
        s = set()
        for sp in spans:
            a, b = int(sp.get("start_line") or 0), int(sp.get("end_line") or 0)
            f = _norm_path(sp.get("file", ""))
            if a and b >= a and f:
                s.update((f, n) for n in range(a, min(b, a + 5000) + 1))
        return s
    gl, pl = lines(gold), lines(preds)
    hit = 0                       # strict: a gold span counts only on real line overlap
    for g in gold:
        a, b = g.get("start_line", 0), g.get("end_line", 0)
        gf = _norm_path(g.get("file", ""))
        for p in preds:
            if _norm_path(p.get("file", "")) == gf and int(p.get("start_line") or 0) <= b and a <= int(p.get("end_line") or 0):
                hit += 1
                break
    out["span_recall"] = round(hit / max(1, len(gold)), 3)
    out["line_recall"] = round(len(gl & pl) / max(1, len(gl)), 3)
    out["line_precision"] = round(len(gl & pl) / max(1, len(pl)), 3)
    return out


def boot_ci(xs, n=4000, seed=7):
    if not xs:
        return (0.0, 0.0)
    rng = random.Random(seed)
    m = len(xs)
    means = sorted(sum(xs[rng.randrange(m)] for _ in range(m)) / m for _ in range(n))
    return (means[int(0.025 * n)], means[int(0.975 * n) - 1])


def summarize(rows, arms):
    metrics = ["prompt_tokens", "completion_tokens", "turns", "api_calls", "wall_s",
               "file_f1", "span_recall", "line_recall", "line_precision"]
    print(f"\n{len(rows)} tasks")
    print(f"{'arm':9s} " + " ".join(f"{m[:11]:>11s}" for m in metrics) + "  submitted")
    for a in arms:
        vals = {m: [r[a][m] for r in rows if a in r] for m in metrics}
        sub = sum(1 for r in rows if a in r and r[a]["submitted"])
        print(f"{a:9s} " + " ".join(f"{sum(v)/max(1,len(v)):11.2f}" for v in vals.values())
              + f"  {sub}/{len(rows)}")
    if "base" in arms:
        for a in arms:
            if a == "base":
                continue
            print(f"\npaired {a} - base (mean [95% CI]):")
            for m in metrics:
                d = [r[a][m] - r["base"][m] for r in rows if a in r and "base" in r]
                if d:
                    lo, hi = boot_ci(d)
                    print(f"  {m:18s} {sum(d)/len(d):+9.3f}  [{lo:+.3f}, {hi:+.3f}]  n={len(d)}")


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--list-models":
        return list_models(sys.argv[2])
    ap = argparse.ArgumentParser()
    ap.add_argument("fixtures"), ap.add_argument("repo"), ap.add_argument("workdir"), ap.add_argument("out")
    ap.add_argument("--provider", required=True, choices=PROVIDERS)
    ap.add_argument("--model", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--arms", default="base,tool,prefetch")
    ap.add_argument("--max-steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--logs-dir", default=str(ROOT / "bench" / "results" / "logs"))
    a = ap.parse_args()
    arms = a.arms.split(",")
    logs_dir = Path(a.logs_dir)
    episodes_dir = logs_dir / "episodes"
    markdown_dir = logs_dir / "markdown"
    episodes_dir.mkdir(parents=True, exist_ok=True)
    markdown_dir.mkdir(parents=True, exist_ok=True)

    fix_file = a.fixtures if a.fixtures.endswith(".jsonl") else f"{a.fixtures}/tasks.jsonl"
    tasks = [json.loads(l) for l in open(fix_file) if l.strip()]
    if a.seed:
        random.Random(a.seed).shuffle(tasks)
    if a.limit:
        tasks = tasks[:a.limit]
    done = {}
    if os.path.exists(a.out):
        for l in open(a.out):
            r = json.loads(l)
            done[r["id"]] = r
    llm = Llm(a.provider, a.model)
    rows = []
    
    local_repo_map = {
        "django/django": str(ROOT / "bench" / "fixtures" / "contextbench_django" / "repos" / "django__django"),
        "sympy/sympy": str(ROOT / "bench" / "fixtures" / "contextbench_sympy" / "repos" / "sympy__sympy"),
        "astropy/astropy": str(ROOT / "bench" / "fixtures" / "contextbench_astropy" / "repos" / "astropy__astropy"),
        "psf/requests": str(ROOT / "bench" / "fixtures" / "repos_cache" / "psf__requests"),
        "facebook/zstd": str(ROOT / "bench" / "fixtures" / "contextbench_zstd" / "repos" / "facebook__zstd"),
        "iamkun/dayjs": str(ROOT / "bench" / "fixtures" / "contextbench_dayjs" / "repos" / "iamkun__dayjs"),
        "clap-rs/clap": str(ROOT / "bench" / "fixtures" / "contextbench_clap" / "repos" / "clap-rs__clap"),
    }
    
    with open(a.out, "a") as f:
        for t in tasks:
            if t["instance_id"] in done:
                rows.append(done[t["instance_id"]])
                continue
            raw_gold = t.get("gold_context", "[]")
            gold = json.loads(raw_gold) if isinstance(raw_gold, str) else raw_gold
            row = {"id": t["instance_id"], "model": a.model, "gold_files": len({g.get("file","") for g in gold if isinstance(g, dict)})}
            order = arms[:]
            random.Random(__import__("zlib").crc32(t["instance_id"].encode())).shuffle(order)
            
            task_repo = local_repo_map.get(t.get("repo"), a.repo)
            task_workdir = task_repo
            
            for arm in order:
                logger = EpisodeLogger(
                    task_id=t["instance_id"],
                    repo=t.get("repo", Path(task_repo).name),
                    language=t.get("language", "Unknown"),
                    arm=arm,
                    model=a.model,
                    problem_statement=t["problem_statement"],
                    gold_context=gold,
                )
                r = run_episode(llm, task_repo, task_workdir, t["problem_statement"], arm, a.max_steps, logger=logger)
                sc = score(r["final"], gold)
                r.update(sc)
                row[arm] = r
                
                logger.finish(r["final"], sc, r["error"])
                clean_id = t["instance_id"].replace("/", "__")
                logger.save_json(episodes_dir / f"{clean_id}_{arm}.json")
                logger.save_markdown(markdown_dir / f"{clean_id}_{arm}.md")
                generate_interactive_html(episodes_dir, logs_dir / "index.html")

                print(f"{t['instance_id'][-8:]} {arm:8s} turns={r['turns']:2d} calls={r['api_calls']:2d} "
                      f"tok={r['prompt_tokens']+r['completion_tokens']:6d} f1={r['file_f1']:.2f} "
                      f"span={r['span_recall']:.2f} lines={r['line_recall']:.2f}"
                      f"{' ERR '+r['error'][:60] if r['error'] else ''}", flush=True)
            f.write(json.dumps(row) + "\n")
            f.flush()
            rows.append(row)
    summarize(rows, arms)


if __name__ == "__main__":
    main()
