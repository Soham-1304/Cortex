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
}
SKIP_DIRS = {".git", "__pycache__", ".venv", "node_modules", "dist", "build", ".cortex", ".tox"}
TOOL_OUT_CHARS = 3000   # identical truncation in every arm
READ_MAX_LINES = 100


def prune_messages(msgs, max_chars=12000):
    total = sum(len(str(m.get("content", ""))) for m in msgs)
    if total <= max_chars:
        return msgs
    tool_indices = [i for i, m in enumerate(msgs) if m.get("role") == "tool"]
    to_prune = tool_indices[:-2] if len(tool_indices) > 2 else tool_indices[:-1]
    for idx in to_prune:
        c = msgs[idx].get("content", "")
        if len(c) > 300:
            msgs[idx]["content"] = c[:200] + "\n...[trimmed to conserve context]"
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

    def chat(self, messages, tools, tool_choice="auto", max_attempts=8):
        body = {"model": self.model, "messages": messages, "tools": tools,
                "tool_choice": tool_choice, "temperature": 0}
        delay = 4.0
        for attempt in range(max_attempts):
            self.calls += 1
            try:
                r = self._post("/chat/completions", body)
                u = r.get("usage") or {}
                self.prompt_tokens += u.get("prompt_tokens", 0)
                self.completion_tokens += u.get("completion_tokens", 0)
                return r["choices"][0]["message"]
            except urllib.error.HTTPError as e:
                detail = e.read().decode("utf-8", "replace")[:300]
                if e.code == 413 and attempt < max_attempts - 1:
                    self.retries += 1
                    # Aggressively trim older tool observations to fit TPM limit
                    for m in messages:
                        if m.get("role") == "tool" and len(m.get("content", "")) > 150:
                            m["content"] = m["content"][:100] + "\n...[trimmed]"
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
                    time.sleep(min(wait + 1.0, 60))
                    delay *= 2
                    continue
                raise RuntimeError(f"HTTP {e.code}: {detail}")
            except (urllib.error.URLError, TimeoutError):
                if attempt < max_attempts - 1:
                    self.retries += 1
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
    try:
        p = json.loads(out[out.find("{"):])
    except Exception:
        return "cortex: no result"
    lines = []
    for s in p.get("symbols", []):
        lines.append(f"{s['path']}:{s.get('line_start')}-{s.get('line_end')}  {s.get('signature') or s.get('name')}")
    for m in p.get("memories", []):
        lines.append(f"note: {m.get('content','')[:200]}")
    return _clip("\n".join(lines) or "cortex: nothing relevant indexed")


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
                        "Ask the project memory index for the symbols most relevant to a "
                        "description. Returns file:line ranges + signatures. Fast, local.",
                        {"query": s("natural-language description or identifiers")}, ["query"]))
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
def run_episode(llm, repo, workdir, issue, arm, max_steps):
    use_tool = arm == "tool"
    msgs = [{"role": "system", "content": SYSTEM}]
    prompt = f"Bug report:\n{issue[:4000]}\n"
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
    msgs.append({"role": "user", "content": prompt})
    specs = tool_specs(use_tool)
    c_start = (llm.calls, llm.prompt_tokens, llm.completion_tokens, llm.retries)
    final, steps, tool_calls, err = None, 0, 0, None
    read_history = []
    try:
        for step in range(max_steps):
            last = step == max_steps - 1
            if last:
                msgs.append({"role": "user", "content": "This is your final step. You must call submit now with your best estimate."})
            msgs = prune_messages(msgs)
            msg = llm.chat(msgs, specs, tool_choice="auto")
            steps += 1
            msgs.append({k: v for k, v in msg.items() if v is not None})
            tcs = msg.get("tool_calls") or []
            if not tcs:
                msgs.append({"role": "user", "content": "Use the tools; call `submit` when done."})
                continue
            for tc in tcs:
                name = tc["function"]["name"]
                try:
                    args = json.loads(tc["function"].get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                tool_calls += 1
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
                    out = cortex_context(workdir, args.get("query", ""))
                    cortex_ms += int((time.time() - c0) * 1000)
                else:
                    out = f"unknown tool {name}"
                msgs.append({"role": "tool", "tool_call_id": tc.get("id", name), "content": out})
            if final is not None:
                break
        if final is None and not err:
            # One explicit submission turn if loop ended without submit
            msgs.append({"role": "user", "content": "Investigation finished. Call `submit` now with the code locations to fix."})
            msg = llm.chat(prune_messages(msgs), specs, tool_choice="auto")
            steps += 1
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
def score(final, gold):
    gold_files = {g["file"] for g in gold}
    out = {"file_f1": 0.0, "file_recall": 0.0, "span_recall": 0.0,
           "line_recall": 0.0, "line_precision": 0.0, "submitted": bool(final)}
    if not final:
        return out
    if isinstance(final, dict):
        final = [final]
    elif not isinstance(final, list):
        return out
    preds = [p for p in final if isinstance(p, dict) and p.get("file")]
    pf = {p["file"] for p in preds}
    inter = len(pf & gold_files)
    prec, rec = inter / max(1, len(pf)), inter / max(1, len(gold_files))
    out["file_recall"] = round(rec, 3)
    out["file_f1"] = round(2 * prec * rec / max(1e-9, prec + rec), 3)

    def lines(spans):
        s = set()
        for sp in spans:
            a, b = int(sp.get("start_line") or 0), int(sp.get("end_line") or 0)
            if a and b >= a:
                s.update((sp["file"], n) for n in range(a, min(b, a + 5000) + 1))
        return s
    gl, pl = lines(gold), lines(preds)
    hit = 0                       # strict: a gold span counts only on real line overlap
    for g in gold:
        a, b = g.get("start_line", 0), g.get("end_line", 0)
        for p in preds:
            if p["file"] == g["file"] and int(p.get("start_line") or 0) <= b and a <= int(p.get("end_line") or 0):
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
    a = ap.parse_args()
    arms = a.arms.split(",")
    tasks = [json.loads(l) for l in open(f"{a.fixtures}/tasks.jsonl") if l.strip()]
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
    with open(a.out, "a") as f:
        for t in tasks:
            if t["instance_id"] in done:
                rows.append(done[t["instance_id"]])
                continue
            gold = json.loads(t["gold_context"])
            row = {"id": t["instance_id"], "model": a.model, "gold_files": len({g["file"] for g in gold})}
            order = arms[:]
            random.Random(__import__("zlib").crc32(t["instance_id"].encode())).shuffle(order)
            for arm in order:
                r = run_episode(llm, a.repo, a.workdir, t["problem_statement"], arm, a.max_steps)
                r.update(score(r["final"], gold))
                row[arm] = r
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
