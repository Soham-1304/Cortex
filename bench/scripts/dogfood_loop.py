#!/usr/bin/env python3
"""Dogfood agent loop: fix real Cortex tasks with edit+test tools.

Usage:
  OPENROUTER_API_KEY=... python3 dogfood_loop.py <repo> <workdir> <out.json>
    [--model m] [--tasks t1,t2]

Tasks (hidden tests already in repo, red):
  health-ping: embed.rs ollama_health_fails_on_unreachable_server
  redact:      compile.rs compile_redacts_aws_keys_in_memories

Tools both arms: list_files, read_file, edit_file (exact unique
old->new, repo-relative, allowlisted), run_tests (cargo test filter).
WITH arm adds: recall (cortex binary on <workdir> index).

Scoring per task/arm: hidden test passes AND full `cargo test -p cortex
--lib` green (no regressions). Same model, same step budget.
Stdlib only. Never prints the API key.
"""
import json
import os
import subprocess
import sys
import time
import urllib.request

CORTEX = os.environ.get("CORTEX_BIN",
                         str(__import__("pathlib").Path(__file__).resolve().parents[2]
                             / "target" / "release" / "cortex"))
API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
API_URL = "https://openrouter.ai/api/v1/chat/completions"

TASKS = {
    "health-ping": {
        "brief": ("OllamaProvider::health_check in cortex/src/embed.rs only checks "
                  "that the URL is non-empty, so a dead server passes health. Make it "
                  "probe the server (GET /api/tags) with a short timeout so "
                  "embed::tests::ollama_health_fails_on_unreachable_server passes. "
                  "Keep the empty-URL error. Do not break other tests."),
        "hidden": ["ollama_health_fails_on_unreachable_server"],
    },
    "redact": {
        "brief": ("compile() in cortex/src/compile.rs leaks secrets: a recalled note "
                  "containing an AWS key (AKIA[0-9A-Z]{16}) reaches the payload "
                  "verbatim. Redact AWS keys in recalled memory content before packing "
                  "so compile::tests::compile_redacts_aws_keys_in_memories passes. "
                  "Do not break other tests."),
        "hidden": ["compile_redacts_aws_keys_in_memories"],
    },
}

ALLOW_SUFFIX = (".rs", ".toml", ".md", ".py")


def extract_json(text):
    """First balanced {...} block (handles nested edit payloads and
    prose-wrapped answers)."""
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object")
    depth, instr, esc = 0, False, False
    for i in range(start, len(text)):
        c = text[i]
        if esc:
            esc = False
        elif c == "\\" and instr:
            esc = True
        elif c == '"' :
            instr = not instr
        elif not instr and c == "{":
            depth += 1
        elif not instr and c == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start:i + 1])
    raise ValueError("unbalanced JSON")


def call_llm(model, messages, retries=4):
    req = {"model": model, "messages": messages, "temperature": 0,
           "max_tokens": 2000}
    backoff = [15, 45, 120, 300]
    last = None
    for attempt in range(retries):
        try:
            r = urllib.request.Request(
                API_URL, data=json.dumps(req).encode(),
                headers={"Authorization": f"Bearer {API_KEY}",
                         "Content-Type": "application/json",
                         "HTTP-Referer": "https://github.com/cortex",
                         "X-Title": "cortex-dogfood"})
            with urllib.request.urlopen(r, timeout=180) as resp:
                body = json.load(resp)
            msg = body["choices"][0]["message"].get("content") or ""
            use = body.get("usage", {}) or {}
            return msg, use.get("prompt_tokens", 0) + use.get("completion_tokens", 0)
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}"
            if e.code == 429 and attempt + 1 < retries:
                time.sleep(backoff[attempt] if attempt < len(backoff) else 300)
                continue
            raise RuntimeError(last)
        except Exception as e:
            raise RuntimeError(str(e)[:200])
    raise RuntimeError(last or "llm failed")


def tool_list_files(repo, arg):
    out = []
    root = os.path.join(repo, arg) if arg else repo
    for base, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in (".git", "target", ".cortex")]
        rel = os.path.relpath(base, repo)
        if rel.count(os.sep) > 3:
            dirs[:] = []
            continue
        for f in sorted(files):
            if f.endswith(ALLOW_SUFFIX):
                out.append(os.path.join(rel, f))
        if len(out) > 200:
            break
    return "\n".join(out)[:2500]


def tool_read_file(repo, arg):
    parts = arg.split(":")
    full = os.path.normpath(os.path.join(repo, parts[0]))
    if not full.startswith(os.path.normpath(repo)):
        return "error: path escapes repo"
    if not full.endswith(ALLOW_SUFFIX):
        return "error: file type not editable"
    try:
        lines = open(full).read().splitlines()
    except Exception as e:
        return f"error: {e}"
    lo, hi, off = 1, len(lines), 1
    if len(parts) >= 2:
        try:
            lo = max(1, int(parts[1])); off = lo
            hi = int(parts[2]) if len(parts) >= 3 else lo + 79
        except ValueError:
            return "error: bad range, use path[:lo-hi]"
    numbered = [f"{off + i}:{l}" for i, l in enumerate(lines[off - 1:hi][:150])]
    return "\n".join(numbered)[:5000]


def tool_edit_file(repo, arg):
    try:
        obj = json.loads(arg) if isinstance(arg, str) else arg
        if isinstance(obj, str):
            obj = json.loads(obj)
        path, old, new = obj["path"], obj["old"], obj["new"]
    except Exception:
        return "error: arg must be JSON {path, old, new}"
    full = os.path.normpath(os.path.join(repo, path))
    if not full.startswith(os.path.normpath(repo)) or not full.endswith(ALLOW_SUFFIX):
        return "error: path not allowed"
    try:
        src = open(full).read()
    except Exception as e:
        return f"error: {e}"
    if src.count(old) != 1:
        return f"error: old block matches {src.count(old)} times (need exactly 1)"
    open(full, "w").write(src.replace(old, new, 1))
    return f"edited {path} ({len(old)} -> {len(new)} chars)"


def tool_run_tests(repo, arg):
    filt = arg.strip() or "--lib"
    r = subprocess.run(
        ["cargo", "test", "-p", "cortex", "--lib", filt] if filt != "FULL"
        else ["cargo", "test", "-p", "cortex", "--lib"],
        capture_output=True, text=True, cwd=repo, timeout=600)
    tail = (r.stdout + r.stderr)[-1500:]
    ok = "test result: ok" in tail
    return f"exit={r.returncode} pass={ok}\n{tail}"


def tool_recall(workdir, query):
    env = dict(os.environ)
    r = subprocess.run(
        [CORTEX, "recall", query[:500], "--budget", "2000", "--json"],
        capture_output=True, text=True, env=env, cwd=workdir, timeout=180)
    try:
        p = json.loads(r.stdout[r.stdout.find("{"):])
    except Exception:
        return f"recall failed: {r.stderr[-300:]}"
    lines = []
    for m in p.get("memories", [])[:5]:
        lines.append(f"MEM: {m['content'][:300]}")
    for s in p.get("symbols", [])[:15]:
        lines.append(f"SYM: {s['path']}:{s['line_start']}-{s['line_end']} {s['name']}")
    return "\n".join(lines)[:4000] or "(no hits)"


SYS = ("You fix a bug in the Cortex repo with tools; call exactly one per turn "
       "as JSON {{\"tool\": name, \"arg\": string}} or finish with JSON "
       "{{\"done\": true, \"summary\": string}} when the hidden tests should pass "
       "and the full suite stays green. Tools: list_files [subdir], read_file "
       "<path[:lo-hi]>, edit_file <JSON string {{\"path\": \"cortex/src/x.rs\", "
       "\"old\": \"exact lines to replace\", \"new\": \"replacement lines\"}}>, "
       "run_tests <test-filter|FULL>{recall}. "
       "For edit_file, old must match the file EXACTLY once (copy it from a "
       "read_file result). Verify with run_tests before finishing.")


def run_arm(model, task, repo, workdir, with_recall, max_steps=14):
    recall_doc = ", recall <query>" if with_recall else ""
    tools = {"list_files": lambda a: tool_list_files(repo, a),
             "read_file": lambda a: tool_read_file(repo, a),
             "edit_file": lambda a: tool_edit_file(repo, a),
             "run_tests": lambda a: tool_run_tests(repo, a)}
    if with_recall:
        tools["recall"] = lambda a: tool_recall(workdir, a)
    messages = [{"role": "system", "content": SYS.format(recall=recall_doc)},
                {"role": "user", "content": "TASK:\n" + TASKS[task]["brief"]}]
    steps = []
    for _ in range(max_steps):
        try:
            text, _ = call_llm(model, messages)
        except Exception as e:
            steps.append({"error": str(e)[:200]})
            break
        messages.append({"role": "assistant", "content": text})
        try:
            obj = extract_json(text)
        except Exception:
            steps.append({"raw": text[:300]})
            messages.append({"role": "user", "content": "Reply with one JSON tool call or done."})
            continue
        if obj.get("done"):
            steps.append({"done": obj.get("summary", "")[:300]})
            break
        name, arg = obj.get("tool", ""), str(obj.get("arg", ""))
        if name not in tools:
            steps.append({"bad_tool": name})
            messages.append({"role": "user", "content": "unknown tool. Use list_files, read_file, edit_file, run_tests" + recall_doc + "."})
            continue
        log_arg = arg if name != "edit_file" else arg[:2000]
        steps.append({"tool": name, "arg": log_arg})
        try:
            result = tools[name](arg)
        except Exception as e:
            result = f"error: {e}"
        messages.append({"role": "user", "content": "RESULT:\n" + result})
    return {"steps": steps}


def score_task(repo, task):
    """Hidden tests pass AND full lib suite green apart from the OTHER
    tasks' known-red hidden tests (each arm fixes one task; the other
    task's tests are red by design until fixed)."""
    hidden = TASKS[task]["hidden"]
    hidden_ok = True
    for h in hidden:
        r = subprocess.run(["cargo", "test", "-p", "cortex", "--lib", h],
                           capture_output=True, text=True, cwd=repo, timeout=300)
        if "test result: ok" not in (r.stdout + r.stderr):
            hidden_ok = False
    other_hidden = {t for task2 in TASKS for t in TASKS[task2]["hidden"]} - set(hidden)
    r2 = subprocess.run(["cargo", "test", "-p", "cortex", "--lib"],
                        capture_output=True, text=True, cwd=repo, timeout=600)
    out = r2.stdout + r2.stderr
    failed = set()
    if "failures:" in out:
        tail = out.split("failures:")[-1]
        for line in tail.splitlines():
            tok = line.strip().split()[0].rstrip(":,") if line.strip().split() else ""
            if "::" in tok:
                failed.add(tok)
    unexpected = {f for f in failed if not any(h in f for h in other_hidden)}
    suite_ok = not unexpected
    return {"hidden_pass": hidden_ok, "suite_green": suite_ok,
            "failed_tests": sorted(failed),
            "solved": hidden_ok and suite_ok}


def main():
    repo, workdir, outpath = sys.argv[1], sys.argv[2], sys.argv[3]
    args = sys.argv[4:]
    model, only, max_steps = "stealth/space-bunny-alpha", None, 14
    for i, a in enumerate(args):
        if a == "--model":
            model = args[i + 1]
        if a == "--tasks":
            only = args[i + 1].split(",")
        if a == "--steps":
            max_steps = int(args[i + 1])
    if not API_KEY:
        print("OPENROUTER_API_KEY not set", file=sys.stderr)
        sys.exit(2)
    log = {"provenance": {"generated_at_unix": int(time.time()), "model": model,
                          "max_steps": max_steps, "repo": repo},
           "runs": []}
    for task in TASKS:
        if only and task not in only:
            continue
        row = {"task": task}
        for arm, with_r in (("without", False), ("with", True)):
            # WITHOUT first on pristine tree; snapshot/restore between arms
            if arm == "with":
                subprocess.run(["git", "checkout", "--", "cortex/src"],
                               cwd=repo, capture_output=True)
            r = run_arm(model, task, repo, workdir, with_r, max_steps)
            r["score"] = score_task(repo, task)
            row[arm] = {"steps": r["steps"], "score": r["score"]}
            print(f"{task} {arm}: solved={r['score']['solved']} "
                  f"hidden={r['score']['hidden_pass']} suite={r['score']['suite_green']} "
                  f"steps={len(r['steps'])}", flush=True)
            subprocess.run(["git", "checkout", "--", "cortex/src"],
                           cwd=repo, capture_output=True)
        log["runs"].append(row)
        json.dump(log, open(outpath, "w"), indent=1)
    solved = {a: sum(1 for r in log["runs"] if r[a]["score"]["solved"]) for a in ("without", "with")}
    print(f"solved without={solved['without']} with={solved['with']} / {len(log['runs'])}")


if __name__ == "__main__":
    main()
