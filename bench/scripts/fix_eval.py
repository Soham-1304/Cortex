#!/usr/bin/env python3
"""Fix-location eval: WITH vs WITHOUT cortex, paired per real issue.

Usage:
  OPENROUTER_API_KEY=... python3 fix_eval.py <tasks.jsonl> <repo> <workdir> <out.json>
    [--model m] [--limit n] [--steps 8]

  tasks.jsonl: {instance_id, problem_statement, gold_context}
  repo:        source checkout the agent explores (read-only)
  workdir:    cortex index dir (.cortex/memory.db) — WITH arm only
  out.json:   per-issue paired log + scores (gitignored)

Method: each arm gets list_files/read_file tools (the cmd-F baseline);
WITH additionally gets recall. LLM picks tools (temp 0, step cap), then
names files + line ranges to change. Scored automatically against gold:
file-F1 + span overlap. Same model, same budget, both arms.

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

SKIP_DIRS = {".git", "__pycache__", ".venv", "node_modules", "dist", "build"}


def call_llm(model, messages, retries=4):
    req = {"model": model, "messages": messages, "temperature": 0,
           "max_tokens": 1200}
    backoff = [15, 45, 120, 300]
    last = None
    for attempt in range(retries):
        try:
            r = urllib.request.Request(
                API_URL, data=json.dumps(req).encode(),
                headers={"Authorization": f"Bearer {API_KEY}",
                         "Content-Type": "application/json",
                         "HTTP-Referer": "https://github.com/cortex",
                         "X-Title": "cortex-fix-eval"})
            with urllib.request.urlopen(r, timeout=120) as resp:
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
            last = str(e)[:200]
            raise RuntimeError(last)
    raise RuntimeError(last or "llm failed")


def tool_list_files(repo, arg):
    root = os.path.join(repo, arg) if arg else repo
    out = []
    try:
        for base, dirs, files in os.walk(root):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
            rel = os.path.relpath(base, repo)
            depth = 0 if rel == "." else rel.count(os.sep) + 1
            if depth > 3:
                dirs[:] = []
                continue
            for d in sorted(dirs):
                out.append(os.path.join(rel, d) + "/")
            for f in sorted(files):
                if f.endswith((".py", ".rs", ".js", ".ts", ".go", ".java")):
                    out.append(os.path.join(rel, f))
            if len(out) > 400:
                out.append("... (truncated)")
                break
    except Exception as e:
        return f"error: {e}"
    return "\n".join(out)[:3000]


def tool_read_file(repo, arg):
    parts = arg.split(":")
    path = parts[0]
    full = os.path.normpath(os.path.join(repo, path))
    if not full.startswith(os.path.normpath(repo)):
        return "error: path escapes repo"
    try:
        lines = open(full).read().splitlines()
    except Exception as e:
        return f"error: {e}"
    if len(parts) >= 3:
        try:
            lo, hi = max(1, int(parts[1])), int(parts[2])
            lines = lines[lo - 1:hi]
            off = lo
        except ValueError:
            return "error: bad range, use path:lo-hi"
    else:
        off = 1
    if len(parts) == 2:
        try:
            lo = max(1, int(parts[1]))
            lines, off = lines[lo - 1:lo + 79], lo
        except ValueError:
            return "error: bad range"
    numbered = [f"{off + i}:{l}" for i, l in enumerate(lines[:120])]
    return "\n".join(numbered)[:4000]


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


SYS = ("You are fixing a bug in a large repo. You have tools; call exactly one "
       "per turn as JSON {{\"tool\": name, \"arg\": string}} or give the final "
       "answer as JSON {{\"files\": [paths], \"spans\": [{{\"file\": p, "
       "\"start_line\": n, \"end_line\": m}}], \"rationale\": string}}. "
       "Tools: list_files [subdir], read_file <path[:lo-hi]>{recall}. "
       "Be precise: name the smallest region that must change.")


def run_arm(model, issue, repo, workdir, with_recall, max_steps):
    recall_doc = ", recall <query>" if with_recall else ""
    tools = {"list_files": lambda a: tool_list_files(repo, a),
             "read_file": lambda a: tool_read_file(repo, a)}
    if with_recall:
        tools["recall"] = lambda a: tool_recall(workdir, a)
    messages = [{"role": "system", "content": SYS.format(recall=recall_doc)},
                {"role": "user", "content": "ISSUE:\n" + issue[:3000]}]
    steps, tokens = [], 0
    final = None
    for _ in range(max_steps):
        try:
            text, tk = call_llm(model, messages)
        except Exception as e:
            steps.append({"error": f"llm: {e}"})
            break
        tokens += tk
        messages.append({"role": "assistant", "content": text})
        try:
            obj = json.loads(text[text.find("{"):text.rfind("}") + 1])
        except Exception:
            steps.append({"raw": text[:500]})
            messages.append({"role": "user", "content": "Reply with one JSON tool call or the final answer."})
            continue
        if "files" in obj:
            final = obj
            steps.append({"final": True})
            break
        name, arg = obj.get("tool", ""), str(obj.get("arg", ""))
        if name not in tools:
            messages.append({"role": "user", "content": f"unknown tool {name!r}. Use list_files, read_file{recall_doc}."})
            steps.append({"bad_tool": name})
            continue
        try:
            result = tools[name](arg)
        except Exception as e:
            result = f"error: {e}"
        steps.append({"tool": name, "arg": arg[:120]})
        messages.append({"role": "user", "content": "RESULT:\n" + result})
    if final is None:
        # Tool budget spent exploring: force the verdict, no more tools.
        messages.append({"role": "user", "content": "No more tool calls. Output the final answer JSON now: {\"files\": [...], \"spans\": [...], \"rationale\": string}."})
        try:
            text, tk = call_llm(model, messages)
            tokens += tk
            final = json.loads(text[text.find("{"):text.rfind("}") + 1])
            steps.append({"final": "forced"})
        except Exception as e:
            steps.append({"final_failed": str(e)[:200]})
    return {"final": final, "steps": steps, "tokens_api": tokens}


def score(final, gold):
    gold_files = sorted(set(s["file"] for s in gold))
    if not final:
        return {"file_f1": 0.0, "span_hit": 0.0, "files_p": [], "gold_files": gold_files}
    pred_files = sorted(set(final.get("files", [])))
    gf, pf = set(gold_files), set(pred_files)
    inter = len(gf & pf)
    prec = inter / max(1, len(pf))
    rec = inter / max(1, len(gf))
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    span_hit = 0
    for sp in gold:
        for q in final.get("spans", []):
            if q.get("file") != sp["file"]:
                continue
            a0, a1 = q.get("start_line", 0), q.get("end_line", 0)
            b0, b1 = sp.get("start_line", 0), sp.get("end_line", 0)
            if (a0 == 0 and a1 == 0) or (a0 <= b1 and b0 <= a1):
                span_hit += 1
                break
    return {"file_f1": round(f1, 3), "span_hit": f"{span_hit}/{len(gold)}",
            "files_p": pred_files, "gold_files": gold_files}


def main():
    fix, repo, workdir, outpath = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    args = sys.argv[5:]
    model = "stealth/space-bunny-alpha"
    limit, max_steps = None, 8
    for i, a in enumerate(args):
        if a == "--model":
            model = args[i + 1]
        if a == "--limit":
            limit = int(args[i + 1])
        if a == "--steps":
            max_steps = int(args[i + 1])
    if not API_KEY:
        print("OPENROUTER_API_KEY not set", file=sys.stderr)
        sys.exit(2)
    tasks = [json.loads(l) for l in open(fix) if l.strip()]
    if limit:
        tasks = tasks[:limit]
    log = {"provenance": {"generated_at_unix": int(time.time()), "model": model,
                          "max_steps": max_steps, "repo": repo},
           "issues": []}
    for t in tasks:
        gold = json.loads(t["gold_context"])
        row = {"id": t["instance_id"], "gold_n": len(gold)}
        for arm, with_r in (("without", False), ("with", True)):
            r = run_arm(model, t["problem_statement"], repo, workdir, with_r, max_steps)
            r["score"] = score(r["final"], gold)
            row[arm] = r
            print(f"{t['instance_id'][-8:]} {arm}: f1={r['score']['file_f1']} "
                  f"span={r['score']['span_hit']} steps={len(r['steps'])}", flush=True)
        log["issues"].append(row)
        json.dump(log, open(outpath, "w"), indent=1)
    # paired summary
    for arm in ("without", "with"):
        f1s = [i[arm]["score"]["file_f1"] for i in log["issues"]]
        print(f"{arm}: mean file-F1 {sum(f1s)/max(1,len(f1s)):.3f} over {len(f1s)} issues")


if __name__ == "__main__":
    main()
