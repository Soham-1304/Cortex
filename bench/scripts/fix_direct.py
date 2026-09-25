#!/usr/bin/env python3
"""Direct fix-location eval: one LLM call per arm, no tools, no steps.

Usage:
  python3 fix_direct.py <tasks.jsonl> <workdir> <out.json>
    [--backend openrouter|nvidia] [--model m]

  WITHOUT: issue text -> name files+spans.
  WITH:    issue text + cortex recall+map output pre-attached -> same question.
  Scored: file-F1 + span overlap vs gold. 2 calls/issue. Backend-agnostic.

Stdlib only. Never prints keys.
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
BACKEND = {"name": "openrouter"}

URLS = {
    "openrouter": (os.environ.get("OPENROUTER_API_KEY", ""),
                   "https://openrouter.ai/api/v1/chat/completions"),
    "nvidia": (os.environ.get("NVIDIA_API_KEY", ""),
               "https://integrate.api.nvidia.com/v1/chat/completions"),
}


def call_llm(model, messages, retries=4):
    api_key, api_url = URLS[BACKEND["name"]]
    req = {"model": model, "messages": messages, "temperature": 0,
           "max_tokens": 1500}
    backoff = [15, 45, 120, 300]
    last = None
    for attempt in range(retries):
        try:
            r = urllib.request.Request(
                api_url, data=json.dumps(req).encode(),
                headers={"Authorization": f"Bearer {api_key}",
                         "Content-Type": "application/json",
                         "HTTP-Referer": "https://github.com/cortex",
                         "X-Title": "cortex-fix-direct"})
            with urllib.request.urlopen(r, timeout=180) as resp:
                body = json.load(resp)
            msg = body["choices"][0]["message"].get("content") or ""
            if not msg.strip():
                msg = body["choices"][0]["message"].get("reasoning_content") or ""
            return msg
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}"
            if e.code == 429 and attempt + 1 < retries:
                time.sleep(backoff[attempt] if attempt < len(backoff) else 300)
                continue
            raise RuntimeError(last)
        except Exception as e:
            raise RuntimeError(str(e)[:200])
    raise RuntimeError(last or "llm failed")


def cortex_context(workdir, issue):
    env = dict(os.environ)
    parts = []
    try:
        r = subprocess.run(
            [CORTEX, "recall", issue[:500], "--budget", "2000", "--json"],
            capture_output=True, text=True, env=env, cwd=workdir, timeout=180)
        p = json.loads(r.stdout[r.stdout.find("{"):])
        for m in p.get("memories", [])[:5]:
            parts.append(f"MEM: {m['content'][:300]}")
        for s in p.get("symbols", [])[:15]:
            parts.append(f"SYM: {s['path']}:{s['line_start']}-{s['line_end']} {s['name']}")
    except Exception as e:
        parts.append(f"(recall failed: {e})")
    try:
        r = subprocess.run(
            [CORTEX, "map", issue[:200]],
            capture_output=True, text=True, env=env, cwd=workdir, timeout=120)
        parts.append("MAP:\n" + (r.stdout or "")[:1500])
    except Exception as e:
        parts.append(f"(map failed: {e})")
    return "\n".join(parts)[:5000]


ASK = ("Name the files and line ranges that must change to fix this issue. "
       "Reply with JSON only: {{\"files\": [paths], \"spans\": [{{\"file\": p, "
       "\"start_line\": n, \"end_line\": m}}], \"rationale\": string}}. "
       "Name the smallest region that must change.")


def ask(model, prompt):
    text = call_llm(model, [{"role": "user", "content": prompt}])
    cands = []
    i = 0
    while True:
        s = text.find("{", i)
        if s < 0:
            break
        depth, instr, esc = 0, False, False
        for j in range(s, len(text)):
            c = text[j]
            if esc:
                esc = False
            elif c == "\\" and instr:
                esc = True
            elif c == '"':
                instr = not instr
            elif not instr and c == "{":
                depth += 1
            elif not instr and c == "}":
                depth -= 1
                if depth == 0:
                    try:
                        o = json.loads(text[s:j + 1])
                        if isinstance(o, dict) and any(
                                k in o for k in ("files", "done")):
                            cands.append(o)
                    except Exception:
                        pass
                    break
        i = s + 1
    if not cands:
        return None, text[:500]
    return cands[-1], None


def score(final, gold):
    gold_files = sorted(set(s["file"] for s in gold))
    if not final:
        return {"file_f1": 0.0, "span_hit": "0/%d" % len(gold),
                "files_p": [], "gold_files": gold_files}
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
    fix, workdir, outpath = sys.argv[1], sys.argv[2], sys.argv[3]
    args = sys.argv[4:]
    model = "stealth/space-bunny-alpha"
    for i, a in enumerate(args):
        if a == "--model":
            model = args[i + 1]
        if a == "--backend":
            BACKEND["name"] = args[i + 1]
    tasks = [json.loads(l) for l in open(fix) if l.strip()]
    log = {"provenance": {"generated_at_unix": int(time.time()), "model": model,
                          "backend": BACKEND["name"], "mode": "direct"},
           "issues": []}
    for t in tasks:
        gold = json.loads(t["gold_context"])
        row = {"id": t["instance_id"]}
        try:
            final, raw = ask(model, "ISSUE:\n" + t["problem_statement"][:3000] + "\n\n" + ASK)
            row["without"] = {"score": score(final, gold), "raw": raw}
        except Exception as e:
            row["without"] = {"score": score(None, gold), "error": str(e)[:200]}
        try:
            ctx = cortex_context(workdir, t["problem_statement"])
            final, raw = ask(model, "ISSUE:\n" + t["problem_statement"][:3000] +
                             "\n\nCORTEX CONTEXT:\n" + ctx + "\n\n" + ASK)
            row["with"] = {"score": score(final, gold), "raw": raw}
        except Exception as e:
            row["with"] = {"score": score(None, gold), "error": str(e)[:200]}
        log["issues"].append(row)
        print(f"{t['instance_id'][-8:]} without={row['without']['score']['file_f1']} "
              f"with={row['with']['score']['file_f1']}", flush=True)
        json.dump(log, open(outpath, "w"), indent=1)
    for arm in ("without", "with"):
        f1s = [i[arm]["score"]["file_f1"] for i in log["issues"]]
        spans = sum(int(i[arm]["score"]["span_hit"].split("/")[0]) for i in log["issues"])
        tot = sum(int(i[arm]["score"]["span_hit"].split("/")[1]) for i in log["issues"])
        print(f"{arm}: mean file-F1 {sum(f1s)/max(1,len(f1s)):.3f}, spans {spans}/{tot}")


if __name__ == "__main__":
    main()
