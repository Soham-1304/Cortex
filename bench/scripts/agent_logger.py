#!/usr/bin/env python3
"""Agent Evaluation Logging & Audit Visualizer for Cortex.

Provides structured, turn-by-turn transparent logging for live agent evaluations:
- Detailed JSON traces (`bench/results/logs/episodes/<task_id>_<arm>.json`)
- Clean, readable GitHub Markdown transcripts (`bench/results/logs/markdown/<task_id>_<arm>.md`)
- Interactive Single-Page HTML Dashboard (`bench/results/logs/index.html`)

Tracks:
- Every prompt, model thought, tool call, and terminal response
- Exact HTTP 429 rate limit events, backoff wait times, and reset headers
- Exact Cortex index queries (`cortex_search`) vs baseline commands (`grep`, `read_file`)
- Token usage per turn and cumulative cost
- Precision, Recall, F1 against gold ground truth
- Cognitive time vs rate-limit stall time
"""

import datetime
import html
import json
import os
from pathlib import Path
import time


class EpisodeLogger:
    def __init__(self, task_id, repo, language, arm, model, problem_statement, gold_context):
        self.task_id = task_id
        self.repo = repo
        self.language = language
        self.arm = arm  # 'base', 'tool' (cortex tool), 'prefetch' (cortex prefetch)
        self.model = model
        self.problem_statement = problem_statement
        self.gold_context = gold_context
        self.start_time = time.time()
        self.end_time = None
        self.wall_s = 0.0
        self.cognitive_s = 0.0
        self.rate_limit_wait_s = 0.0
        
        self.turns = []
        self.current_turn = None
        self.rate_limit_events = []
        self.cortex_calls = 0
        self.grep_calls = 0
        self.read_file_calls = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        
        self.final_submission = None
        self.score_metrics = {}
        self.error = None

    def start_turn(self, turn_idx):
        self.current_turn = {
            "turn_index": turn_idx,
            "start_time": time.time(),
            "elapsed_s": round(time.time() - self.start_time, 2),
            "attempts": [],
            "model_thought": "",
            "tool_calls": [],
            "prompt_tokens": 0,
            "completion_tokens": 0,
        }

    def record_attempt(self, attempt, status_code, wait_s=0.0, error_msg=None,
                       prompt_tokens=0, completion_tokens=0, latency_ms=0, headers=None):
        evt = {
            "attempt": attempt,
            "status_code": status_code,
            "latency_ms": latency_ms,
            "timestamp": datetime.datetime.now().isoformat(),
        }
        if status_code == 429:
            evt["is_rate_limit"] = True
            evt["wait_s"] = wait_s
            evt["headers"] = headers or {}
            evt["error"] = error_msg or "Rate limit exceeded"
            self.rate_limit_wait_s += wait_s
            self.rate_limit_events.append({
                "turn": self.current_turn["turn_index"] if self.current_turn else len(self.turns) + 1,
                "attempt": attempt,
                "wait_s": wait_s,
                "error": error_msg,
                "timestamp": evt["timestamp"]
            })
        elif status_code == 200:
            evt["is_rate_limit"] = False
            evt["prompt_tokens"] = prompt_tokens
            evt["completion_tokens"] = completion_tokens
            if self.current_turn:
                self.current_turn["prompt_tokens"] += prompt_tokens
                self.current_turn["completion_tokens"] += completion_tokens
            self.total_prompt_tokens += prompt_tokens
            self.total_completion_tokens += completion_tokens
        else:
            evt["error"] = error_msg
            
        if self.current_turn:
            self.current_turn["attempts"].append(evt)

    def record_model_response(self, content):
        if self.current_turn and content:
            self.current_turn["model_thought"] = content

    def record_tool_call(self, name, args, output, duration_ms=0):
        is_cortex = "cortex" in name
        if is_cortex:
            self.cortex_calls += 1
        elif name == "grep":
            self.grep_calls += 1
        elif name == "read_file":
            self.read_file_calls += 1

        tc = {
            "name": name,
            "args": args,
            "output": output,
            "duration_ms": duration_ms,
            "is_cortex": is_cortex,
            "is_submit": name == "submit"
        }
        if self.current_turn:
            self.current_turn["tool_calls"].append(tc)

    def end_turn(self):
        if self.current_turn:
            self.current_turn["duration_s"] = round(time.time() - self.current_turn["start_time"], 2)
            self.turns.append(self.current_turn)
            self.current_turn = None

    def finish(self, final_submission, score_metrics, error=None):
        self.end_time = time.time()
        self.wall_s = round(self.end_time - self.start_time, 2)
        self.cognitive_s = max(0.0, round(self.wall_s - self.rate_limit_wait_s, 2))
        self.final_submission = final_submission
        self.score_metrics = score_metrics or {}
        self.error = error

    def to_dict(self):
        return {
            "task_id": self.task_id,
            "repo": self.repo,
            "language": self.language,
            "arm": self.arm,
            "model": self.model,
            "problem_statement": self.problem_statement,
            "gold_context": self.gold_context,
            "wall_s": self.wall_s,
            "cognitive_s": self.cognitive_s,
            "rate_limit_wait_s": round(self.rate_limit_wait_s, 2),
            "turns_count": len(self.turns),
            "cortex_calls": self.cortex_calls,
            "grep_calls": self.grep_calls,
            "read_file_calls": self.read_file_calls,
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_prompt_tokens + self.total_completion_tokens,
            "score": self.score_metrics,
            "final_submission": self.final_submission,
            "error": self.error,
            "rate_limit_events": self.rate_limit_events,
            "turns": self.turns,
        }

    def save_json(self, filepath):
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    def to_markdown(self):
        f1 = self.score_metrics.get("file_f1", 0.0)
        rec = self.score_metrics.get("file_recall", 0.0)
        prec = self.score_metrics.get("file_precision", 0.0)
        submitted = self.score_metrics.get("submitted", False)
        
        status_badge = "🟩 SOLVED (F1=1.0)" if f1 == 1.0 else ("🟨 PARTIAL" if f1 > 0 else "🟥 FAILED")
        arm_title = "🧠 Cortex Arm" if self.arm in ("tool", "prefetch") else "🪵 Baseline Arm (No Cortex)"
        
        gold_files = [g.get("file", "") for g in self.gold_context if isinstance(g, dict)]
        pred_files = [p.get("file", "") for p in (self.final_submission or []) if isinstance(p, dict)]
        
        md = []
        md.append(f"# Task Audit Transcript: `{self.task_id}`")
        md.append(f"**Repository:** `{self.repo}` | **Language:** `{self.language}` | **Arm:** `{self.arm}` ({arm_title})")
        md.append(f"**Model:** `{self.model}` | **Status:** {status_badge} | **Submitted:** `{submitted}`\n")
        
        md.append("## Executive Scorecard")
        md.append("| Metric | Value | Reference / Notes |")
        md.append("|---|---|---|")
        md.append(f"| **File F1 Score** | `{f1:.3f}` | Exact match against gold modified files |")
        md.append(f"| **File Recall** | `{rec:.3f}` | Fraction of gold files localized |")
        md.append(f"| **Total Wall-Clock Time** | `{self.wall_s:.2f}s` | End-to-end elapsed time |")
        md.append(f"| **Cognitive Reasoning Time** | `{self.cognitive_s:.2f}s` | Wall-clock minus rate-limit stalls |")
        md.append(f"| **Rate-Limit Stalls** | `{self.rate_limit_wait_s:.2f}s` | Total sleep time from HTTP 429 backoffs |")
        md.append(f"| **Total Turns / Steps** | `{len(self.turns)}` | Reasoning loops executed |")
        md.append(f"| **Cortex Invocations** | `{self.cortex_calls}` | Semantic symbol & memory lookups |")
        md.append(f"| **Grep Invocations** | `{self.grep_calls}` | Brute-force filesystem searches |")
        md.append(f"| **Read File Invocations** | `{self.read_file_calls}` | File contents read |")
        md.append(f"| **Prompt Tokens** | `{self.total_prompt_tokens:,}` | Tokens sent to API |")
        md.append(f"| **Completion Tokens** | `{self.total_completion_tokens:,}` | Tokens generated by API |")
        md.append(f"| **Total Tokens** | `{self.total_prompt_tokens + self.total_completion_tokens:,}` | Cumulative token footprint |\n")
        
        if self.rate_limit_events:
            md.append("### ⚠️ Rate Limit Incidents (HTTP 429)")
            md.append("| Turn | Attempt | Wait Duration | Details | Timestamp |")
            md.append("|---|---|---|---|---|")
            for rle in self.rate_limit_events:
                md.append(f"| Turn {rle['turn']} | Attempt {rle['attempt']} | **{rle['wait_s']}s** | `{rle['error']}` | `{rle['timestamp']}` |")
            md.append("")

        md.append("## Ground Truth vs Model Prediction")
        md.append(f"- **Gold Files ({len(gold_files)}):** {', '.join(f'`{f}`' for f in gold_files) or 'None'}")
        md.append(f"- **Predicted Files ({len(pred_files)}):** {', '.join(f'`{f}`' for f in pred_files) or 'None'}")
        if self.final_submission:
            md.append("\n```json")
            md.append(json.dumps(self.final_submission, indent=2))
            md.append("```\n")

        md.append("## Turn-by-Turn Audit Trace")
        for turn in self.turns:
            t_idx = turn["turn_index"]
            dur = turn.get("duration_s", 0.0)
            toks = turn.get("prompt_tokens", 0) + turn.get("completion_tokens", 0)
            md.append(f"<details open>")
            md.append(f"<summary><strong>Turn {t_idx}</strong> ({dur:.1f}s | {toks:,} tokens)</summary>\n")
            
            # Show attempt badges
            for att in turn.get("attempts", []):
                if att.get("is_rate_limit"):
                    md.append(f"> 🔴 **[HTTP 429 RATE LIMIT]** Waited **{att.get('wait_s')}s** | Detail: `{att.get('error')}`\n")
                elif att.get("status_code") == 200:
                    md.append(f"> 🟢 **[HTTP 200 OK]** Latency: `{att.get('latency_ms')}ms` | Tokens: `{att.get('prompt_tokens')} prompt, {att.get('completion_tokens')} completion`\n")
            
            if turn.get("model_thought"):
                md.append(f"**Model Reasoning / Response:**\n\n{turn['model_thought']}\n")
                
            for tc in turn.get("tool_calls", []):
                name = tc["name"]
                args = tc["args"]
                out = tc["output"]
                dur_ms = tc.get("duration_ms", 0)
                
                if tc.get("is_cortex"):
                    badge = "🟣 `[CORTEX SEARCH]`"
                elif name == "submit":
                    badge = "🎯 `[SUBMIT]`"
                elif name == "grep":
                    badge = "🔎 `[GREP]`"
                elif name == "read_file":
                    badge = "📄 `[READ FILE]`"
                else:
                    badge = f"🔧 `[{name.upper()}]`"
                    
                md.append(f"{badge} **Call:** `{name}({json.dumps(args)})` *({dur_ms}ms)*\n")
                md.append("<pre style='background:#f6f8fa;padding:10px;border-radius:6px;max-height:350px;overflow-y:auto'>")
                md.append(html.escape(out))
                md.append("</pre>\n")
                
            md.append("</details>\n")
            
        return "\n".join(md)

    def save_markdown(self, filepath):
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            f.write(self.to_markdown())


def generate_interactive_html(json_logs_dir, out_html_path):
    """Compiles all JSON episode logs into a standalone, beautiful HTML dashboard."""
    logs = []
    p = Path(json_logs_dir)
    if p.is_dir():
        for f in sorted(p.glob("*.json")):
            try:
                with open(f, encoding="utf-8") as fh:
                    logs.append(json.load(fh))
            except Exception:
                pass

    # Build embedded JSON string for client-side interactivity
    logs_json = json.dumps(logs)
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Cortex vs Baseline: Agent Evaluation & Audit Dashboard</title>
<style>
  :root {{
    --bg-main: #0d1117;
    --bg-card: #161b22;
    --bg-card-hover: #1f242c;
    --border: #30363d;
    --text-primary: #c9d1d9;
    --text-muted: #8b949e;
    --cortex-purple: #8957e5;
    --cortex-purple-bg: rgba(137, 87, 229, 0.15);
    --green: #238636;
    --green-bg: rgba(35, 134, 54, 0.2);
    --red: #da3633;
    --red-bg: rgba(218, 54, 51, 0.2);
    --amber: #d29922;
    --amber-bg: rgba(210, 153, 34, 0.2);
    --blue: #58a6ff;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; }}
  body {{ background: var(--bg-main); color: var(--text-primary); padding: 24px; }}
  .header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 24px; padding-bottom: 16px; border-bottom: 1px solid var(--border); }}
  .header h1 {{ font-size: 24px; font-weight: 600; color: #f0f6fc; display: flex; align-items: center; gap: 10px; }}
  .header .badge {{ background: var(--cortex-purple); color: #fff; font-size: 12px; padding: 4px 10px; border-radius: 12px; font-weight: 500; }}
  
  /* Stats Cards Grid */
  .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-bottom: 24px; }}
  .stat-card {{ background: var(--bg-card); border: 1px solid var(--border); border-radius: 8px; padding: 16px; }}
  .stat-label {{ font-size: 12px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px; }}
  .stat-val {{ font-size: 26px; font-weight: 700; color: #f0f6fc; }}
  .stat-sub {{ font-size: 12px; color: var(--text-muted); margin-top: 4px; }}

  /* Filter Controls */
  .controls {{ display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 20px; align-items: center; }}
  .filter-group {{ display: flex; align-items: center; gap: 6px; background: var(--bg-card); border: 1px solid var(--border); border-radius: 6px; padding: 4px 10px; }}
  .filter-group label {{ font-size: 12px; color: var(--text-muted); }}
  .filter-group select, .filter-group input {{ background: transparent; border: none; color: #fff; font-size: 13px; outline: none; }}
  .btn {{ background: #21262d; border: 1px solid var(--border); color: #c9d1d9; padding: 6px 14px; border-radius: 6px; font-size: 13px; cursor: pointer; transition: 0.2s; }}
  .btn:hover {{ background: #30363d; color: #fff; }}
  .btn-primary {{ background: var(--cortex-purple); color: #fff; border: none; }}
  .btn-primary:hover {{ background: #9a67ea; }}

  /* Table */
  .table-container {{ background: var(--bg-card); border: 1px solid var(--border); border-radius: 8px; overflow: hidden; margin-bottom: 30px; }}
  table {{ width: 100%; border-collapse: collapse; text-align: left; font-size: 13px; }}
  th {{ background: #1c2128; padding: 12px 14px; color: var(--text-muted); font-weight: 600; border-bottom: 1px solid var(--border); }}
  td {{ padding: 12px 14px; border-bottom: 1px solid var(--border); vertical-align: middle; }}
  tr:hover {{ background: var(--bg-card-hover); }}
  .badge-tag {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; }}
  .badge-green {{ background: var(--green-bg); color: #3fb950; border: 1px solid rgba(63, 185, 80, 0.3); }}
  .badge-red {{ background: var(--red-bg); color: #f85149; border: 1px solid rgba(248, 81, 73, 0.3); }}
  .badge-amber {{ background: var(--amber-bg); color: #d29922; border: 1px solid rgba(210, 153, 34, 0.3); }}
  .badge-purple {{ background: var(--cortex-purple-bg); color: #bc8cff; border: 1px solid rgba(188, 140, 255, 0.3); }}
  .badge-gray {{ background: #21262d; color: #8b949e; border: 1px solid var(--border); }}

  /* Modal / Side-by-Side */
  .modal-overlay {{ display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.8); z-index: 1000; align-items: center; justify-content: center; }}
  .modal-box {{ background: var(--bg-card); border: 1px solid var(--border); border-radius: 10px; width: 90%; max-width: 1300px; height: 85vh; display: flex; flex-direction: column; overflow: hidden; }}
  .modal-header {{ display: flex; justify-content: space-between; align-items: center; padding: 16px 20px; border-bottom: 1px solid var(--border); background: #1c2128; }}
  .modal-body {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; padding: 20px; overflow-y: auto; flex: 1; }}
  .col-pane {{ background: #0d1117; border: 1px solid var(--border); border-radius: 8px; padding: 16px; overflow-y: auto; }}
  .turn-block {{ background: #161b22; border: 1px solid var(--border); border-radius: 6px; padding: 12px; margin-bottom: 12px; }}
  .tool-box {{ background: #0d1117; border: 1px solid #30363d; border-radius: 4px; padding: 8px 10px; font-family: monospace; font-size: 11px; white-space: pre-wrap; word-break: break-all; margin-top: 6px; max-height: 220px; overflow-y: auto; }}
  .close-btn {{ cursor: pointer; color: var(--text-muted); font-size: 20px; font-weight: bold; }}
  .close-btn:hover {{ color: #fff; }}
</style>
</head>
<body>

<div class="header">
  <div>
    <h1>🧠 Cortex Benchmark Audit & Live Agent Viewer</h1>
    <p style="font-size:13px;color:var(--text-muted);margin-top:4px;">Turn-by-turn transparent trace: tool calls, terminal outputs, HTTP 429 rate limit events, and token costs.</p>
  </div>
  <div>
    <span class="badge" style="background:#1f6feb;color:#fff;font-size:12px;padding:6px 14px;">Model: Google Gemini Flash</span>
    <span class="badge" style="background:#238636;color:#fff;font-size:12px;padding:6px 14px;margin-left:8px;">250K TPM Enterprise Quota | Live Multi-Turn Evaluation</span>
  </div>
</div>

<div class="stats-grid">
  <div class="stat-card">
    <div class="stat-label">Total Evaluated Episodes</div>
    <div class="stat-val" id="stTotal">0</div>
    <div class="stat-sub" id="stSubTasks">0 unique tasks</div>
  </div>
  <div class="stat-card">
    <div class="stat-label">Cortex Localization F1</div>
    <div class="stat-val" id="stCxF1" style="color:#3fb950;">0.00</div>
    <div class="stat-sub">Ground-truth code accuracy</div>
  </div>
  <div class="stat-card">
    <div class="stat-label">Baseline Localization F1</div>
    <div class="stat-val" id="stBaseF1" style="color:#f85149;">0.00</div>
    <div class="stat-sub">Without Cortex memory</div>
  </div>
  <div class="stat-card">
    <div class="stat-label">HTTP 429 Rate-Limit Stalls</div>
    <div class="stat-val" id="st429">0s</div>
    <div class="stat-sub" id="st429Sub">Cortex: 0s | Baseline: 0s</div>
  </div>
  <div class="stat-card">
    <div class="stat-label">Token Footprint Advantage</div>
    <div class="stat-val" id="stTokAdv" style="color:#bc8cff;">0%</div>
    <div class="stat-sub">Knapsack budget compression</div>
  </div>
</div>

<div class="controls">
  <div class="filter-group">
    <label>Language:</label>
    <select id="selLang" onchange="applyFilters()">
      <option value="ALL">All Languages</option>
      <option value="Python">Python</option>
      <option value="TypeScript/JS">TypeScript / JS</option>
      <option value="Go">Go</option>
      <option value="Rust">Rust</option>
      <option value="Java">Java</option>
      <option value="C/C++">C / C++</option>
    </select>
  </div>
  <div class="filter-group">
    <label>Arm:</label>
    <select id="selArm" onchange="applyFilters()">
      <option value="ALL">All Arms</option>
      <option value="tool">Cortex Tool Call</option>
      <option value="prefetch">Cortex Prefetch</option>
      <option value="base">Baseline (No Cortex)</option>
    </select>
  </div>
  <div class="filter-group">
    <label>Result:</label>
    <select id="selResult" onchange="applyFilters()">
      <option value="ALL">All Outcomes</option>
      <option value="SOLVED">Solved (F1 = 1.0)</option>
      <option value="429">Hit 429 Rate Limit</option>
      <option value="FAILED">Failed / Timeout</option>
    </select>
  </div>
  <div class="filter-group" style="flex:1;">
    <label>Search:</label>
    <input type="text" id="txtSearch" placeholder="Search task ID, repository, issue text..." oninput="applyFilters()" style="width:100%;">
  </div>
</div>

<div class="table-container">
  <table>
    <thead>
      <tr>
        <th>Task ID</th>
        <th>Repository</th>
        <th>Language</th>
        <th>Arm</th>
        <th>Outcome</th>
        <th>Turns</th>
        <th>Cognitive Time</th>
        <th>429 Stalls</th>
        <th>Tokens</th>
        <th>Cortex Calls</th>
        <th>Actions</th>
      </tr>
    </thead>
    <tbody id="tbRuns"></tbody>
  </table>
</div>

<!-- Modal for Side-by-Side Comparison -->
<div class="modal-overlay" id="compareModal" onclick="if(event.target===this)closeModal()">
  <div class="modal-box">
    <div class="modal-header">
      <div>
        <h3 id="modalTitle" style="color:#fff;">Head-to-Head Comparison</h3>
        <span id="modalSub" style="font-size:12px;color:var(--text-muted);"></span>
      </div>
      <span class="close-btn" onclick="closeModal()">&times;</span>
    </div>
    <div class="modal-body">
      <div class="col-pane" id="colCortex">
        <h4 style="color:#bc8cff;margin-bottom:12px;display:flex;align-items:center;gap:6px;">
          <span>🧠 Cortex Arm</span>
          <span class="badge-tag badge-purple" id="badgeCxScore">F1=1.00</span>
        </h4>
        <div id="contentCortex"></div>
      </div>
      <div class="col-pane" id="colBase">
        <h4 style="color:#8b949e;margin-bottom:12px;display:flex;align-items:center;gap:6px;">
          <span>🪵 Baseline Arm</span>
          <span class="badge-tag badge-gray" id="badgeBaseScore">F1=0.00</span>
        </h4>
        <div id="contentBase"></div>
      </div>
    </div>
  </div>
</div>

<script>
const RAW_LOGS = {logs_json};

function formatTokens(n) {{
  return n >= 1000 ? (n/1000).toFixed(1) + 'k' : n;
}}

function renderStats(logs) {{
  document.getElementById('stTotal').innerText = logs.length;
  const uniqueTasks = new Set(logs.map(l => l.task_id)).size;
  document.getElementById('stSubTasks').innerText = `${{uniqueTasks}} unique tasks`;
  
  const cxLogs = logs.filter(l => l.arm !== 'base');
  const baseLogs = logs.filter(l => l.arm === 'base');
  
  const cxAvgF1 = cxLogs.length ? (cxLogs.reduce((acc, l) => acc + (l.score?.file_f1 || 0), 0) / cxLogs.length).toFixed(2) : "0.00";
  const baseAvgF1 = baseLogs.length ? (baseLogs.reduce((acc, l) => acc + (l.score?.file_f1 || 0), 0) / baseLogs.length).toFixed(2) : "0.00";
  
  document.getElementById('stCxF1').innerText = cxAvgF1;
  document.getElementById('stBaseF1').innerText = baseAvgF1;
  
  const cx429Wait = cxLogs.reduce((acc, l) => acc + (l.rate_limit_wait_s || 0), 0);
  const base429Wait = baseLogs.reduce((acc, l) => acc + (l.rate_limit_wait_s || 0), 0);
  document.getElementById('st429').innerText = (cx429Wait + base429Wait).toFixed(1) + 's';
  document.getElementById('st429Sub').innerText = `Cortex: ${{cx429Wait.toFixed(1)}}s | Baseline: ${{base429Wait.toFixed(1)}}s`;
  
  const cxAvgTok = cxLogs.length ? (cxLogs.reduce((acc, l) => acc + (l.total_tokens || 0), 0) / cxLogs.length) : 0;
  const baseAvgTok = baseLogs.length ? (baseLogs.reduce((acc, l) => acc + (l.total_tokens || 0), 0) / baseLogs.length) : 0;
  if (baseAvgTok > 0 && cxAvgTok > 0) {{
    const adv = Math.round((1 - cxAvgTok / baseAvgTok) * 100);
    document.getElementById('stTokAdv').innerText = `+${{adv}}%`;
  }} else {{
    document.getElementById('stTokAdv').innerText = `--`;
  }}
}}

function renderTable(logs) {{
  const tbody = document.getElementById('tbRuns');
  tbody.innerHTML = '';
  logs.forEach(l => {{
    const tr = document.createElement('tr');
    const f1 = l.score?.file_f1 || 0;
    const isSolved = f1 === 1.0;
    const isPartial = f1 > 0 && f1 < 1.0;
    const has429 = (l.rate_limit_wait_s || 0) > 0 || (l.rate_limit_events?.length || 0) > 0;
    
    let outcomeBadge = isSolved 
      ? '<span class="badge-tag badge-green">F1=1.00 Solved</span>' 
      : (isPartial ? `<span class="badge-tag badge-amber">F1=${{f1.toFixed(2)}} Partial</span>` : '<span class="badge-tag badge-red">F1=0.00 Failed</span>');
    if (has429) {{
      outcomeBadge += ` <span class="badge-tag badge-red" title="Waited ${{l.rate_limit_wait_s}}s on 429">⚠️ 429 Stalled</span>`;
    }}
    
    let armBadge = l.arm === 'tool' 
      ? '<span class="badge-tag badge-purple">Cortex Tool</span>'
      : (l.arm === 'prefetch' ? '<span class="badge-tag badge-purple">Cortex Prefetch</span>' : '<span class="badge-tag badge-gray">Baseline</span>');

    tr.innerHTML = `
      <td><strong>${{l.task_id.substring(0, 16)}}...</strong></td>
      <td><code>${{l.repo}}</code></td>
      <td>${{l.language}}</td>
      <td>${{armBadge}}</td>
      <td>${{outcomeBadge}}</td>
      <td><strong>${{l.turns_count || (l.turns?.length || 0)}}</strong></td>
      <td>${{l.cognitive_s?.toFixed(2) || l.wall_s?.toFixed(2)}}s</td>
      <td>${{l.rate_limit_wait_s > 0 ? `<span style="color:#f85149;font-weight:600;">${{l.rate_limit_wait_s.toFixed(1)}}s</span>` : '0s'}}</td>
      <td>${{formatTokens(l.total_tokens || 0)}}</td>
      <td>${{l.cortex_calls || 0}}</td>
      <td>
        <button class="btn btn-primary" onclick="openCompare('${{l.task_id}}')">Audit Trace</button>
      </td>
    `;
    tbody.appendChild(tr);
  }});
}}

function applyFilters() {{
  const lang = document.getElementById('selLang').value;
  const arm = document.getElementById('selArm').value;
  const result = document.getElementById('selResult').value;
  const query = document.getElementById('txtSearch').value.toLowerCase();
  
  const filtered = RAW_LOGS.filter(l => {{
    if (lang !== 'ALL' && l.language !== lang) return false;
    if (arm !== 'ALL' && l.arm !== arm) return false;
    if (result === 'SOLVED' && (l.score?.file_f1 || 0) < 1.0) return false;
    if (result === '429' && ((l.rate_limit_wait_s || 0) === 0 && (l.rate_limit_events?.length || 0) === 0)) return false;
    if (result === 'FAILED' && (l.score?.file_f1 || 0) > 0) return false;
    if (query) {{
      const matchText = `${{l.task_id}} ${{l.repo}} ${{l.problem_statement}}`.toLowerCase();
      if (!matchText.includes(query)) return false;
    }}
    return true;
  }});
  
  renderTable(filtered);
}}

function renderTurnHtml(turn) {{
  let attHtml = '';
  (turn.attempts || []).forEach(a => {{
    if (a.is_rate_limit) {{
      attHtml += `<div style="background:var(--red-bg);border:1px solid rgba(248,81,73,0.4);border-radius:4px;padding:6px 10px;margin-bottom:6px;font-size:12px;color:#f85149;">
        ⚠️ <strong>HTTP 429 RATE LIMIT EXCEEDED</strong> &mdash; Waited <strong>${{a.wait_s}}s</strong> before retry.
      </div>`;
    }} else if (a.status_code === 200) {{
      attHtml += `<div style="font-size:11px;color:var(--text-muted);margin-bottom:6px;">
        🟢 HTTP 200 OK (${{a.latency_ms}}ms) | ${{a.prompt_tokens}} prompt + ${{a.completion_tokens}} compl tokens
      </div>`;
    }}
  }});

  let tcHtml = '';
  (turn.tool_calls || []).forEach(tc => {{
    const isCx = tc.is_cortex;
    const isSub = tc.is_submit;
    const badge = isCx ? '<span class="badge-tag badge-purple">Cortex Search</span>' :
                  (isSub ? '<span class="badge-tag badge-green">Submit</span>' : `<span class="badge-tag badge-gray">${{tc.name}}</span>`);
    tcHtml += `
      <div style="margin-top:8px;">
        <div style="font-size:12px;font-weight:600;display:flex;align-items:center;gap:6px;">
          ${{badge}} <code>${{tc.name}}(${{JSON.stringify(tc.args)}})</code>
        </div>
        <div class="tool-box">${{escapeHtml(tc.output)}}</div>
      </div>
    `;
  }});

  return `
    <div class="turn-block">
      <div style="display:flex;justify-content:space-between;margin-bottom:6px;font-size:12px;">
        <strong style="color:#f0f6fc;">Turn ${{turn.turn_index}}</strong>
        <span style="color:var(--text-muted);">${{turn.duration_s?.toFixed(1) || 0}}s</span>
      </div>
      ${{attHtml}}
      ${{turn.model_thought ? `<div style="font-size:12px;color:#c9d1d9;margin:6px 0;line-height:1.4;">${{escapeHtml(turn.model_thought)}}</div>` : ''}}
      ${{tcHtml}}
    </div>
  `;
}}

function escapeHtml(str) {{
  if (!str) return '';
  return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}}

function openCompare(taskId) {{
  const cx = RAW_LOGS.find(l => l.task_id === taskId && l.arm !== 'base');
  const base = RAW_LOGS.find(l => l.task_id === taskId && l.arm === 'base');
  
  document.getElementById('modalTitle').innerText = `Audit Inspection: ${{taskId}}`;
  document.getElementById('modalSub').innerText = `Repo: ${{cx?.repo || base?.repo || 'Unknown'}} | Language: ${{cx?.language || base?.language || 'Unknown'}}`;
  
  const cxContent = document.getElementById('contentCortex');
  const cxBadge = document.getElementById('badgeCxScore');
  if (cx) {{
    cxBadge.innerText = `F1=${{(cx.score?.file_f1 || 0).toFixed(2)}} | Turns: ${{cx.turns_count}} | Wall: ${{cx.wall_s}}s | 429: ${{cx.rate_limit_wait_s}}s`;
    cxContent.innerHTML = (cx.turns || []).map(renderTurnHtml).join('');
  }} else {{
    cxBadge.innerText = 'Not Evaluated';
    cxContent.innerHTML = '<div style="color:var(--text-muted);padding:20px;">No Cortex run for this task.</div>';
  }}
  
  const baseContent = document.getElementById('contentBase');
  const baseBadge = document.getElementById('badgeBaseScore');
  if (base) {{
    baseBadge.innerText = `F1=${{(base.score?.file_f1 || 0).toFixed(2)}} | Turns: ${{base.turns_count}} | Wall: ${{base.wall_s}}s | 429: ${{base.rate_limit_wait_s}}s`;
    baseContent.innerHTML = (base.turns || []).map(renderTurnHtml).join('');
  }} else {{
    baseBadge.innerText = 'Not Evaluated';
    baseContent.innerHTML = '<div style="color:var(--text-muted);padding:20px;">No Baseline run for this task.</div>';
  }}
  
  document.getElementById('compareModal').style.display = 'flex';
}}

function closeModal() {{
  document.getElementById('compareModal').style.display = 'none';
}}

// Initialize
renderStats(RAW_LOGS);
renderTable(RAW_LOGS);
</script>
</body>
</html>
"""
    p_out = Path(out_html_path)
    p_out.parent.mkdir(parents=True, exist_ok=True)
    with open(p_out, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"Generated interactive audit dashboard: {out_html_path} ({len(logs)} episodes)")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 2:
        generate_interactive_html(sys.argv[1], sys.argv[2])
    else:
        print("Usage: python3 agent_logger.py <json_logs_dir> <out_html_path>")
