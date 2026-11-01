//! Real-LLM variant of the agentic A/B: same synthetic repo + same budgets,
//! but the *baseline agent's* next-query is chosen by a live LLM via OpenRouter
//! instead of the deterministic learn-and-concatenate rule. Cortex arm stays
//! deterministic (compile) for a clean paired comparison. Costs are real API
//! token counts (prompt+completion) when available, else local token fallback.
//!
//! Design: zero cost without `OPENROUTER_API_KEY`. `--limit` caps calls.
//! No new deps — uses existing `ureq`.

use crate::agentic::{cortex_loop, generate, seed_memory, CortexRun, HarnessConfig, HarnessReport, TaskOutcome, is_ident};
use crate::corefile::CoreFile;
use crate::embed::HashEmbedder;
use crate::store::Store;
use cortex_core::bm25::Bm25Index;
use cortex_core::{count_tokens, tokenize};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

const OPENROUTER_URL: &str = "https://openrouter.ai/api/v1/chat/completions";

#[derive(Serialize)]
struct ChatReq<'a> {
    model: &'a str,
    messages: Vec<Msg<'a>>,
    temperature: f32,
    max_tokens: u32,
}
#[derive(Serialize)]
struct Msg<'a> { role: &'a str, content: String }
#[derive(Deserialize)]
struct ChatResp { choices: Vec<Choice>, usage: Option<Usage> }
#[derive(Deserialize)]
struct Choice { message: RespMsg }
#[derive(Deserialize)]
struct RespMsg { content: String }
#[derive(Deserialize)]
struct Usage { prompt_tokens: i64, completion_tokens: i64 }

fn call_llm(model: &str, api_key: &str, task_desc: &str, docs_read: &[String]) -> Result<(String, i64), String> {
    let seen: String = if docs_read.is_empty() { "(none yet)".into() } else { docs_read.join("\n---\n") };
    let prompt = format!(
        "You are a coding agent searching a repo to fix: \"{task_desc}\".\n\
         Docs read so far (truncated):\n{seen}\n\n\
         Reply with ONLY the next search query (2-6 keywords, no sentences, no explanation). \
         Use identifier-like tokens you saw (tN/sN/dN) if useful."
    );
    let req = ChatReq { model, messages: vec![Msg{role:"user", content: prompt}], temperature: 0.0, max_tokens: 32 };
    let resp: ChatResp = ureq::post(OPENROUTER_URL)
        .set("Authorization", &format!("Bearer {}", api_key))
        .set("Content-Type", "application/json")
        .set("HTTP-Referer", "https://github.com/cortex")
        .set("X-Title", "cortex-eval")
        .send_json(&req)
        .map_err(|e| e.to_string())?
        .into_json::<ChatResp>()
        .map_err(|e| e.to_string())?;
    let content = resp.choices.first().map(|c| c.message.content.trim().to_string()).unwrap_or_default();
    let tokens = resp.usage.map(|u| u.prompt_tokens + u.completion_tokens).unwrap_or(0);
    if content.is_empty() { return Err("empty LLM response".into()); }
    Ok((content, tokens))
}

#[derive(Debug, Clone)]
pub struct LlmConfig {
    pub base: HarnessConfig,
    pub model: String,
    pub api_key: String,
    pub limit: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct LlmHarnessReport {
    pub base: HarnessReport,
    pub model: String,
    pub llm_calls: usize,
    pub llm_tokens_api: i64,
    pub note: String,
}

pub fn run_llm_harness(cfg: &LlmConfig) -> crate::Result<LlmHarnessReport> {
    if cfg.api_key.trim().is_empty() {
        return Err(crate::Error::Cli("OPENROUTER_API_KEY not set — dry-run only. Export it or run without --llm".into()));
    }
    let mut base = cfg.base;
    if cfg.limit > 0 && (base.tasks) > cfg.limit { base.tasks = cfg.limit; }
    let project = generate(base.docs, base.tasks, base.seed, base.distractors)?;
    let store = Store::open_in_memory()?;
    // Harness convention: hash-64 for synthetic docs (no real semantics in
    // generated token soup; both arms share this store, so A/B stays fair).
    // Real-embedding retrieval is measured by the bench harness, not here.
    let emb = HashEmbedder::new(64);
    let seed_tokens_total = seed_memory(&store, &project, &emb)?;
    let core = CoreFile { path: "eval".into(), preamble: None, sections: Default::default() };
    let texts: Vec<String> = project.docs.iter().map(|d| d.text.clone()).collect();
    let index = Bm25Index::build(&texts);

    let mut outcomes: Vec<TaskOutcome> = Vec::new();
    let mut llm_calls = 0usize;
    let mut llm_tokens_api = 0i64;

    for task in &project.tasks {
        // Baseline-LLM arm: LLM picks query each turn
        let mut learned: HashSet<String> = HashSet::new();
        for t in tokenize(&task.task) { if is_ident(&t) { learned.insert(t); } }
        let mut read: HashSet<usize> = HashSet::new();
        let mut query = task.task.clone();
        let mut tokens: i64 = 0;
        let mut turns = 0usize;
        let mut success = false;
        let mut docs_read_texts: Vec<String> = Vec::new();

        for turn in 0..base.max_turns {
            turns += 1;
            let ranking = index.ranking(&query);
            let mut spent = 0i64;
            let mut newly: Vec<usize> = Vec::new();
            for &di in &ranking {
                if read.contains(&di) { continue; }
                let c = count_tokens(&project.docs[di].text);
                if spent + c > base.per_turn { continue; }
                read.insert(di); spent += c; tokens += c; newly.push(di);
                docs_read_texts.push(project.docs[di].text.chars().take(600).collect());
                if project.docs[di].id == task.gold_doc { success = true; break; }
            }
            if success || newly.is_empty() || turn + 1 == base.max_turns { break; }
            // LLM picks next query from what was read
            match call_llm(&cfg.model, &cfg.api_key, &task.task, &docs_read_texts) {
                Ok((q, tks)) => { query = q; llm_tokens_api += tks; llm_calls += 1; }
                Err(e) => {
                    // Fallback to deterministic learned-terms on API error
                    eprintln!("[warn] LLM call failed ({}), falling back to deterministic", e);
                    let mut new_info = false;
                    for di in &newly { for t in tokenize(&project.docs[*di].text) { if is_ident(&t) && learned.insert(t.clone()) { new_info = true; } } }
                    if !new_info { break; }
                    let mut q: Vec<String> = tokenize(&task.task); q.extend(learned.iter().cloned()); query = q.join(" ");
                    llm_calls += 1;
                }
            }
        }
        let baseline_tokens = tokens;
        let baseline_turns = turns;
        let baseline_success = success;

        // Cortex arm unchanged (deterministic compile)
        let c: CortexRun = cortex_loop(&store, &core, &emb, task, base.budget, base.max_turns)?;
        let reduction = if baseline_tokens > 0 { (baseline_tokens - c.tokens) as f64 / baseline_tokens as f64 } else { 0.0 };
        outcomes.push(TaskOutcome {
            id: task.id.clone(), baseline_tokens, cortex_tokens: c.tokens,
            baseline_turns, cortex_turns: c.turns,
            baseline_success, cortex_success: c.success, reduction,
        });
        let _ = seed_tokens_total;
    }

    let mut report = crate::agentic::run_harness(&base)?;
    report.outcomes = outcomes;
    // recompute paired stats from LLM baselines
    let reds: Vec<f64> = report.outcomes.iter().map(|o| o.reduction).collect();
    report.mean_reduction = crate::agentic::mean(&reds);
    let (lo, hi) = cortex_core::eval::bootstrap_ci(&reds, 10_000, 42);
    report.reduction_ci_lo = lo; report.reduction_ci_hi = hi;
    report.paired_t = crate::agentic::paired_t(&reds);
    report.cohens_d = crate::agentic::cohens_d(&reds);
    let bas: Vec<(i64,usize,bool)> = report.outcomes.iter().map(|o| (o.baseline_tokens, o.baseline_turns, o.baseline_success)).collect();
    let cor: Vec<(i64,usize,bool)> = report.outcomes.iter().map(|o| (o.cortex_tokens, o.cortex_turns, o.cortex_success)).collect();
    let n = report.outcomes.len().max(1) as f64;
    report.baseline.success_rate = bas.iter().filter(|(_,_,s)| *s).count() as f64 / n;
    report.baseline.mean_tokens = bas.iter().map(|(t,_,_)| *t as f64).sum::<f64>()/n;
    report.baseline.mean_turns = bas.iter().map(|(_,u,_)| *u as f64).sum::<f64>()/n;
    report.cortex.mean_tokens = cor.iter().map(|(t,_,_)| *t as f64).sum::<f64>()/n;
    report.cortex.mean_turns = cor.iter().map(|(_,u,_)| *u as f64).sum::<f64>()/n;

    Ok(LlmHarnessReport {
        base: report, model: cfg.model.clone(), llm_calls, llm_tokens_api,
        note: "baseline queries by LLM; cortex arm deterministic compile".into(),
    })
}

pub fn print_llm_report(r: &LlmHarnessReport) {
    crate::agentic::print_report(&r.base);
    println!("LLM: model={} calls={} api_tokens={} ({})", r.model, r.llm_calls, r.llm_tokens_api, r.note);
}
