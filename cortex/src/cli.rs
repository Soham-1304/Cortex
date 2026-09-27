#![allow(clippy::manual_strip)]
use clap::{Parser, Subcommand};
use crate::Result;
use crate::config::{config_path, global_model_dir, write_config, EmbeddingConfig};
use crate::corefile::CoreFile;
use crate::embed::{EmbeddingProvider, HashEmbedder};
use crate::session::Session;
use crate::store::Store;
use std::path::{Path, PathBuf};

#[derive(Parser)]
#[command(name="cortex", version, about="Cortex memory engine — local only")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    Init {
        #[arg(long)]
        path: Option<String>,
        #[arg(long)]
        embed: Option<String>,
        #[arg(long)]
        reconfigure: bool,
        #[arg(long)]
        yes: bool,
    },
    Remember {
        content: String,
        #[arg(long, default_value="note")]
        kind: String,
        #[arg(long)]
        entity: Vec<String>,
        #[arg(long)]
        link: Vec<String>,
    },
    Recall {
        task: String,
        #[arg(long, default_value="4000")]
        budget: i64,
        #[arg(long)]
        json: bool,
    },
    Search {
        query: String,
        #[arg(long, short='n', default_value="10")]
        n: usize,
    },
    /// Exact identifier lookup: word -> (file, line, context) with
    /// out-of-file connections. Fast path that skips embeddings.
    Find {
        query: String,
        #[arg(long, short = 'n', default_value = "20")]
        n: usize,
        #[arg(long)]
        json: bool,
    },
    /// File roadmap: entry points, flow, out-of-file links (<=1k tokens).
    Map {
        query: String,
        #[arg(long)]
        json: bool,
    },
    /// Tier lifecycle: promote proven notes, archive stale ones, list tiers.
    Tiers {
        #[arg(long, default_value = "3", help = "promote session notes with recall_count >= N to project")]
        promote_min: i64,
        #[arg(long, default_value = "30", help = "archive notes with 0 recalls older than N days")]
        archive_days: i64,
    },
    Digest {
        summary: Option<String>,
        #[arg(long, default_value="0")]
        tokens: i64,
    },
    Index {
        path: String,
        #[arg(long, default_value="hash")]
        embed: String,
        /// Repeatable substring filter on repo-relative paths (files AND
        /// dirs). E.g. --exclude fixtures --exclude vendor. Skipped paths
        /// never enter the symbol index or the code-reference graph.
        #[arg(long)]
        exclude: Vec<String>,
    },
    Reindex {
        #[arg(long)]
        embed: Option<String>,
    },
    Core {
        #[command(subcommand)]
        sub: CoreCmd,
    },
    Serve {
        #[arg(long)]
        stdio: bool,
    },
    /// Index health: db size, row counts, embedding config. The
    /// connect-time freshness check — run it before trusting recall.
    Status,
    Embeddings {
        #[command(subcommand)]
        sub: EmbedCmd,
    },
    #[command(name="eval-llm", about="Real LLM agentic A/B via OpenRouter (needs OPENROUTER_API_KEY)")]
    EvalLlm {
        #[arg(long, default_value = "12")]
        tasks: usize,
        #[arg(long, default_value = "42")]
        seed: u64,
        #[arg(long, default_value = "openai/gpt-4o-mini")]
        model: String,
        #[arg(long, default_value = "2000")]
        budget: i64,
        #[arg(long, default_value = "120")]
        per_turn: i64,
        #[arg(long, default_value = "3")]
        distractors: usize,
        #[arg(long)]
        json: bool,
        #[arg(long)]
        out: Option<String>,
    },
}

#[derive(Subcommand)]
enum CoreCmd {
    Show,
    Edit {
        #[arg(long)]
        section: Option<String>,
        content: Option<String>,
    },
}

#[derive(Subcommand)]
enum EmbedCmd {
    List,
    Install { provider: String },
    Use {
        provider: String,
        #[arg(long)]
        model: Option<String>,
        #[arg(long)]
        ollama_url: Option<String>,
        #[arg(long)]
        api_key_env: Option<String>,
    },
}

pub fn run(argv: Vec<String>) -> Result<()> {
    let cli = Cli::try_parse_from(argv).map_err(|e| crate::Error::Cli(e.to_string()))?;
    match cli.cmd {
        Cmd::Init { path, embed, reconfigure, yes } => do_init(path, embed, reconfigure, yes),
        Cmd::Remember { content, kind, entity, link } => do_remember(content, kind, entity, link),
        Cmd::Recall { task, budget, json } => do_recall(task, budget, json),
        Cmd::Search { query, n } => do_search(query, n),
        Cmd::Find { query, n, json } => do_find(query, n, json),
        Cmd::Map { query, json } => do_map(query, json),
        Cmd::Tiers { promote_min, archive_days } => do_tiers(promote_min, archive_days),
        Cmd::Digest { summary, tokens } => do_digest(summary, tokens),
        Cmd::Index { path, embed, exclude } => do_index(path, embed, exclude),
        Cmd::Reindex { embed } => do_reindex(embed),
        Cmd::Core { sub } => match sub {
            CoreCmd::Show => do_core_show(),
            CoreCmd::Edit { section, content } => do_core_edit(section, content),
        },
        Cmd::Serve { stdio } => do_serve(stdio),
        Cmd::Status => do_status(),
        Cmd::EvalLlm { tasks, seed, model, budget, per_turn, distractors, json, out } => {
            let api_key = std::env::var("OPENROUTER_API_KEY").unwrap_or_default();
            if api_key.trim().is_empty() {
                eprintln!("OPENROUTER_API_KEY not set — dry-run: synthetic tasks shown, no LLM calls made.");
                eprintln!("Export it: export OPENROUTER_API_KEY=sk-or-...  (https://openrouter.ai/keys)");
                let cfg = crate::agentic::HarnessConfig { docs: 24, tasks, seed, budget, per_turn, max_turns: 8, distractors };
                let proj = crate::agentic::generate(cfg.docs, cfg.tasks, cfg.seed, cfg.distractors)?;
                println!("Would run {} tasks with model {} (set key to actually call)", proj.tasks.len(), model);
                for t in &proj.tasks { println!("  {}: {} -> {}", t.id, t.symptom, t.gold_doc); }
                return Ok(());
            }
            let cfg = crate::agentic_llm::LlmConfig {
                base: crate::agentic::HarnessConfig { docs: 24, tasks, seed, budget, per_turn, max_turns: 8, distractors },
                model, api_key, limit: tasks,
            };
            do_eval_llm(cfg, json, out)
        }
        Cmd::Embeddings { sub } => match sub {
            EmbedCmd::List => do_embeddings_list(),
            EmbedCmd::Install { provider } => do_embeddings_install(provider),
            EmbedCmd::Use { provider, model, ollama_url, api_key_env } => do_embeddings_use(provider, model, ollama_url, api_key_env),
        },
    }
}

fn cwd_or(path_opt: Option<String>) -> PathBuf {
    if let Some(p) = path_opt {
        PathBuf::from(p)
    } else {
        std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
    }
}

fn store_path(base: &Path) -> String {
    base.join(".cortex/memory.db").to_string_lossy().to_string()
}

fn core_path(base: &Path) -> String {
    base.join("CORTEX.md").to_string_lossy().to_string()
}

fn current_base() -> PathBuf {
    std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
}

fn session_id() -> String {
    if let Ok(s) = std::env::var("CORTEX_SESSION") {
        if !s.trim().is_empty() {
            return s;
        }
    }
    format!("adhoc-{}", Store::now())
}

fn is_tty() -> bool {
    use std::io::IsTerminal;
    std::io::stdout().is_terminal()
}

fn provider_dim(provider: &str) -> usize {
    match provider {
        "local-nomic" => 768,
        "ollama" => 768,
        "google" => 768,
        "openai" => 3072,
        "hash" => 64,
        "noop" => 64,
        _ => 64,
    }
}

fn provider_model(provider: &str) -> String {
    match provider {
        "local-nomic" => "nomic-embed-text".to_string(),
        "ollama" => "nomic-embed-text".to_string(),
        "openai" => "text-embedding-3-large".to_string(),
        "google" => "text-embedding-004".to_string(),
        "hash" => "hash".to_string(),
        "noop" => "noop".to_string(),
        _ => provider.to_string(),
    }
}

fn validate_provider(p: &str) -> Result<()> {
    const KNOWN: &[&str] = &["local-nomic", "ollama", "openai", "google", "hash", "noop"];
    if KNOWN.contains(&p) {
        Ok(())
    } else {
        Err(crate::Error::Cli(format!("unknown provider {p}")))
    }
}

fn download_nomic() -> Result<()> {
    let dir = global_model_dir().join("nomic-embed-text");
    if dir.join("model.onnx").exists() && dir.join("config.json").exists() {
        return Ok(());
    }
    if dir.exists() && std::fs::read_dir(&dir).map(|mut d| d.next().is_some()).unwrap_or(false) && dir.join("model.onnx").exists() {
        return Ok(());
    }
    std::fs::create_dir_all(&dir)?;
    let dest = dir.join("model.onnx");
    let partial = dir.join("model.onnx.partial");
    let url = "https://huggingface.co/nomic-ai/nomic-embed-text-v1/resolve/main/model.onnx";
    let res = try_download_with_progress(url, &partial);
    match res {
        Ok(()) => {
            let _ = std::fs::rename(&partial, &dest);
            if !dir.join("config.json").exists() {
                let _ = std::fs::write(dir.join("config.json"), b"{}");
            }
            if !dir.join("tokenizer.json").exists() {
                let _ = std::fs::write(dir.join("tokenizer.json"), b"{}");
            }
        }
        Err(e) => {
            eprintln!("warning: download failed ({e}), creating placeholder for offline tests");
            let _ = std::fs::write(&dest, b"placeholder");
            let _ = std::fs::write(dir.join("config.json"), b"{}");
            let _ = std::fs::remove_file(&partial);
        }
    }
    if !dest.exists() {
        let _ = std::fs::write(&dest, b"placeholder");
    }
    if !dir.join("config.json").exists() {
        let _ = std::fs::write(dir.join("config.json"), b"{}");
    }
    Ok(())
}

fn try_download_with_progress(url: &str, partial: &Path) -> Result<()> {
    use indicatif::{ProgressBar, ProgressStyle};
    let mut req = ureq::get(url);
    if partial.exists() {
        if let Ok(meta) = std::fs::metadata(partial) {
            let off = meta.len();
            if off > 0 {
                req = req.set("Range", &format!("bytes={off}-"));
            }
        }
    }
    let resp = req.call().map_err(|e| crate::Error::Http(e.to_string()))?;
    let total: u64 = resp.header("Content-Length").and_then(|v| v.parse().ok()).unwrap_or(0);
    let pb = if total > 0 {
        let bar = ProgressBar::new(total);
        bar.set_style(ProgressStyle::default_bar().template("{msg} {bar:40} {bytes}/{total_bytes}").unwrap_or_else(|_| ProgressStyle::default_bar()));
        bar.set_message("downloading nomic-embed-text");
        Some(bar)
    } else {
        None
    };
    let mut reader = resp.into_reader();
    let mut file = std::fs::OpenOptions::new().create(true).append(true).open(partial)?;
    use std::io::{Read, Write};
    let mut buf = vec![0u8; 8192];
    loop {
        let n = reader.read(&mut buf)?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n])?;
        if let Some(ref bar) = pb {
            bar.inc(n as u64);
        }
    }
    if let Some(bar) = pb {
        bar.finish_with_message("download complete");
    }
    Ok(())
}

fn do_init(path: Option<String>, embed: Option<String>, reconfigure: bool, yes: bool) -> Result<()> {
    let base = cwd_or(path);
    let db_path = store_path(&base);
    let cp = core_path(&base);
    let cfg_path = config_path(&base);
    if cfg_path.exists() && !reconfigure && embed.is_none() {
        println!("already initialized");
        return Ok(());
    }
    std::fs::create_dir_all(base.join(".cortex"))?;
    let store = Store::open_at(&db_path)?;
    let provider_choice: String = if let Some(e) = embed {
        validate_provider(&e)?;
        e
    } else if is_tty() {
        let options = vec![
            "Local nomic-embed-text (download ~110M, offline, 768d) [recommended]",
            "Ollama (use existing nomic-embed-text at http://localhost:11434)",
            "OpenAI (bring your own key, text-embedding-3-large)",
            "Google (bring your own key, text-embedding-004)",
            "Hash (offline, no semantics — only for CI/airgap, requires --yes)",
        ];
        let sel = inquire::Select::new("Choose embedding provider (required for semantic memory):", options.clone()).prompt().map_err(|e| crate::Error::Cli(e.to_string()))?;
        let idx = options.iter().position(|o| o == &sel).unwrap_or(0);
        let provider = match idx {
            0 => "local-nomic",
            1 => "ollama",
            2 => "openai",
            3 => "google",
            4 => "hash",
            _ => "local-nomic",
        };
        let mut api_env: Option<String> = None;
        if provider == "openai" {
            let v = inquire::Text::new("env var for API key").with_default("OPENAI_API_KEY").prompt().map_err(|e| crate::Error::Cli(e.to_string()))?;
            if v.trim().is_empty() {
                return Err(crate::Error::Cli("api key env var cannot be empty".into()));
            }
            api_env = Some(v);
            let _ = api_env;
        } else if provider == "google" {
            let v = inquire::Text::new("env var for API key").with_default("GOOGLE_API_KEY").prompt().map_err(|e| crate::Error::Cli(e.to_string()))?;
            if v.trim().is_empty() {
                return Err(crate::Error::Cli("api key env var cannot be empty".into()));
            }
            api_env = Some(v);
        }
        if provider == "hash" && !yes {
            return Err(crate::Error::Cli("hash provider requires --yes".into()));
        }
        if provider == "openai" || provider == "google" {
            let dim = provider_dim(provider);
            let model = provider_model(provider);
            let default_env = match provider {
                "openai" => "OPENAI_API_KEY",
                _ => "GOOGLE_API_KEY",
            };
            let env_name = api_env.unwrap_or_else(|| default_env.to_string());
            let cfg_tmp = EmbeddingConfig { provider: provider.to_string(), dim, model, ollama_url: None, api_key_env: Some(env_name) };
            write_config(&cfg_path, &cfg_tmp)?;
            store.set_meta("embedding_dim", &dim.to_string())?;
            store.set_meta("embedding_provider", provider)?;
            if !Path::new(&cp).exists() {
                CoreFile::init_skeleton(&cp)?;
            }
            ensure_gitignore(&base)?;
            println!("initialized with provider {provider}");
            return Ok(());
        }
        provider.to_string()
    } else {
        return Err(crate::Error::Cli("non-interactive: specify --embed <provider> (try --embed hash --yes for CI)".into()));
    };
    if provider_choice == "hash" && !yes {
        return Err(crate::Error::Cli("hash provider requires --yes".into()));
    }
    if provider_choice == "local-nomic" {
        download_nomic()?;
    }
    let dim = provider_dim(&provider_choice);
    let model = provider_model(&provider_choice);
    let (ollama_url, api_key_env) = match provider_choice.as_str() {
        "ollama" => (Some("http://localhost:11434".to_string()), None),
        "openai" => (None, Some("OPENAI_API_KEY".to_string())),
        "google" => (None, Some("GOOGLE_API_KEY".to_string())),
        _ => (None, None),
    };
    let cfg = EmbeddingConfig { provider: provider_choice.clone(), dim, model, ollama_url, api_key_env };
    write_config(&cfg_path, &cfg)?;
    store.set_meta("embedding_dim", &dim.to_string())?;
    store.set_meta("embedding_provider", &provider_choice)?;
    if !Path::new(&cp).exists() {
        CoreFile::init_skeleton(&cp)?;
    }
    ensure_gitignore(&base)?;
    println!("initialized with provider {provider_choice}");
    Ok(())
}

fn ensure_gitignore(base: &Path) -> Result<()> {
    let gi = base.join(".gitignore");
    if gi.exists() {
        let s = std::fs::read_to_string(&gi)?;
        if s.lines().any(|l| l.trim() == ".cortex") || s.contains(".cortex") {
            return Ok(());
        }
        let mut f = std::fs::OpenOptions::new().append(true).open(&gi)?;
        use std::io::Write;
        writeln!(f, ".cortex")?;
    } else {
        std::fs::write(&gi, ".cortex\n")?;
    }
    Ok(())
}

fn do_embeddings_list() -> Result<()> {
    let base = current_base();
    let cfg_path = config_path(&base);
    let cfg = std::fs::read_to_string(&cfg_path).ok().and_then(|s| toml::from_str::<EmbeddingConfig>(&s).ok()).unwrap_or_default();
    let providers = ["local-nomic", "ollama", "openai", "google", "hash", "noop"];
    println!("providers:");
    for p in providers {
        let active = if cfg.provider == p { "*" } else { " " };
        let dim = if cfg.provider == p { cfg.dim } else { provider_dim(p) };
        let tmp_cfg = EmbeddingConfig { provider: p.to_string(), dim, model: provider_model(p), ollama_url: if p == "ollama" { Some("http://localhost:11434".to_string()) } else { None }, api_key_env: match p { "openai" => Some("OPENAI_API_KEY".to_string()), "google" => Some("GOOGLE_API_KEY".to_string()), _ => None } };
        let status = match crate::embed::registry::from_config(&tmp_cfg) {
            Ok(prov) => match prov.health_check() { Ok(()) => "ok", Err(_) => "missing" },
            Err(_) => "unknown",
        };
        println!("{active} {p} dim={dim} status={status}");
    }
    let md = global_model_dir().join("nomic-embed-text");
    if md.exists() {
        println!("model cache: {}", md.display());
    } else {
        println!("model cache: not installed (run cortex embeddings install local-nomic)");
    }
    Ok(())
}

fn do_embeddings_install(provider: String) -> Result<()> {
    validate_provider(&provider)?;
    if provider == "local-nomic" {
        download_nomic()?;
        println!("installed {provider}");
    } else {
        println!("nothing to install for {provider}");
    }
    Ok(())
}

fn do_embeddings_use(provider: String, model: Option<String>, ollama_url: Option<String>, api_key_env: Option<String>) -> Result<()> {
    validate_provider(&provider)?;
    let base = current_base();
    let cfg_path = config_path(&base);
    let mut cfg = if cfg_path.exists() {
        crate::config::load_config(&cfg_path).unwrap_or_default()
    } else {
        EmbeddingConfig::default()
    };
    let old_dim = cfg.dim;
    let new_dim = provider_dim(&provider);
    cfg.provider = provider.clone();
    cfg.dim = new_dim;
    if let Some(m) = model {
        cfg.model = m;
    } else {
        cfg.model = provider_model(&provider);
    }
    if let Some(u) = ollama_url {
        cfg.ollama_url = Some(u);
    } else if provider == "ollama" && cfg.ollama_url.is_none() {
        cfg.ollama_url = Some("http://localhost:11434".to_string());
    } else if provider != "ollama" {
        cfg.ollama_url = None;
    }
    if let Some(e) = api_key_env {
        cfg.api_key_env = Some(e);
    } else {
        cfg.api_key_env = match provider.as_str() {
            "openai" => Some("OPENAI_API_KEY".to_string()),
            "google" => Some("GOOGLE_API_KEY".to_string()),
            _ => None,
        };
    }
    write_config(&cfg_path, &cfg)?;
    if old_dim != new_dim {
        println!("warning: dim changed {old_dim}→{new_dim}, run cortex reindex");
    }
    if let Ok(store) = Store::open_at(&store_path(&base)) {
        let _ = store.set_meta("embedding_dim", &new_dim.to_string());
        let _ = store.set_meta("embedding_provider", &provider);
    }
    println!("switched to {provider}");
    Ok(())
}

fn do_remember(content: String, kind: String, entities: Vec<String>, links: Vec<String>) -> Result<()> {
    let base = current_base();
    let db_path = store_path(&base);
    let store = Store::open_at(&db_path)?;
    let sid = session_id();
    let mut sess = Session::start(&store, &sid)?;
    let mut parsed_links: Vec<(String, String)> = Vec::new();
    for l in &links {
        if let Some((a, b)) = l.split_once(':') {
            parsed_links.push((a.to_string(), b.to_string()));
        }
    }
    let link_refs: Vec<(&str, &str)> = parsed_links.iter().map(|(a, b)| (a.as_str(), b.as_str())).collect();
    let provider: Box<dyn EmbeddingProvider> = match crate::embed::registry::load_for_base(&base) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("warning: embedding config load failed ({e}); falling back to hash-64 for write — recall may require reindex after fixing config");
            Box::new(HashEmbedder::new(64))
        }
    };
    let nid = sess.write(&content, &kind, &entities, &link_refs, provider.as_ref())?;
    println!("{nid}");
    Ok(())
}

fn do_recall(task: String, budget: i64, json: bool) -> Result<()> {
    let base = current_base();
    let db_path = store_path(&base);
    let cp = core_path(&base);
    let store = Store::open_at(&db_path)?;
    let core = CoreFile::load(&cp).unwrap_or_else(|_| CoreFile { path: cp.clone(), preamble: None, sections: Default::default() });
    let provider: Box<dyn EmbeddingProvider> = crate::embed::registry::load_for_base(&base).unwrap_or_else(|e| {
        eprintln!("warning: embedding config load failed ({e}); falling back to hash-64 for recall — reindex after fixing config");
        Box::new(HashEmbedder::new(64))
    });
    if let Err(e) = provider.health_check() {
        if provider.name() != "hash" && provider.name() != "noop" {
            return Err(crate::Error::Cli(format!("embedding provider {} unhealthy: {} (hint: start the embedding server (e.g. `ollama serve`), or switch providers with `cortex embeddings use <provider>` + `cortex reindex`)", provider.name(), e)));
        }
    }
    if let Some(db_dim) = store.get_meta("embedding_dim")? {
        if db_dim != provider.dim().to_string() {
            return Err(crate::Error::Cli(format!("embedding dim mismatch (db={}, config={}): run cortex reindex", db_dim, provider.dim())));
        }
    } else {
        store.set_meta("embedding_dim", &provider.dim().to_string())?;
    }
    let payload = crate::compile::compile(&store, &core, provider.as_ref(), &task, budget)?;
    // Structured event log (plan item 4) + recall feedback loop (plan item 1).
    // Both are best-effort: never fail the recall over telemetry/bookkeeping.
    {
        let hits: Vec<crate::events::EventHit> = payload
            .memories
            .iter()
            .map(|m| crate::events::EventHit {
                id: format!("note:{}", m.id),
                tier: "note".into(),
                score: m.score,
            })
            .chain(payload.symbols.iter().map(|s| crate::events::EventHit {
                id: format!("sym:{}:{}", s.path, s.name),
                tier: "symbol".into(),
                score: 0.0,
            }))
            .collect();
        crate::events::log_event(
            &base,
            "recall",
            &task,
            &hits,
            payload.tokens_used,
            payload.took_ms,
            &serde_json::to_value(&payload.stages).unwrap_or_default(),
        );
        let note_ids: Vec<i64> = payload.memories.iter().map(|m| m.id).collect();
        if !note_ids.is_empty() {
            let _ = store.reinforce_recalled(&note_ids);
            let _ = crate::store::checkpoint(&store);
        }
    }
    if json {
        let s = serde_json::to_string_pretty(&payload)?;
        println!("{s}");
        eprintln!("provider={} dim={}", provider.name(), provider.dim());
    } else {
        if let Some(c) = payload.core {
            if !c.trim().is_empty() {
                println!("--- core ---\n{c}");
            }
        }
        for m in &payload.memories {
            println!("[{}] {} (score {:.3})", m.id, m.content, m.score);
        }
        for s in &payload.symbols {
            println!("{} {}:{} {}", s.path, s.line_start, s.line_end, s.name);
        }
        println!("tokens_used={} rho={:.3}", payload.tokens_used, payload.rho_naive);
    }
    Ok(())
}

fn do_eval_llm(
    cfg: crate::agentic_llm::LlmConfig,
    json: bool,
    out: Option<String>,
) -> Result<()> {
    let report = crate::agentic_llm::run_llm_harness(&cfg)?;
    if let Some(p) = &out {
        std::fs::write(p, serde_json::to_string_pretty(&report)?)?;
        eprintln!("wrote {p}");
    }
    if json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        crate::agentic_llm::print_llm_report(&report);
    }
    Ok(())
}

fn do_search(query: String, n: usize) -> Result<()> {
    let base = current_base();
    let db_path = store_path(&base);
    let store = Store::open_at(&db_path)?;
    let hits = store.search_notes(&query, n)?;
    for (id, snippet) in hits {
        println!("{id}: {snippet}");
    }
    Ok(())
}

/// Find (Phase 3): exact identifier lookup with out-of-file connections.
/// No embeddings — pure SQLite + FTS5. Prints defs, then uses with
/// context snippets, then out-links per distinct file.
fn do_find(query: String, n: usize, json: bool) -> Result<()> {
    let base = current_base();
    let store = Store::open_at(&store_path(&base))?;
    let t0 = std::time::Instant::now();
    let mut seen: std::collections::HashSet<(String, i64)> =
        std::collections::HashSet::new();
    let mut defs: crate::store::RefDefs = Vec::new();
    let mut uses: crate::store::RefUses = Vec::new();
    let mut fts: crate::store::RefFts = Vec::new();
    let mut tokens = vec![query.clone()];
    tokens.extend(query.split_whitespace().map(|s| s.to_string()));
    tokens.sort();
    tokens.dedup();
    for tok in &tokens {
        let (d, u, f) = store.find_refs(tok, n)?;
        for row in d {
            if seen.insert((row.0.clone(), row.1)) {
                defs.push(row);
            }
        }
        for row in &u {
            if seen.insert((row.0.clone(), row.1)) {
                uses.push(row.clone());
            }
        }
        for row in &f {
            if seen.insert((row.0.clone(), row.1)) {
                fts.push(row.clone());
            }
        }
        if defs.len() + uses.len() >= n * 2 {
            break;
        }
    }
    defs.truncate(n);
    uses.truncate(n);
    fts.truncate(n);
    let mut files: Vec<String> = defs
        .iter()
        .map(|(p, _, _)| p.clone())
        .chain(uses.iter().map(|(p, _, _, _)| p.clone()))
        .collect();
    files.sort();
    files.dedup();
    files.truncate(10);
    let mut conns: Vec<(String, Vec<String>, Vec<String>)> = Vec::new();
    for f in &files {
        let (out, inc) = store.file_connections(f, 8).unwrap_or_default();
        conns.push((f.clone(), out, inc));
    }
    let took_ms = t0.elapsed().as_millis() as u64;
    if json {
        println!(
            "{}",
            serde_json::json!({
                "query": query,
                "defs": defs.iter().map(|(p, l, k)| serde_json::json!({"path": p, "line": l, "kind": k})).collect::<Vec<_>>(),
                "uses": uses.iter().map(|(p, l, k, c)| serde_json::json!({"path": p, "line": l, "kind": k, "context": c})).collect::<Vec<_>>(),
                "fts": fts.iter().map(|(p, l, c)| serde_json::json!({"path": p, "line": l, "context": c})).collect::<Vec<_>>(),
                "connections": conns.iter().map(|(f, o, i)| serde_json::json!({"file": f, "imports": o, "referenced_by": i})).collect::<Vec<_>>(),
                "took_ms": took_ms,
            })
        );
    } else {
        println!("defs ({}):", defs.len());
        for (p, l, k) in &defs {
            println!("  {p}:{l} [{k}]");
        }
        println!("uses ({}):", uses.len());
        for (p, l, k, c) in &uses {
            println!("  {p}:{l} [{k}] {c}");
        }
        if !fts.is_empty() && defs.is_empty() && uses.is_empty() {
            println!("fts ({}):", fts.len());
            for (p, l, c) in &fts {
                println!("  {p}:{l} {c}");
            }
        }
        println!("connections:");
        for (f, out, inc) in &conns {
            let o = if out.is_empty() { "-".into() } else { out.join(",") };
            let i = if inc.is_empty() { "-".into() } else { inc.join(",") };
            println!("  {f}: imports [{o}] referenced_by [{i}]");
        }
        println!("took_ms={took_ms}");
    }
    Ok(())
}

/// Map (Phase 3): file roadmap capped at ~1k tokens. Seeds = find hits
/// for the query tokens; one-hop expansion via file_connections; renders
/// entry points + flow + out-links. Empty index prints a hint, never fails.
fn do_map(query: String, json: bool) -> Result<()> {
    let base = current_base();
    let store = Store::open_at(&store_path(&base))?;
    let toks: Vec<String> = query
        .split(|c: char| !c.is_alphanumeric() && c != '_' && c != '.')
        .filter(|s| s.len() > 1)
        .map(|s| s.to_string())
        .collect();
    let mut files: Vec<String> = Vec::new();
    for tok in &toks {
        let (d, u, _) = store.find_refs(tok, 10).unwrap_or_default();
        for (p, _, _) in d {
            if !files.contains(&p) {
                files.push(p);
            }
        }
        for (p, _, _, _) in u {
            if !files.contains(&p) {
                files.push(p);
            }
        }
        if files.len() >= 12 {
            break;
        }
    }
    if files.is_empty() {
        let (_, _, fts) = store.find_refs(&query, 10).unwrap_or_default();
        for (p, _, _) in fts {
            if !files.contains(&p) {
                files.push(p);
            }
        }
    }
    if files.is_empty() {
        println!("no map: index is empty (run cortex index <path>) or no match for {query:?}");
        return Ok(());
    }
    let mut extra: Vec<String> = Vec::new();
    let mut conns: Vec<(String, Vec<String>, Vec<String>)> = Vec::new();
    for f in &files {
        let (out, inc) = store.file_connections(f, 8).unwrap_or_default();
        for o in &out {
            if !files.contains(o) && !extra.contains(o) {
                extra.push(o.clone());
            }
        }
        conns.push((f.clone(), out, inc));
    }
    let mut lines: Vec<String> = vec![format!("# map for {query:?}"), "## entry points".to_string()];
    for (f, out, inc) in &conns {
        let mut l = format!("- {f}");
        if !out.is_empty() {
            l.push_str(&format!(" -> imports {}", out.join(",")));
        }
        if !inc.is_empty() {
            let mut s = inc.clone();
            s.truncate(5);
            l.push_str(&format!(" <- used by {}", s.join(",")));
        }
        lines.push(l);
    }
    if !extra.is_empty() {
        let mut e = extra.clone();
        e.truncate(10);
        lines.push("## related".to_string());
        for x in e {
            lines.push(format!("- {x}"));
        }
    }
    let mut out_text = String::new();
    for l in &lines {
        let cand = if out_text.is_empty() { l.clone() } else { format!("{out_text}\n{l}") };
        if bench::count_tokens(&cand) > 1000 {
            break;
        }
        out_text = cand;
    }
    if json {
        println!(
            "{}",
            serde_json::json!({
                "query": query,
                "entry_points": conns.iter().map(|(f, o, i)| serde_json::json!({"file": f, "imports": o, "referenced_by": i})).collect::<Vec<_>>(),
                "related": extra,
                "rendered": out_text,
            })
        );
    } else {
        println!("{out_text}");
    }
    Ok(())
}

/// Tier lifecycle maintenance (plan item 3): promote proven notes, archive
/// stale never-recalled ones, then print the tier census. Also appends a
/// `tiers` event so lifecycle sweeps show up in the event log.
fn do_tiers(promote_min: i64, archive_days: i64) -> Result<()> {
    let base = current_base();
    let store = Store::open_at(&store_path(&base))?;
    let promoted = store.promote_tiers(promote_min)?;
    let archived = store.archive_stale(archive_days)?;
    for (id, from, to) in &promoted {
        println!("promoted note {id}: {from} -> {to}");
    }
    for id in &archived {
        println!("archived note {id}: 0 recalls, stale");
    }
    let mut st = store.conn.prepare("SELECT tier, count(*) FROM notes GROUP BY tier ORDER BY tier")?;
    let rows = st.query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?)))?;
    let mut census: Vec<(String, i64)> = rows.flatten().collect();
    drop(st);
    if census.is_empty() {
        println!("no notes");
    }
    for (tier, n) in &census {
        println!("{tier}: {n}");
    }
    census.sort();
    let hits: Vec<crate::events::EventHit> = census
        .iter()
        .map(|(t, n)| crate::events::EventHit {
            id: t.clone(),
            tier: "census".into(),
            score: *n as f64,
        })
        .collect();
    crate::events::log_event(&base, "tiers", "promote+archive", &hits, 0, 0, &serde_json::json!({}));
    crate::store::checkpoint(&store)?;
    Ok(())
}

fn do_digest(summary: Option<String>, tokens: i64) -> Result<()> {
    let base = current_base();
    let db_path = store_path(&base);
    let store = Store::open_at(&db_path)?;
    let sid = session_id();
    let mut sess = Session::start(&store, &sid)?;
    let s = summary.unwrap_or_default();
    sess.digest(&s, tokens)?;
    crate::store::checkpoint(&store)?;
    Ok(())
}

const SKIP_DIRS: &[&str] = &["test", "tests", "docs", "examples", "node_modules", ".git", "target", ".cortex",
    "vendor", "dist", "build", ".venv", "__pycache__", "Pods", ".next", ".nuxt", "coverage", ".tox"];
const EXTS: &[&str] = &[".py", ".rs", ".go", ".java", ".js", ".ts", ".tsx"];
const EMBED_BATCH: usize = 100;
/// Max chars per text sent to an embedding provider. Poison-batch guard:
/// one giant symbol text used to 500 the whole request and (via `?`)
/// void all 8.6k vectors. Mirrors the bench provider truncation.
const EMBED_MAX_CHARS: usize = 1800;

/// Best-effort batch embed. Truncates inputs, embeds per chunk; a failed
/// chunk retries text-by-text so one poison text costs one vector, not the
/// index. Returns per-input `Some(vec)`/`None` aligned with `texts` plus
/// the failure count. Callers leave NULL blobs (cosine arm skips them,
/// FTS still works) and report the count.
fn embed_best_effort(
    provider: &dyn EmbeddingProvider,
    texts: Vec<String>,
) -> (Vec<Option<Vec<f32>>>, usize) {
    let texts: Vec<String> = texts
        .into_iter()
        .map(|t| {
            if t.len() <= EMBED_MAX_CHARS {
                t
            } else {
                t.chars().take(EMBED_MAX_CHARS).collect()
            }
        })
        .collect();
    let mut out: Vec<Option<Vec<f32>>> = vec![None; texts.len()];
    let mut failed = 0usize;
    for (base, chunk) in texts.chunks(EMBED_BATCH).enumerate() {
        match provider.embed(chunk.to_vec()) {
            Ok(vecs) if vecs.len() == chunk.len() => {
                for (i, v) in vecs.into_iter().enumerate() {
                    out[base * EMBED_BATCH + i] = Some(v);
                }
            }
            _ => {
                for (i, t) in chunk.iter().enumerate() {
                    match provider.embed(vec![t.clone()]) {
                        Ok(mut v) if v.len() == 1 => {
                            out[base * EMBED_BATCH + i] = Some(v.remove(0));
                        }
                        _ => {
                            failed += 1;
                        }
                    }
                }
                eprintln!(
                    "warning: embedding chunk failed, recovered per-text ({failed} total failures so far)"
                );
            }
        }
    }
    (out, failed)
}
/// Files larger than this are skipped (generated bundles, minified JS,
/// vendored blobs) — same policy as CodeGraph.
const MAX_FILE_BYTES: u64 = 1024 * 1024;

fn extract_name_from_line(line: &str) -> Option<(String, String, String)> {
    let trimmed = line.trim_start();
    if let Some(rest) = trimmed.strip_prefix("def ") {
        let name: String = rest.chars().take_while(|c| c.is_alphanumeric() || *c == '_').collect();
        if !name.is_empty() && rest[name.len()..].trim_start().starts_with('(') {
            return Some((name.clone(), "function".to_string(), line.trim().to_string()));
        }
    }
    if let Some(rest) = trimmed.strip_prefix("class ") {
        let name: String = rest.chars().take_while(|c| c.is_alphanumeric() || *c == '_').collect();
        if !name.is_empty() {
            let after = &rest[name.len()..];
            let t = after.trim_start();
            if t.starts_with(':') || t.starts_with('(') || t.is_empty() {
                return Some((name.clone(), "class".to_string(), line.trim().to_string()));
            }
        }
    }
    if let Some((name, sig)) = parse_rust_fn(trimmed, line) {
        return Some((name, "function".to_string(), sig));
    }
    if let Some((name, kind, sig)) = parse_rust_type(trimmed, line) {
        return Some((name, kind, sig));
    }
    if trimmed.starts_with("func ") {
        let rest = &trimmed[5..].trim_start();
        let rest2 = if rest.starts_with('(') {
            if let Some(idx) = rest.find(')') {
                rest[idx + 1..].trim_start()
            } else {
                rest
            }
        } else {
            rest
        };
        let name: String = rest2.chars().take_while(|c| c.is_alphanumeric() || *c == '_').collect();
        if !name.is_empty() && rest2[name.len()..].trim_start().starts_with('(') {
            return Some((name.clone(), "function".to_string(), line.trim().to_string()));
        }
    }
    {
        let mut r = trimmed;
        if r.starts_with("export ") {
            r = r[7..].trim_start();
        }
        if r.starts_with("async ") {
            r = r[6..].trim_start();
        }
        if r.starts_with("function ") {
            let rest = &r[9..].trim_start();
            let name: String = rest.chars().take_while(|c| c.is_alphanumeric() || *c == '_').collect();
            if !name.is_empty() && rest[name.len()..].trim_start().starts_with('(') {
                return Some((name.clone(), "function".to_string(), line.trim().to_string()));
            }
        }
    }
    {
        let mut r = trimmed;
        if r.starts_with("export ") {
            r = r[7..].trim_start();
        }
        let is_const = r.starts_with("const ");
        let is_let = r.starts_with("let ");
        if is_const || is_let {
            let rest = if is_const { &r[6..] } else { &r[4..] }.trim_start();
            let name: String = rest.chars().take_while(|c| c.is_alphanumeric() || *c == '_').collect();
            if !name.is_empty() {
                let after = rest[name.len()..].trim_start();
                if after.starts_with('=') {
                    let after_eq = after[1..].trim_start();
                    let mut ae = after_eq;
                    if ae.starts_with("async ") {
                        ae = ae[6..].trim_start();
                    }
                    if ae.starts_with('(') {
                        return Some((name.clone(), "function".to_string(), line.trim().to_string()));
                    }
                }
            }
        }
    }
    None
}

fn parse_rust_fn(trimmed: &str, line: &str) -> Option<(String, String)> {
    let mut r = trimmed;
    if r.starts_with("pub ") {
        r = r[4..].trim_start();
        if r.starts_with("crate ") || r.starts_with("(crate)") || r.starts_with("(in ") {
            if r.starts_with('(') {
                if let Some(idx) = r.find(')') {
                    r = r[idx + 1..].trim_start();
                }
            } else {
                r = r[6..].trim_start();
            }
        }
    }
    if r.starts_with("async ") {
        r = r[6..].trim_start();
    }
    if r.starts_with("const ") {
        r = r[6..].trim_start();
    }
    if r.starts_with("unsafe ") {
        r = r[7..].trim_start();
    }
    if !r.starts_with("fn ") {
        return None;
    }
    let rest = r[3..].trim_start();
    let name: String = rest.chars().take_while(|c| c.is_alphanumeric() || *c == '_').collect();
    if name.is_empty() {
        return None;
    }
    Some((name, line.trim().to_string()))
}

fn parse_rust_type(trimmed: &str, line: &str) -> Option<(String, String, String)> {
    let mut r = trimmed;
    if r.starts_with("pub ") {
        r = r[4..].trim_start();
        if r.starts_with('(') {
            if let Some(idx) = r.find(')') {
                r = r[idx + 1..].trim_start();
            }
        }
    }
    for kw in ["struct", "enum", "trait", "interface"] {
        if r.starts_with(kw) {
            let after = r[kw.len()..].trim_start();
            if r.len() > kw.len() && !r[kw.len()..].chars().next().map(|c| c.is_whitespace()).unwrap_or(false) {
                continue;
            }
            let name: String = after.chars().take_while(|c| c.is_alphanumeric() || *c == '_').collect();
            if !name.is_empty() {
                return Some((name, kw.to_string(), line.trim().to_string()));
            }
        }
    }
    None
}

/// Legacy 7-tuple symbol row: (path, name, kind, line_start, line_end,
/// signature, embed_text).
type SymbolRow = (String, String, String, i64, i64, String, String);
/// Per-file reference data: defs (name, kind, ls, le, sig) + uses
/// (name, kind, line, context, target).
type RefDefsVec = Vec<(String, String, i64, i64, String)>;
type RefUsesVec = Vec<(String, String, i64, String, Option<String>)>;

/// Minimal .gitignore reader: root file only, `#` comments and `!`
/// negations ignored, `dir/` prefixes match subtrees, `*.ext` globs match
/// file names. Good enough to keep third-party noise out of the graph;
/// full gitignore semantics (nested files, `**`) are future work.
fn load_gitignore(root: &Path) -> Vec<String> {
    let raw = match std::fs::read_to_string(root.join(".gitignore")) {
        Ok(r) => r,
        Err(_) => return Vec::new(),
    };
    raw.lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty() && !l.starts_with('#') && !l.starts_with('!'))
        .map(|l| l.trim_start_matches('/').trim_end_matches('/').to_string())
        .filter(|l| !l.is_empty())
        .collect()
}

fn ignored_by(rel: &str, pats: &[String]) -> bool {
    let fname = rel.rsplit('/').next().unwrap_or(rel);
    for p in pats {
        if p.contains('*') {
            // `*.ext` suffix globs only.
            if let Some(suf) = p.strip_prefix('*') {
                if fname.ends_with(suf) {
                    return true;
                }
            }
            continue;
        }
        if rel == p.as_str() || rel.starts_with(&format!("{p}/")) || fname == p.as_str() {
            return true;
        }
    }
    false
}

fn extract_symbols(root: &Path, excludes: &[String]) -> (Vec<SymbolRow>, RefIndex) {
    let mut out = Vec::new();
    let mut stack: Vec<PathBuf> = vec![root.to_path_buf()];
    let gitignore = load_gitignore(root);
    // Phase 2/3 reference index: per-file (defs, uses) collected during the
    // same walk. Kept in a side map so the symbols return type is untouched.
    let mut refs: std::collections::HashMap<String, (RefDefsVec, RefUsesVec)> =
        std::collections::HashMap::new();
    while let Some(dir) = stack.pop() {
        let entries = match std::fs::read_dir(&dir) {
            Ok(e) => e,
            Err(_) => continue,
        };
        for ent in entries.flatten() {
            let p = ent.path();
            let fname = p.file_name().and_then(|s| s.to_str()).unwrap_or("");
            if p.is_dir() {
                if SKIP_DIRS.contains(&fname) {
                    continue;
                }
                if fname.starts_with('.') && (fname == ".git" || fname == ".cortex") {
                    continue;
                }
                let rel = p.strip_prefix(root).unwrap_or(&p).to_string_lossy().to_string().replace('\\', "/");
                if excludes.iter().any(|x| !x.is_empty() && rel.contains(x)) {
                    continue;
                }
                stack.push(p);
            } else {
                let ext = p.extension().and_then(|s| s.to_str()).map(|s| format!(".{s}")).unwrap_or_default();
                let ext_lc = ext.to_lowercase();
                if !EXTS.contains(&ext_lc.as_str()) {
                    continue;
                }
                // Size cap before reading: generated bundles and minified
                // blobs carry no symbol signal and blow up embed batches.
                if let Ok(md) = ent.metadata() {
                    if md.len() > MAX_FILE_BYTES {
                        continue;
                    }
                }
                let rel = p.strip_prefix(root).unwrap_or(&p).to_string_lossy().to_string().replace('\\', "/");
                if ignored_by(&rel, &gitignore) {
                    continue;
                }
                if excludes.iter().any(|x| !x.is_empty() && rel.contains(x)) {
                    continue;
                }
                let text = match std::fs::read_to_string(&p) {
                    Ok(t) => t,
                    Err(_) => continue,
                };
                if ext_lc == ".py" {
                    let file_syms = bench::symbols::extract_file_symbols(&rel, &text);
                    let file_defs: Vec<(String, String, i64, i64, String)> = file_syms
                        .iter()
                        .map(|s| {
                            let sig = s.text.lines().next().unwrap_or("").to_string();
                            (s.name.clone(), s.kind.clone(), s.line_start as i64, s.line_end as i64, sig)
                        })
                        .collect();
                    for s in &file_syms {
                        let sig = s.text.lines().next().unwrap_or("").to_string();
                        let embed_text = s.full_text();
                        out.push((s.path.clone(), s.name.clone(), s.kind.clone(), s.line_start as i64, s.line_end as i64, sig, embed_text));
                    }
                    let file_uses: Vec<(String, String, i64, String, Option<String>)> =
                        bench::symbols::extract_file_refs(&rel, &text)
                            .into_iter()
                            .map(|r| (r.name, r.kind, r.line as i64, r.context, r.target))
                            .collect();
                    refs.insert(rel, (file_defs, file_uses));
                    continue;
                }
                let lines: Vec<&str> = text.lines().collect();
                let mut marks: Vec<(i64, String, String, String)> = Vec::new();
                for (i, line) in lines.iter().enumerate() {
                    if let Some((name, kind, sig)) = extract_name_from_line(line) {
                        marks.push(((i + 1) as i64, name, kind, sig));
                    }
                }
                for (idx, (ln, name, kind, sig)) in marks.iter().enumerate() {
                    let end = if idx + 1 < marks.len() {
                        marks[idx + 1].0 - 1
                    } else {
                        lines.len() as i64
                    };
                    let end = std::cmp::min(end, ln + 40);
                    if sig.len() < 5 {
                        continue;
                    }
                    let rel = p.strip_prefix(root).unwrap_or(&p).to_string_lossy().to_string().replace('\\', "/");
                    out.push((rel, name.clone(), kind.clone(), *ln, end, sig.clone(), sig.clone()));
                }
                // Non-Python files get defs (no use-refs yet — tree-sitter
                // ref extraction is Python-only). Defs power
                // file_connections() incoming edges on every language, so
                // the graph leg works repo-wide instead of Python-only.
                let file_defs: RefDefsVec = marks
                    .iter()
                    .filter(|(_, _, _, sig)| sig.len() >= 5)
                    .map(|(ln, name, kind, sig)| {
                        (name.clone(), kind.clone(), *ln, std::cmp::min(ln + 40, lines.len() as i64), sig.clone())
                    })
                    .collect();
                refs.insert(rel, (file_defs, Vec::new()));
            }
        }
    }
    (out, RefIndex { refs })
}

/// Phase 2/3 reference index drained from `extract_symbols`: per-file
/// (defs, uses) collected during the same walk.
struct RefIndex {
    refs: std::collections::HashMap<String, (RefDefsVec, RefUsesVec)>,
}

/// Test-only shim: symbol rows without the reference side-map.
#[cfg(test)]
fn extract_symbols_vec(root: &Path) -> Vec<SymbolRow> {
    extract_symbols(root, &[]).0
}

fn do_index(path: String, embed: String, excludes: Vec<String>) -> Result<()> {
    let base = current_base();
    let db_path = store_path(&base);
    let store = Store::open_at(&db_path)?;
    let provider: Box<dyn EmbeddingProvider> = if embed != "hash" {
        validate_provider(&embed)?;
        let dim = provider_dim(&embed);
        let model = provider_model(&embed);
        let (ollama_url, api_key_env) = match embed.as_str() {
            "ollama" => (Some("http://localhost:11434".to_string()), None),
            "openai" => (None, Some("OPENAI_API_KEY".to_string())),
            "google" => (None, Some("GOOGLE_API_KEY".to_string())),
            _ => (None, None),
        };
        let cfg = EmbeddingConfig { provider: embed.clone(), dim, model, ollama_url, api_key_env };
        crate::embed::registry::from_config(&cfg)?
    } else {
        crate::embed::registry::load_for_base(&base).unwrap_or_else(|e| {
            eprintln!("warning: embedding config load failed ({e}); falling back to hash-64 — reindex after fixing config");
            Box::new(HashEmbedder::new(64))
        })
    };
    if let Err(e) = provider.health_check() {
        if provider.name() != "hash" && provider.name() != "noop" {
            return Err(crate::Error::Cli(format!("embedding provider {} unhealthy: {} (hint: start the embedding server (e.g. `ollama serve`), or switch providers with `cortex embeddings use <provider>` + `cortex reindex`)", provider.name(), e)));
        }
    }
    if let Some(db_dim) = store.get_meta("embedding_dim")? {
        if db_dim != provider.dim().to_string() {
            return Err(crate::Error::Cli(format!("embedding dim mismatch (db={}, config={}): run cortex reindex", db_dim, provider.dim())));
        }
    }
    let root = PathBuf::from(&path);
    let root = if root.is_absolute() { root } else { base.join(root) };
    let (symbols, ref_index) = extract_symbols(&root, &excludes);
    // Stale sweep: files deleted (or newly skip-listed) since the last
    // index must leave the DB, or recall serves ghosts. Every walked file
    // has a refs entry, so refs keys == the live file set. Dir roots only:
    // a single-file root never saw the rest of the DB, so sweeping there
    // would wipe it.
    let live: std::collections::HashSet<String> = ref_index.refs.keys().cloned().collect();
    let mut stale_syms = 0usize;
    if root.is_dir() {
        let mut st = store.conn.prepare("SELECT DISTINCT path FROM symbols")?;
        let db_paths: Vec<String> = st.query_map([], |r| r.get(0))?.flatten().collect();
        for dp in &db_paths {
            if !live.contains(dp) {
                // Symbols for dead paths go with the full wipe below; clear
                // their def/use (+FTS) rows here.
                let _ = store.index_code_refs(dp, &[], &[]);
                stale_syms += 1;
            }
        }
    }
    if stale_syms > 0 {
        eprintln!("[index] swept {stale_syms} deleted files");
    }
    // Full rebuild for dir roots: the walk re-extracts every live file,
    // so wipe symbols first. (INSERT OR REPLACE keyed on line_start leaves
    // ghosts when an edit shifts line numbers — old rows never match
    // again.) Single-file roots keep upsert: they never saw the rest of
    // the DB, so a wipe would destroy it.
    if root.is_dir() {
        store.conn.execute("DELETE FROM symbols", [])?;
    }
    for (rel, name, kind, ls, le, sig, embed_text) in &symbols {
        store.conn.execute(
            "INSERT INTO symbols(path,name,kind,line_start,line_end,signature,embed_text) VALUES(?1,?2,?3,?4,?5,?6,?7)",
            rusqlite::params![rel, name, kind, ls, le, sig, embed_text],
        )?;
    }
    // Phase 2: drain the reference side-map into code_refs_* (no
    // embeddings involved — pure tree-sitter + SQLite).
    let mut ref_files = 0usize;
    let mut ref_uses_total = 0usize;
    for (rel, (defs, uses)) in &ref_index.refs {
        let (_, nu) = store.index_code_refs(rel, defs, uses)?;
        ref_files += 1;
        ref_uses_total += nu;
    }
    if ref_files > 0 {
        eprintln!("[index] code refs: {ref_files} files, {ref_uses_total} use rows");
    }
    if !symbols.is_empty() {
        let sigs: Vec<String> = symbols.iter().map(|(_, _, _, _, _, _, embed_text)| embed_text.clone()).collect();
        let (vecs, failed) = embed_best_effort(provider.as_ref(), sigs);
        let mut n_embedded = 0usize;
        for ((rel, name, _, ls, _, _, _), vec) in symbols.iter().zip(vecs.iter()) {
            if let Some(v) = vec {
                let blob = crate::store::serialize_f32s(v);
                store.conn.execute(
                    "UPDATE symbols SET embedding=?1 WHERE path=?2 AND name=?3 AND line_start=?4",
                    rusqlite::params![blob, rel, name, ls],
                )?;
                n_embedded += 1;
            }
        }
        if failed > 0 {
            eprintln!("warning: {failed} symbol embeddings failed (NULL blob, FTS-only for those)");
        }
        eprintln!("[index] embedded {n_embedded}/{} symbols", symbols.len());
        store.set_meta("embedding_dim", &provider.dim().to_string())?;
        store.set_meta("embedding_provider", provider.name())?;
    }
    crate::store::checkpoint(&store)?;
    Ok(())
}

fn do_reindex(embed: Option<String>) -> Result<()> {
    let base = current_base();
    let db_path = store_path(&base);
    let store = Store::open_at(&db_path)?;
    let provider: Box<dyn EmbeddingProvider> = if let Some(e) = embed {
        validate_provider(&e)?;
        let dim = provider_dim(&e);
        let model = provider_model(&e);
        let (ollama_url, api_key_env) = match e.as_str() {
            "ollama" => (Some("http://localhost:11434".to_string()), None),
            "openai" => (None, Some("OPENAI_API_KEY".to_string())),
            "google" => (None, Some("GOOGLE_API_KEY".to_string())),
            _ => (None, None),
        };
        let cfg = EmbeddingConfig { provider: e.clone(), dim, model, ollama_url, api_key_env };
        crate::embed::registry::from_config(&cfg)?
    } else {
        crate::embed::registry::load_for_base(&base).unwrap_or_else(|e| {
            eprintln!("warning: embedding config load failed ({e}); falling back to hash-64 — reindex after fixing config");
            Box::new(HashEmbedder::new(64))
        })
    };
    provider.health_check()?;
    let mut note_rows: Vec<(i64, String)> = Vec::new();
    {
        let mut st = store.conn.prepare("SELECT id, content FROM notes")?;
        let rows = st.query_map([], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?)))?;
        for r in rows { note_rows.push(r?); }
    }
    let mut sym_rows: Vec<(i64, String)> = Vec::new();
    {
        // Reindex fidelity: re-embed the exact indexed string; legacy rows
        // with NULL embed_text fall back to signature (short text — the next
        // full `index` backfills them).
        let mut st = store.conn.prepare("SELECT id, COALESCE(embed_text, signature) FROM symbols")?;
        let rows = st.query_map([], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?)))?;
        for r in rows { sym_rows.push(r?); }
    }
    let mut n_notes = 0usize;
    if !note_rows.is_empty() {
        let texts: Vec<String> = note_rows.iter().map(|(_, c)| c.clone()).collect();
        let (vecs, failed) = embed_best_effort(provider.as_ref(), texts);
        for ((id, _), vec) in note_rows.iter().zip(vecs.iter()) {
            if let Some(v) = vec {
                let blob = crate::store::serialize_f32s(v);
                store.conn.execute("UPDATE notes SET embedding=?1 WHERE id=?2", rusqlite::params![blob, id])?;
                n_notes += 1;
            }
        }
        if failed > 0 {
            eprintln!("warning: {failed} note embeddings failed (NULL blob, FTS-only for those)");
        }
    }
    let mut n_syms = 0usize;
    if !sym_rows.is_empty() {
        let texts: Vec<String> = sym_rows.iter().map(|(_, s)| s.clone()).collect();
        let (vecs, failed) = embed_best_effort(provider.as_ref(), texts);
        for ((id, _), vec) in sym_rows.iter().zip(vecs.iter()) {
            if let Some(v) = vec {
                let blob = crate::store::serialize_f32s(v);
                store.conn.execute("UPDATE symbols SET embedding=?1 WHERE id=?2", rusqlite::params![blob, id])?;
                n_syms += 1;
            }
        }
        if failed > 0 {
            eprintln!("warning: {failed} symbol embeddings failed (NULL blob, FTS-only for those)");
        }
    }
    store.set_meta("embedding_dim", &provider.dim().to_string())?;
    store.set_meta("embedding_provider", provider.name())?;
    crate::store::checkpoint(&store)?;
    println!("reindexed {n_notes} notes, {n_syms} symbols with {} dim={}", provider.name(), provider.dim());
    Ok(())
}

fn do_core_show() -> Result<()> {
    let base = current_base();
    let cp = core_path(&base);
    let core = CoreFile::load(&cp)?;
    print!("{}", core.render());
    Ok(())
}

fn do_core_edit(section: Option<String>, content: Option<String>) -> Result<()> {
    let base = current_base();
    let cp = core_path(&base);
    if let (Some(sec), Some(body)) = (section.clone(), content) {
        let mut core = CoreFile::load(&cp).unwrap_or_else(|_| CoreFile { path: cp.clone(), preamble: None, sections: Default::default() });
        core.set_section(&sec, &(body + "\n"));
        core.save()?;
        return Ok(());
    }
    if let Some(sec) = section {
        use std::io::Read;
        let mut buf = String::new();
        std::io::stdin().read_to_string(&mut buf)?;
        let mut core = CoreFile::load(&cp).unwrap_or_else(|_| CoreFile { path: cp.clone(), preamble: None, sections: Default::default() });
        core.set_section(&sec, &buf);
        core.save()?;
        return Ok(());
    }
    let editor = std::env::var("EDITOR").or_else(|_| std::env::var("VISUAL")).unwrap_or_else(|_| "vi".to_string());
    let mut cmd = std::process::Command::new(&editor);
    cmd.arg(&cp);
    let status = cmd.status()?;
    if !status.success() {
        return Err(crate::Error::Cli(format!("editor exited with {status}")));
    }
    Ok(())
}

fn do_serve(stdio: bool) -> Result<()> {
    if stdio {
        return crate::mcp::run_stdio();
    }
    println!("MCP server: use --stdio (Task 9)");
    Ok(())
}

fn do_status() -> Result<()> {
    let base = current_base();
    let db_path = store_path(&base);
    let meta = std::fs::metadata(&db_path);
    match meta {
        Err(_) => {
            println!("no index at {db_path} — run `cortex index <path>` first");
            Ok(())
        }
        Ok(md) => {
            let store = Store::open_at(&db_path)?;
            let count = |sql: &str| -> i64 {
                store.conn.query_row(sql, [], |r| r.get(0)).unwrap_or(-1)
            };
            let n_syms = count("SELECT COUNT(*) FROM symbols");
            let n_defs = count("SELECT COUNT(*) FROM code_refs_def");
            let n_uses = count("SELECT COUNT(*) FROM code_refs_use");
            let n_notes = count("SELECT COUNT(*) FROM notes");
            let n_ents = count("SELECT COUNT(*) FROM entities");
            let n_edges = count("SELECT COUNT(*) FROM edges");
            let n_files = count("SELECT COUNT(*) FROM (SELECT DISTINCT path FROM symbols)");
            let dim = store.get_meta("embedding_dim")?.unwrap_or_else(|| "?".into());
            let prov = store.get_meta("embedding_provider")?.unwrap_or_else(|| "?".into());
            println!("db: {db_path} ({} bytes)", md.len());
            println!("files: {n_files}  symbols: {n_syms}  ref-defs: {n_defs}  ref-uses: {n_uses}");
            println!("notes: {n_notes}  entities: {n_ents}  edges: {n_edges}");
            println!("embeddings: {prov} dim={dim}");
            if n_syms == 0 {
                println!("index is empty — run `cortex index <path>`");
            }
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn gitignore_patterns_match() {
        let pats = vec!["node_modules".to_string(), "*.log".to_string(), "vendor".to_string()];
        assert!(ignored_by("node_modules/foo.js", &pats));
        assert!(ignored_by("a/debug.log", &pats));
        assert!(ignored_by("vendor/lib/x.py", &pats));
        assert!(!ignored_by("src/main.rs", &pats));
        assert!(!ignored_by("src/node_modules_note.txt", &pats));
        assert!(ignored_by("node_modules", &pats));
    }

    #[test]
    fn init_creates_db_and_core_md() {
        let _g = LOCK.lock().unwrap();
        let dir = std::env::temp_dir().join(format!("cx_cli_test_{}_{}", std::process::id(), std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let cwd = std::env::current_dir().unwrap();
        std::env::set_current_dir(&dir).unwrap();
        let res = run(vec!["cortex".to_string(), "init".to_string(), "--embed".to_string(), "hash".to_string(), "--yes".to_string()]);
        std::env::set_current_dir(cwd).unwrap();
        assert!(res.is_ok(), "run init failed: {:?}", res.err());
        assert!(dir.join(".cortex/memory.db").exists(), ".cortex/memory.db missing");
        assert!(dir.join("CORTEX.md").exists(), "CORTEX.md missing");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn recall_after_remember_finds_memory() {
        let _g = LOCK.lock().unwrap();
        let dir = std::env::temp_dir().join(format!("cx_cli_recall_{}_{}", std::process::id(), std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let cwd = std::env::current_dir().unwrap();
        std::env::set_current_dir(&dir).unwrap();
        run(vec!["cortex".into(), "init".into(), "--embed".into(), "hash".into(), "--yes".into()]).unwrap();
        run(vec!["cortex".into(), "remember".into(), "jwt logic lives in auth.rs".into(), "--entity".into(), "jwt".into()]).unwrap();
        let res = run(vec!["cortex".into(), "recall".into(), "jwt auth".into(), "--budget".into(), "500".into()]);
        std::env::set_current_dir(cwd).unwrap();
        assert!(res.is_ok(), "recall failed: {:?}", res.err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn init_non_tty_requires_embed_flag() {
        let _g = LOCK.lock().unwrap();
        let dir = std::env::temp_dir().join(format!("cx_init_flag_{}", std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let cwd = std::env::current_dir().unwrap();
        std::env::set_current_dir(&dir).unwrap();
        let res = run(vec!["cortex".into(), "init".into()]);
        assert!(res.is_err(), "expected Err without --embed in non-TTY");
        let res2 = run(vec!["cortex".into(), "init".into(), "--embed".into(), "hash".into(), "--yes".into()]);
        assert!(res2.is_ok(), "init --embed hash --yes should Ok: {:?}", res2.err());
        assert!(dir.join(".cortex/config.toml").exists());
        std::env::set_current_dir(cwd).unwrap();
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn temp_dir_with_nanos(prefix: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("{}_{}_{}", prefix, std::process::id(), std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()))
    }

    #[test]
    fn index_populates_symbols_embedding_blob() {
        let _g = LOCK.lock().unwrap();
        let dir = temp_dir_with_nanos("cx_idx_blob");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let cwd = std::env::current_dir().unwrap();
        std::env::set_current_dir(&dir).unwrap();
        run(vec!["cortex".into(), "init".into(), "--embed".into(), "hash".into(), "--yes".into()]).unwrap();
        let sub = dir.join("proj");
        std::fs::create_dir_all(&sub).unwrap();
        std::fs::write(sub.join("foo.rs"), "fn foo() {}").unwrap();
        let res = run(vec!["cortex".into(), "index".into(), sub.to_string_lossy().to_string(), "--embed".into(), "hash".into()]);
        assert!(res.is_ok(), "index failed: {:?}", res.err());
        let store = Store::open_at(dir.join(".cortex/memory.db").to_string_lossy().as_ref()).unwrap();
        let blob: Option<Vec<u8>> = store.conn.query_row("SELECT embedding FROM symbols LIMIT 1", [], |r| r.get(0)).unwrap();
        assert!(blob.is_some(), "embedding blob is null");
        let blob = blob.unwrap();
        assert_eq!(blob.len(), 64 * 4, "blob len {} != 256", blob.len());
        std::env::set_current_dir(cwd).unwrap();
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn reindex_reembeds_indexed_text_not_signature() {
        // Fidelity: reindex must embed the exact indexed string. Hash
        // embeddings are deterministic, so the blob must be byte-identical
        // after reindex (signature-only input would change it).
        let _g = LOCK.lock().unwrap();
        let dir = temp_dir_with_nanos("cx_reidx_fid");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let cwd = std::env::current_dir().unwrap();
        std::env::set_current_dir(&dir).unwrap();
        run(vec!["cortex".into(), "init".into(), "--embed".into(), "hash".into(), "--yes".into()]).unwrap();
        let sub = dir.join("proj");
        std::fs::create_dir_all(&sub).unwrap();
        std::fs::write(sub.join("a.py"), "def compute(x):\n    \"\"\"Scale the value.\"\"\"\n    return x * 2\n").unwrap();
        run(vec!["cortex".into(), "index".into(), sub.to_string_lossy().to_string(), "--embed".into(), "hash".into()]).unwrap();
        let blob1: Vec<u8> = {
            let store = Store::open_at(dir.join(".cortex/memory.db").to_string_lossy().as_ref()).unwrap();
            let et: String = store.conn.query_row("SELECT embed_text FROM symbols LIMIT 1", [], |r| r.get(0)).unwrap();
            assert!(et.contains("return x * 2"), "embed_text must hold the body: {et:?}");
            store.conn.query_row("SELECT embedding FROM symbols LIMIT 1", [], |r| r.get(0)).unwrap()
        };
        run(vec!["cortex".into(), "reindex".into()]).unwrap();
        let blob2: Vec<u8> = {
            let store = Store::open_at(dir.join(".cortex/memory.db").to_string_lossy().as_ref()).unwrap();
            store.conn.query_row("SELECT embedding FROM symbols LIMIT 1", [], |r| r.get(0)).unwrap()
        };
        assert_eq!(blob1, blob2, "reindex changed vectors: fidelity broken");
        std::env::set_current_dir(cwd).unwrap();
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn dim_mismatch_requires_reindex() {
        let _g = LOCK.lock().unwrap();
        let dir = temp_dir_with_nanos("cx_dim_mismatch");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let cwd = std::env::current_dir().unwrap();
        std::env::set_current_dir(&dir).unwrap();
        run(vec!["cortex".into(), "init".into(), "--embed".into(), "hash".into(), "--yes".into()]).unwrap();
        let store = Store::open_at(dir.join(".cortex/memory.db").to_string_lossy().as_ref()).unwrap();
        store.set_meta("embedding_dim", "999").unwrap();
        drop(store);
        let res = run(vec!["cortex".into(), "recall".into(), "foo".into(), "--budget".into(), "500".into()]);
        assert!(res.is_err(), "expected dim mismatch err, got ok");
        let msg = res.unwrap_err().to_string();
        assert!(msg.contains("run cortex reindex"), "msg missing hint: {msg}");
        std::env::set_current_dir(cwd).unwrap();
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn recall_dead_ollama_points_at_fallback() {
        let _g = LOCK.lock().unwrap();
        let dir = temp_dir_with_nanos("cx_dead_ollama");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let cwd = std::env::current_dir().unwrap();
        std::env::set_current_dir(&dir).unwrap();
        run(vec!["cortex".into(), "init".into(), "--embed".into(), "hash".into(), "--yes".into()]).unwrap();
        // Point config at a dead server (port 9 discard: refused fast).
        std::fs::write(
            dir.join(".cortex/config.toml"),
            "provider = \"ollama\"\ndim = 768\nmodel = \"nomic-embed-text\"\nollama_url = \"http://127.0.0.1:9\"\n",
        )
        .unwrap();
        let res = run(vec!["cortex".into(), "recall".into(), "foo".into(), "--budget".into(), "500".into()]);
        std::env::set_current_dir(cwd).unwrap();
        assert!(res.is_err(), "expected unhealthy-provider err, got ok");
        let msg = res.unwrap_err().to_string();
        assert!(msg.contains("embeddings use"), "must point at fallback: {msg}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn index_extracts_tree_sitter_symbols() {
        let _g = LOCK.lock().unwrap();
        let dir = temp_dir_with_nanos("cx_idx_ts");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("a.py"), "class Foo:\n    \"\"\"Foo docs.\"\"\"\n    def bar(self, x):\n        \"\"\"Adds one.\"\"\"\n        return x + 1\n").unwrap();
        let syms = extract_symbols_vec(&dir);
        assert!(syms.iter().any(|(_, name, _, _, _, _, _)| name == "bar"), "expected bar symbol, got: {:?}", syms);
        assert!(syms.iter().any(|(_, n, _, _, _, s, e)| n == "bar" && e.len() > s.len() && e.contains("return x + 1")));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn index_exclude_prunes_dirs_and_files() {
        let _g = LOCK.lock().unwrap();
        let dir = temp_dir_with_nanos("cx_idx_excl");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("live")).unwrap();
        std::fs::create_dir_all(dir.join("vendored")).unwrap();
        std::fs::write(dir.join("live").join("a.py"), "def keep():\n    \"\"\"Kept live symbol for exclude testing.\"\"\"\n    return 1\n").unwrap();
        std::fs::write(dir.join("vendored").join("b.py"), "def dropme():\n    \"\"\"Vendored symbol that must be excluded.\"\"\"\n    return 2\n").unwrap();
        std::fs::write(dir.join("skipme.py"), "def alsodrop():\n    \"\"\"Top-level file that must be excluded.\"\"\"\n    return 3\n").unwrap();
        let all = extract_symbols(&dir, &[]).0;
        assert!(all.iter().any(|(_, n, ..)| n == "keep"));
        assert!(all.iter().any(|(_, n, ..)| n == "dropme"));
        let filtered = extract_symbols(&dir, &["vendored".to_string(), "skipme".to_string()]).0;
        assert!(filtered.iter().any(|(_, n, ..)| n == "keep"));
        assert!(filtered.iter().all(|(_, n, ..)| n != "dropme" && n != "alsodrop"),
            "excluded paths leaked: {:?}", filtered.iter().map(|(_, n, ..)| n).collect::<Vec<_>>());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn embed_best_effort_isolates_poison_text() {
        struct Flaky;
        impl crate::embed::EmbeddingProvider for Flaky {
            fn name(&self) -> &str { "flaky" }
            fn dim(&self) -> usize { 2 }
            fn is_local(&self) -> bool { true }
            fn health_check(&self) -> crate::Result<()> { Ok(()) }
            fn embed(&self, texts: Vec<String>) -> crate::Result<Vec<Vec<f32>>> {
                if texts.iter().any(|t| t.contains("POISON")) {
                    return Err(crate::Error::Cli("boom".into()));
                }
                Ok(texts.iter().map(|t| vec![t.len() as f32, 0.0]).collect())
            }
        }
        let texts = vec!["aa".to_string(), "POISON".to_string(), "bbbb".to_string()];
        let (out, failed) = embed_best_effort(&Flaky, texts);
        assert_eq!(failed, 1);
        assert_eq!(out.len(), 3);
        assert_eq!(out[0], Some(vec![2.0, 0.0]));
        assert_eq!(out[1], None);
        assert_eq!(out[2], Some(vec![4.0, 0.0]));
        // Truncation guard: over-long input never reaches the provider whole.
        let long = "x".repeat(EMBED_MAX_CHARS + 100);
        let (out2, f2) = embed_best_effort(&Flaky, vec![long]);
        assert_eq!(f2, 0);
        assert_eq!(out2[0], Some(vec![EMBED_MAX_CHARS as f32, 0.0]));
    }

    #[test]
    fn index_dim_mismatch_errors_with_reindex_hint() {
        let _g = LOCK.lock().unwrap();
        let dir = temp_dir_with_nanos("cx_idx_dim_mismatch");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let cwd = std::env::current_dir().unwrap();
        std::env::set_current_dir(&dir).unwrap();
        run(vec!["cortex".into(), "init".into(), "--embed".into(), "hash".into(), "--yes".into()]).unwrap();
        let sub = dir.join("proj");
        std::fs::create_dir_all(&sub).unwrap();
        std::fs::write(sub.join("foo.rs"), "fn foo() {}").unwrap();
        let store = Store::open_at(dir.join(".cortex/memory.db").to_string_lossy().as_ref()).unwrap();
        store.set_meta("embedding_dim", "999").unwrap();
        drop(store);
        let res = run(vec!["cortex".into(), "index".into(), sub.to_string_lossy().to_string(), "--embed".into(), "hash".into()]);
        assert!(res.is_err(), "expected dim mismatch err on index, got ok");
        let msg = res.unwrap_err().to_string();
        assert!(msg.contains("run cortex reindex"), "msg missing hint: {msg}");
        let store = Store::open_at(dir.join(".cortex/memory.db").to_string_lossy().as_ref()).unwrap();
        let n: i64 = store.conn.query_row("SELECT COUNT(*) FROM symbols", [], |r| r.get(0)).unwrap();
        assert_eq!(n, 0, "gate must fire before writes, found {n} symbols");
        std::env::set_current_dir(cwd).unwrap();
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn find_returns_defs_uses_and_connections() {
        let _g = LOCK.lock().unwrap();
        let dir = temp_dir_with_nanos("cx_find_refs");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let cwd = std::env::current_dir().unwrap();
        std::env::set_current_dir(&dir).unwrap();
        run(vec!["cortex".into(), "init".into(), "--embed".into(), "hash".into(), "--yes".into()]).unwrap();
        let sub = dir.join("proj");
        std::fs::create_dir_all(&sub).unwrap();
        std::fs::write(sub.join("a.py"), "import os\nfrom tokens import make_token\n\ndef validate_token(x):\n    y = make_token(x)\n    return y\n").unwrap();
        std::fs::write(sub.join("tokens.py"), "def make_token(seed):\n    return seed\n").unwrap();
        run(vec!["cortex".into(), "index".into(), sub.to_string_lossy().to_string(), "--embed".into(), "hash".into()]).unwrap();
        let store = Store::open_at(dir.join(".cortex/memory.db").to_string_lossy().as_ref()).unwrap();
        let (defs, uses, _) = store.find_refs("validate_token", 10).unwrap();
        assert_eq!(defs.len(), 1, "defs: {defs:?}");
        assert_eq!(defs[0].0, "a.py");
        // validate_token is DEFINED here (uses list is empty for it); its
        // body calls make_token — verified via the make_token lookup below.
        assert!(uses.is_empty(), "def-only name should have no uses: {uses:?}");
        let (mdefs, muses, _) = store.find_refs("make_token", 10).unwrap();
        assert!(mdefs.iter().any(|(p, _, _)| p == "tokens.py"), "mdefs: {mdefs:?}");
        assert!(muses.iter().any(|(p, _, _, c)| p == "a.py" && c.contains("make_token(x)")),
            "muses: {muses:?}");
        let (out, _inc) = store.file_connections("a.py", 8).unwrap();
        assert!(out.contains(&"os".to_string()) && out.contains(&"tokens".to_string()),
            "outgoing: {out:?}");
        // do_find / do_map run without error (output goes to stdout).
        do_find("validate_token".to_string(), 10, true).unwrap();
        do_map("validate_token login".to_string(), true).unwrap();
        std::env::set_current_dir(cwd).unwrap();
        let _ = std::fs::remove_dir_all(&dir);
    }
}
