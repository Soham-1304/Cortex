pub mod agentic;
pub mod agentic_llm;
pub mod cli;
pub mod mcp;
pub mod compile;
pub mod corefile;
pub mod embed;
pub mod graph;
pub mod session;
pub mod config;
pub mod store;
pub mod events;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("db: {0}")]
    Db(#[from] rusqlite::Error),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
    #[error("http: {0}")]
    Http(String),
    #[error("cli: {0}")]
    Cli(String),
}
pub type Result<T> = std::result::Result<T, Error>;

/// Unix seconds (std-only, saturates to 0 before the epoch — never fails,
/// so report tagging can't break a harness run).
pub fn unix_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}
