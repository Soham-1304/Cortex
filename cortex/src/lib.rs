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
