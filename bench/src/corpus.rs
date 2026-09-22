use serde::Deserialize;
use std::path::Path;

#[derive(Deserialize, Clone)]
pub struct Doc {
    pub id: String,
    pub path: String,
    pub text: String,
    #[serde(default)]
    pub line_start: usize,
    #[serde(default)]
    pub line_end: usize,
}

#[derive(Deserialize, Clone)]
pub struct Query {
    pub id: String,
    pub text: String,
    pub gold: Vec<String>,
}

#[derive(Deserialize)]
pub struct Corpus {
    pub docs: Vec<Doc>,
    pub queries: Vec<Query>,
    #[serde(default)]
    pub _empty_tasks: usize,
}

impl Corpus {
    pub fn load(path: &Path) -> Result<Corpus, String> {
        let raw = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
        serde_json::from_str(&raw).map_err(|e| e.to_string())
    }
}
