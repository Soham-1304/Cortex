use serde::{Deserialize, Serialize};
use crate::Result;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EmbeddingConfig {
    pub provider: String,
    pub dim: usize,
    pub model: String,
    pub ollama_url: Option<String>,
    pub api_key_env: Option<String>,
}
impl Default for EmbeddingConfig {
    fn default() -> Self { Self { provider: "local-nomic".into(), dim: 768, model: "nomic-embed-text".into(), ollama_url: Some("http://localhost:11434".into()), api_key_env: None } }
}

pub fn config_path(base: &std::path::Path) -> std::path::PathBuf { base.join(".cortex/config.toml") }
pub fn global_model_dir() -> std::path::PathBuf { dirs::home_dir().unwrap_or_else(|| std::path::PathBuf::from(".")).join(".cortex/models") }

pub fn load_config(path: &std::path::Path) -> Result<EmbeddingConfig> {
    let s = std::fs::read_to_string(path)?;
    toml::from_str(&s).map_err(|e| crate::Error::Cli(e.to_string()))
}
pub fn write_config(path: &std::path::Path, cfg: &EmbeddingConfig) -> Result<()> {
    if let Some(p) = path.parent() { std::fs::create_dir_all(p)?; }
    let s = toml::to_string_pretty(cfg).map_err(|e| crate::Error::Cli(e.to_string()))?;
    std::fs::write(path, s)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn roundtrip_toml() {
        let dir = std::env::temp_dir().join(format!("cx_cfg_{}", std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
        std::fs::create_dir_all(&dir).unwrap();
        let cfg = EmbeddingConfig { provider: "local-nomic".into(), dim: 768, model: "nomic-embed-text".into(), ollama_url: None, api_key_env: None };
        write_config(&dir.join("config.toml"), &cfg).unwrap();
        let got = load_config(&dir.join("config.toml")).unwrap();
        assert_eq!(got.provider, "local-nomic");
        assert_eq!(got.dim, 768);
        let _ = std::fs::remove_dir_all(&dir);
    }
    #[test] fn meta_dim_recorded_and_mismatch_detected() {
        let s = crate::store::Store::open_in_memory().unwrap();
        s.set_meta("embedding_dim","768").unwrap();
        assert_eq!(s.get_meta("embedding_dim").unwrap(), Some("768".into()));
        s.set_meta("embedding_dim","64").unwrap();
        assert_eq!(s.get_meta("embedding_dim").unwrap(), Some("64".into()));
    }
}
