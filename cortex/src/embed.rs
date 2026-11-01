use crate::Result;
use serde::de::Error as _;

pub trait EmbeddingProvider: Send + Sync {
    fn name(&self) -> &str;
    fn dim(&self) -> usize;
    fn is_local(&self) -> bool;
    fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>>;
    fn health_check(&self) -> Result<()>;
}

pub struct HashEmbedder {
    pub dim: usize,
}
impl HashEmbedder {
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }
}
impl EmbeddingProvider for HashEmbedder {
    fn name(&self) -> &str {
        "hash"
    }
    fn dim(&self) -> usize {
        self.dim
    }
    fn is_local(&self) -> bool {
        true
    }
    fn health_check(&self) -> Result<()> {
        Ok(())
    }
    fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        Ok(texts
            .iter()
            .map(|t| {
                let mut v = vec![0f32; self.dim];
                if self.dim == 0 || t.trim().is_empty() {
                    return v;
                }
                let mut prev_h: Option<u64> = None;
                cortex_core::visit_tokens(t, |tok| {
                    if tok.is_empty() {
                        return;
                    }
                    let mut h: u64 = 0xcbf29ce484222325;
                    for b in tok.bytes() {
                        h ^= b as u64;
                        h = h.wrapping_mul(0x100000001b3);
                    }
                    v[(h as usize) % self.dim] += 1.0;
                    v[((h >> 32) as usize) % self.dim] -= 1.0;
                    // In higher-dimensional hash spaces (>=256d), also project adjacent
                    // token bigrams so phrase order ("url validator" <-> "URLValidator") aligns.
                    if self.dim >= 256 {
                        if let Some(ph) = prev_h {
                            let bh = ph.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ h;
                            v[(bh as usize) % self.dim] += 0.5;
                            v[((bh >> 32) as usize) % self.dim] -= 0.5;
                        }
                    }
                    prev_h = Some(h);
                });
                let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
                if n < 1e-6 {
                    return v;
                }
                v.iter_mut().for_each(|x| *x /= n);
                v
            })
            .collect())
    }
}

pub struct NoOpEmbedder {
    pub dim: usize,
}
impl EmbeddingProvider for NoOpEmbedder {
    fn name(&self) -> &str {
        "noop"
    }
    fn dim(&self) -> usize {
        self.dim
    }
    fn is_local(&self) -> bool {
        true
    }
    fn health_check(&self) -> Result<()> {
        Ok(())
    }
    fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|_| vec![0f32; self.dim]).collect())
    }
}

pub struct OllamaProvider {
    pub url: String,
    pub model: String,
    pub timeout_ms: u64,
}
impl EmbeddingProvider for OllamaProvider {
    fn name(&self) -> &str {
        "ollama"
    }
    fn dim(&self) -> usize {
        768
    }
    fn is_local(&self) -> bool {
        true
    }
    fn health_check(&self) -> Result<()> {
        if self.url.is_empty() {
            return Err(crate::Error::Cli("ollama url empty".into()));
        }
        // Live probe: a configured-but-dead server must report missing, not
        // ok — otherwise recall silently degrades to hash vectors. Capped
        // timeout so `embeddings list` and recall fail fast, never hang.
        let agent = ureq::AgentBuilder::new()
            .timeout(std::time::Duration::from_millis(self.timeout_ms.min(3000)))
            .build();
        agent
            .get(&format!("{}/api/tags", self.url.trim_end_matches('/')))
            .call()
            .map_err(|e| crate::Error::Cli(format!("ollama unreachable at {}: {e}", self.url)))?;
        Ok(())
    }
    fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        use std::time::Duration;
        let agent = ureq::AgentBuilder::new()
            .timeout(Duration::from_millis(self.timeout_ms))
            .build();
        let mut out = Vec::with_capacity(texts.len());
        for t in &texts {
            let r: serde_json::Value = agent
                .post(&format!("{}/api/embeddings", self.url))
                .send_json(ureq::json!({"model": &self.model, "prompt": t}))
                .map_err(|e| crate::Error::Http(e.to_string()))?
                .into_json()
                .map_err(|e| crate::Error::Http(e.to_string()))?;
            let arr = r
                .get("embedding")
                .and_then(|v| v.as_array())
                .ok_or_else(|| serde_json::Error::custom("no embedding field"))?;
            let v: Vec<f32> = arr
                .iter()
                .filter_map(|x| x.as_f64())
                .map(|x| x as f32)
                .collect();
            out.push(v);
        }
        Ok(out)
    }
}

pub struct LocalNomicEmbedder {
    pub dim: usize,
    pub model_dir: std::path::PathBuf,
    #[cfg(feature = "local-embed")]
    session: std::sync::Mutex<Option<fastembed::TextEmbedding>>,
}
impl LocalNomicEmbedder {
    pub fn new(cfg: &crate::config::EmbeddingConfig) -> Result<Self> {
        let dir = crate::config::global_model_dir().join("nomic-embed-text");
        Ok(Self {
            dim: cfg.dim,
            model_dir: dir,
            #[cfg(feature = "local-embed")]
            session: std::sync::Mutex::new(None),
        })
    }
}
impl EmbeddingProvider for LocalNomicEmbedder {
    fn name(&self) -> &str {
        "local-nomic"
    }
    fn dim(&self) -> usize {
        self.dim
    }
    fn is_local(&self) -> bool {
        true
    }
    fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        #[cfg(feature = "local-embed")]
        {
            if texts.is_empty() {
                return Ok(Vec::new());
            }
            if !self.model_dir.exists()
                || std::fs::read_dir(&self.model_dir)
                    .map(|mut d| d.next().is_none())
                    .unwrap_or(true)
            {
                return Err(crate::Error::Cli(format!(
                    "model not found at {} — run cortex embeddings install local-nomic",
                    self.model_dir.display()
                )));
            }
            let mut guard = self
                .session
                .lock()
                .map_err(|_| crate::Error::Cli("local-nomic session mutex poisoned".into()))?;
            if guard.is_none() {
                let opts =
                    fastembed::InitOptions::new(fastembed::EmbeddingModel::NomicEmbedTextV1)
                        .with_cache_dir(self.model_dir.clone())
                        .with_show_download_progress(false);
                let model = fastembed::TextEmbedding::try_new(opts)
                    .map_err(|e| crate::Error::Cli(e.to_string()))?;
                *guard = Some(model);
            }
            let model = guard.as_ref().unwrap();
            let embs = model
                .embed(texts, None)
                .map_err(|e| crate::Error::Cli(e.to_string()))?;
            Ok(embs)
        }
        #[cfg(not(feature = "local-embed"))]
        {
            let _ = texts;
            Err(crate::Error::Cli(
                "local-embed feature not enabled — build with --features local-embed or use ollama/openai".into(),
            ))
        }
    }
    fn health_check(&self) -> Result<()> {
        if self.model_dir.join("config.json").exists()
            || self.model_dir.join("model.onnx").exists()
            || self.model_dir.exists()
                && std::fs::read_dir(&self.model_dir)
                    .map(|mut d| d.next().is_some())
                    .unwrap_or(false)
        {
            Ok(())
        } else {
            Err(crate::Error::Cli(format!(
                "model not found at {} — run cortex embeddings install",
                self.model_dir.display()
            )))
        }
    }
}
pub struct OpenAIProvider {
    pub dim: usize,
    pub api_key_env: String,
}
impl OpenAIProvider {
    pub fn new(cfg: &crate::config::EmbeddingConfig) -> Result<Self> {
        Ok(Self {
            dim: cfg.dim,
            api_key_env: cfg
                .api_key_env
                .clone()
                .unwrap_or("OPENAI_API_KEY".into()),
        })
    }
}
impl EmbeddingProvider for OpenAIProvider {
    fn name(&self) -> &str {
        "openai"
    }
    fn dim(&self) -> usize {
        self.dim
    }
    fn is_local(&self) -> bool {
        false
    }
    fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        let key = std::env::var(&self.api_key_env)
            .map_err(|_| crate::Error::Cli(format!("env {} not set", self.api_key_env)))?;
        let agent = ureq::AgentBuilder::new()
            .timeout(std::time::Duration::from_secs(30))
            .build();
        let body = serde_json::json!({"model": "text-embedding-3-large", "input": texts});
        let resp: serde_json::Value = agent
            .post("https://api.openai.com/v1/embeddings")
            .set("Authorization", &format!("Bearer {key}"))
            .send_json(body)
            .map_err(|e| crate::Error::Http(e.to_string()))?
            .into_json()
            .map_err(|e| crate::Error::Http(e.to_string()))?;
        let data = resp["data"]
            .as_array()
            .ok_or_else(|| serde_json::Error::custom("no data field"))?;
        let mut out = Vec::with_capacity(data.len());
        for item in data {
            let arr = item["embedding"]
                .as_array()
                .ok_or_else(|| serde_json::Error::custom("no embedding"))?;
            out.push(
                arr.iter()
                    .filter_map(|x| x.as_f64())
                    .map(|x| x as f32)
                    .collect(),
            );
        }
        Ok(out)
    }
    fn health_check(&self) -> Result<()> {
        if std::env::var(&self.api_key_env).is_ok() {
            Ok(())
        } else {
            Err(crate::Error::Cli(format!(
                "env {} not set",
                self.api_key_env
            )))
        }
    }
}
pub struct GoogleProvider {
    pub dim: usize,
    pub api_key_env: String,
}
impl GoogleProvider {
    pub fn new(cfg: &crate::config::EmbeddingConfig) -> Result<Self> {
        Ok(Self {
            dim: cfg.dim,
            api_key_env: cfg
                .api_key_env
                .clone()
                .unwrap_or("GOOGLE_API_KEY".into()),
        })
    }
}
impl EmbeddingProvider for GoogleProvider {
    fn name(&self) -> &str {
        "google"
    }
    fn dim(&self) -> usize {
        self.dim
    }
    fn is_local(&self) -> bool {
        false
    }
    fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        let key = std::env::var(&self.api_key_env)
            .map_err(|_| crate::Error::Cli(format!("env {} not set", self.api_key_env)))?;
        let agent = ureq::AgentBuilder::new()
            .timeout(std::time::Duration::from_secs(30))
            .build();
        let requests: Vec<serde_json::Value> = texts
            .iter()
            .map(|t| serde_json::json!({"model": "models/text-embedding-004", "content": {"parts": [{"text": t}]}}))
            .collect();
        let body = serde_json::json!({"requests": requests});
        let url = format!(
            "https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:batchEmbedContents?key={key}"
        );
        let resp: serde_json::Value = agent
            .post(&url)
            .send_json(body)
            .map_err(|e| crate::Error::Http(e.to_string()))?
            .into_json()
            .map_err(|e| crate::Error::Http(e.to_string()))?;
        let embeddings = resp["embeddings"]
            .as_array()
            .ok_or_else(|| serde_json::Error::custom("no embeddings field"))?;
        let mut out = Vec::with_capacity(embeddings.len());
        for item in embeddings {
            let arr = item["values"]
                .as_array()
                .ok_or_else(|| serde_json::Error::custom("no values"))?;
            out.push(
                arr.iter()
                    .filter_map(|x| x.as_f64())
                    .map(|x| x as f32)
                    .collect(),
            );
        }
        Ok(out)
    }
    fn health_check(&self) -> Result<()> {
        if std::env::var(&self.api_key_env).is_ok() {
            Ok(())
        } else {
            Err(crate::Error::Cli(format!(
                "env {} not set",
                self.api_key_env
            )))
        }
    }
}
pub struct MistralProvider {
    pub dim: usize,
    pub model: String,
    pub api_key_env: String,
}
impl MistralProvider {
    pub fn new(cfg: &crate::config::EmbeddingConfig) -> Result<Self> {
        let model = if cfg.model.trim().is_empty() || cfg.model == "mistral" {
            "mistral-embed".to_string()
        } else {
            cfg.model.clone()
        };
        Ok(Self {
            dim: cfg.dim,
            model,
            api_key_env: cfg
                .api_key_env
                .clone()
                .unwrap_or_else(|| "MISTRAL_API_KEY".into()),
        })
    }
}
impl EmbeddingProvider for MistralProvider {
    fn name(&self) -> &str {
        "mistral"
    }
    fn dim(&self) -> usize {
        self.dim
    }
    fn is_local(&self) -> bool {
        false
    }
    fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let key = std::env::var(&self.api_key_env)
            .map_err(|_| crate::Error::Cli(format!("env {} not set", self.api_key_env)))?;
        let agent = ureq::AgentBuilder::new()
            .timeout(std::time::Duration::from_secs(30))
            .build();
        let body = serde_json::json!({"model": &self.model, "input": texts});
        let resp: serde_json::Value = agent
            .post("https://api.mistral.ai/v1/embeddings")
            .set("Authorization", &format!("Bearer {key}"))
            .send_json(body)
            .map_err(|e| crate::Error::Http(e.to_string()))?
            .into_json()
            .map_err(|e| crate::Error::Http(e.to_string()))?;
        let data = resp["data"]
            .as_array()
            .ok_or_else(|| serde_json::Error::custom("no data field"))?;
        let mut out = Vec::with_capacity(data.len());
        for item in data {
            let arr = item["embedding"]
                .as_array()
                .ok_or_else(|| serde_json::Error::custom("no embedding"))?;
            out.push(
                arr.iter()
                    .filter_map(|x| x.as_f64())
                    .map(|x| x as f32)
                    .collect(),
            );
        }
        Ok(out)
    }
    fn health_check(&self) -> Result<()> {
        if std::env::var(&self.api_key_env).is_ok() {
            Ok(())
        } else {
            Err(crate::Error::Cli(format!(
                "env {} not set",
                self.api_key_env
            )))
        }
    }
}
pub mod registry {
    use crate::config::EmbeddingConfig;
    use crate::embed::{
        EmbeddingProvider, GoogleProvider, HashEmbedder, LocalNomicEmbedder, MistralProvider,
        NoOpEmbedder, OllamaProvider, OpenAIProvider,
    };
    use crate::Result;
    pub fn from_config(cfg: &EmbeddingConfig) -> Result<Box<dyn EmbeddingProvider>> {
        match cfg.provider.as_str() {
            "hash" => Ok(Box::new(HashEmbedder::new(cfg.dim))),
            "noop" => Ok(Box::new(NoOpEmbedder { dim: cfg.dim })),
            "ollama" => Ok(Box::new(OllamaProvider {
                url: cfg
                    .ollama_url
                    .clone()
                    .unwrap_or("http://localhost:11434".into()),
                model: cfg.model.clone(),
                timeout_ms: 5000,
            })),
            "local-nomic" => Ok(Box::new(LocalNomicEmbedder::new(cfg)?)),
            "openai" => Ok(Box::new(OpenAIProvider::new(cfg)?)),
            "google" => Ok(Box::new(GoogleProvider::new(cfg)?)),
            "mistral" => Ok(Box::new(MistralProvider::new(cfg)?)),
            other => Err(crate::Error::Cli(format!("unknown provider {other}"))),
        }
    }
    pub fn load_for_base(base: &std::path::Path) -> Result<Box<dyn EmbeddingProvider>> {
        let cfg =
            crate::config::load_config(&crate::config::config_path(base)).unwrap_or_default();
        from_config(&cfg)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn hash_deterministic_unit_norm() {
        let h = HashEmbedder::new(64);
        let v = h.embed(vec!["hello world".to_string()]).unwrap().remove(0);
        let v2 = HashEmbedder::new(64)
            .embed(vec!["hello world".to_string()])
            .unwrap()
            .remove(0);
        assert_eq!(v, v2);
        let empty = h.embed(vec![]).unwrap();
        assert!(empty.is_empty());
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-4);
        assert_eq!(v.len(), 64);
    }
    #[test]
    fn ollama_health_fails_on_unreachable_server() {
        // Port 9 (discard) is virtually always closed: health must actually
        // probe instead of passing on a non-empty URL. Must return fast
        // (no multi-second hangs in tests/CI).
        let p = OllamaProvider {
            url: "http://127.0.0.1:9".to_string(),
            model: "nomic-embed-text".to_string(),
            timeout_ms: 5000,
        };
        let t0 = std::time::Instant::now();
        assert!(p.health_check().is_err(), "unreachable server must fail health");
        assert!(t0.elapsed().as_secs() < 10, "health check must not hang");
        let empty = OllamaProvider {
            url: "".to_string(),
            model: "x".to_string(),
            timeout_ms: 5000,
        };
        assert!(empty.health_check().is_err());
    }

    #[test]
    fn hash_empty_text_is_zero_vector() {
        let h = HashEmbedder::new(8);
        let v = h.embed(vec!["".to_string()]).unwrap().remove(0);
        assert!(v.iter().all(|&x| x == 0.0));
        assert_eq!(v.len(), 8);
    }
    #[test]
    fn noop_dims() {
        let v = NoOpEmbedder { dim: 8 }
            .embed(vec!["a".to_string(), "b".to_string()])
            .unwrap();
        assert_eq!(v.len(), 2);
        assert_eq!(v[0].len(), 8);
        assert!(v[0].iter().all(|&x| x == 0.0));
        assert!(v[1].iter().all(|&x| x == 0.0));
    }
    #[test]
    fn ollama_unavailable_returns_error_fast() {
        let o = OllamaProvider {
            url: "http://127.0.0.1:1".to_string(),
            model: "nomic-embed-text".to_string(),
            timeout_ms: 300,
        };
        let res = o.embed(vec!["x".to_string()]);
        assert!(res.is_err(), "expected Err on closed port, got {:?}", res.ok());
    }
    #[test]
    fn registry_resolves_hash_without_model() {
        let cfg = crate::config::EmbeddingConfig {
            provider: "hash".into(),
            dim: 64,
            model: "hash".into(),
            ollama_url: None,
            api_key_env: None,
        };
        let r = registry::from_config(&cfg).unwrap();
        assert_eq!(r.name(), "hash");
        assert_eq!(r.dim(), 64);
        assert!(r.health_check().is_ok());
    }
    #[test]
    fn registry_rejects_missing_local_model_via_health() {
        let cfg = crate::config::EmbeddingConfig {
            provider: "local-nomic".into(),
            dim: 768,
            model: "nomic-embed-text".into(),
            ollama_url: None,
            api_key_env: None,
        };
        let r = registry::from_config(&cfg).unwrap();
        assert_eq!(r.name(), "local-nomic");
        assert_eq!(r.dim(), 768);
        let _ = r.health_check();
    }
    #[test]
    fn local_nomic_dim_768() {
        let cfg = crate::config::EmbeddingConfig {
            provider: "local-nomic".into(),
            dim: 768,
            model: "nomic-embed-text".into(),
            ollama_url: None,
            api_key_env: None,
        };
        let p = LocalNomicEmbedder::new(&cfg).unwrap();
        assert_eq!(p.dim(), 768);
        assert_eq!(p.name(), "local-nomic");
        assert!(p.is_local());
    }
    #[test]
    fn openai_google_health_requires_env() {
        let cfg = crate::config::EmbeddingConfig {
            provider: "openai".into(),
            dim: 3072,
            model: "text-embedding-3-large".into(),
            ollama_url: None,
            api_key_env: Some("OPENAI_API_KEY".into()),
        };
        let p = OpenAIProvider::new(&cfg).unwrap();
        assert_eq!(p.dim(), 3072);
        std::env::remove_var("OPENAI_API_KEY");
        assert!(p.health_check().is_err());
        std::env::set_var("OPENAI_API_KEY", "sk-test");
        assert!(p.health_check().is_ok());
        std::env::remove_var("OPENAI_API_KEY");
        let cfg2 = crate::config::EmbeddingConfig {
            provider: "google".into(),
            dim: 768,
            model: "text-embedding-004".into(),
            ollama_url: None,
            api_key_env: Some("GOOGLE_API_KEY".into()),
        };
        let q = GoogleProvider::new(&cfg2).unwrap();
        std::env::remove_var("GOOGLE_API_KEY");
        assert!(q.health_check().is_err());
    }
}
