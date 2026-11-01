use crate::embed::{cosine, embed as hash_embed, DIM as HASH_DIM};
use std::time::Duration;

pub trait Provider: Send + Sync {
    fn name(&self) -> &str;
    fn dim(&self) -> usize;
    fn embed_one(&self, text: &str) -> Result<Vec<f32>, String>;
    fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, String> {
        texts.iter().map(|t| self.embed_one(t)).collect()
    }
}

pub struct HashProvider;
impl Provider for HashProvider {
    fn name(&self) -> &str { "hash" }
    fn dim(&self) -> usize { HASH_DIM }
    fn embed_one(&self, text: &str) -> Result<Vec<f32>, String> { Ok(hash_embed(text)) }
}

pub struct OllamaProvider {
    pub url: String,
    pub model: String,
}
impl OllamaProvider {
    pub fn new(url: &str, model: &str) -> Self {
        Self { url: url.into(), model: model.into() }
    }
    pub fn health(&self) -> Result<(), String> {
        let agent = ureq::AgentBuilder::new().timeout(Duration::from_millis(800)).build();
        agent.get(&format!("{}/api/tags", self.url)).call().map_err(|e| e.to_string())?;
        Ok(())
    }
    fn truncate(s: &str) -> String {
        const MAX_CHARS: usize = 1800;
        if s.len() <= MAX_CHARS { return s.to_string(); }
        s.chars().take(MAX_CHARS).collect()
    }
}
impl Provider for OllamaProvider {
    fn name(&self) -> &str { "ollama/nomic-embed-text" }
    fn dim(&self) -> usize { 768 }
    fn embed_one(&self, text: &str) -> Result<Vec<f32>, String> {
        let agent = ureq::AgentBuilder::new().timeout(Duration::from_secs(8)).build();
        let prompt = Self::truncate(text);
        let resp: serde_json::Value = agent
            .post(&format!("{}/api/embeddings", self.url))
            .send_json(ureq::json!({"model": &self.model, "prompt": prompt}))
            .map_err(|e| e.to_string())?
            .into_json().map_err(|e| e.to_string())?;
        let arr = resp.get("embedding").and_then(|v| v.as_array()).ok_or("no embedding field")?;
        let mut v: Vec<f32> = arr.iter().filter_map(|x| x.as_f64().map(|f| f as f32)).collect();
        let n = v.iter().map(|x| x*x).sum::<f32>().sqrt().max(1e-9);
        v.iter_mut().for_each(|x| *x /= n);
        Ok(v)
    }
    fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, String> {
        if texts.is_empty() { return Ok(vec![]); }
        eprintln!("[bench] ollama embedding {} texts (768d, truncated 1800ch, 4 threads) ...", texts.len());
        let n_threads = 4;
        let chunk = texts.len().div_ceil(n_threads);
        let mut handles = Vec::new();
        for (chunk_idx, chunk_texts) in texts.chunks(chunk).enumerate() {
            let chunk_vec: Vec<String> = chunk_texts.to_vec();
            let url = self.url.clone();
            let model = self.model.clone();
            handles.push(std::thread::spawn(move || {
                let prov = OllamaProvider { url, model };
                let mut out = Vec::with_capacity(chunk_vec.len());
                for t in &chunk_vec {
                    match prov.embed_one(t) {
                        Ok(v) => out.push(v),
                        Err(e) => {
                            eprintln!("[bench] warn: ollama failed on chunk {} ({}), using hash", chunk_idx, e);
                            out.push(crate::embed::embed(t));
                        }
                    }
                }
                out
            }));
        }
        let mut out = Vec::with_capacity(texts.len());
        for h in handles {
            out.extend(h.join().map_err(|_| "thread panicked".to_string())?);
        }
        Ok(out)
    }
}

pub struct OpenAIProvider {
    pub api_key: String,
    pub model: String,
}
impl Provider for OpenAIProvider {
    fn name(&self) -> &str { "openai/text-embedding-3-large" }
    fn dim(&self) -> usize { 3072 }
    fn embed_one(&self, text: &str) -> Result<Vec<f32>, String> {
        let agent = ureq::AgentBuilder::new().timeout(Duration::from_secs(12)).build();
        let body = serde_json::json!({"model": &self.model, "input": text});
        let resp: serde_json::Value = agent
            .post("https://api.openai.com/v1/embeddings")
            .set("Authorization", &format!("Bearer {}", self.api_key))
            .send_json(body).map_err(|e| e.to_string())?
            .into_json().map_err(|e| e.to_string())?;
        let arr = resp["data"][0]["embedding"].as_array().ok_or("no embedding")?;
        let mut v: Vec<f32> = arr.iter().filter_map(|x| x.as_f64().map(|f| f as f32)).collect();
        let n = v.iter().map(|x| x*x).sum::<f32>().sqrt().max(1e-9);
        v.iter_mut().for_each(|x| *x /= n);
        Ok(v)
    }
    fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, String> {
        if texts.is_empty() { return Ok(vec![]); }
        const CHUNK: usize = 96;
        let mut out = Vec::with_capacity(texts.len());
        for chunk in texts.chunks(CHUNK) {
            let agent = ureq::AgentBuilder::new().timeout(Duration::from_secs(30)).build();
            let body = serde_json::json!({"model": &self.model, "input": chunk});
            let resp: serde_json::Value = agent
                .post("https://api.openai.com/v1/embeddings")
                .set("Authorization", &format!("Bearer {}", self.api_key))
                .send_json(body).map_err(|e| e.to_string())?
                .into_json().map_err(|e| e.to_string())?;
            let data = resp["data"].as_array().ok_or("no data")?;
            for item in data {
                let arr = item["embedding"].as_array().ok_or("no embedding")?;
                let mut v: Vec<f32> = arr.iter().filter_map(|x| x.as_f64().map(|f| f as f32)).collect();
                let n = v.iter().map(|x| x*x).sum::<f32>().sqrt().max(1e-9);
                v.iter_mut().for_each(|x| *x /= n);
                out.push(v);
            }
        }
        Ok(out)
    }
}

pub struct GoogleProvider {
    pub api_key: String,
}
impl Provider for GoogleProvider {
    fn name(&self) -> &str { "google/text-embedding-004" }
    fn dim(&self) -> usize { 768 }
    fn embed_one(&self, text: &str) -> Result<Vec<f32>, String> {
        self.embed_batch(&[text.to_string()]).map(|mut v| v.remove(0))
    }
    fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, String> {
        if texts.is_empty() { return Ok(vec![]); }
        const CHUNK: usize = 96;
        let mut out = Vec::with_capacity(texts.len());
        for chunk in texts.chunks(CHUNK) {
            let requests: Vec<serde_json::Value> = chunk.iter().map(|t| serde_json::json!({"model":"models/text-embedding-004","content":{"parts":[{"text": t}]}})).collect();
            let body = serde_json::json!({"requests": requests});
            let url = format!("https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:batchEmbedContents?key={}", self.api_key);
            let agent = ureq::AgentBuilder::new().timeout(Duration::from_secs(30)).build();
            let resp: serde_json::Value = agent.post(&url).send_json(body).map_err(|e| e.to_string())?.into_json().map_err(|e| e.to_string())?;
            let embs = resp["embeddings"].as_array().ok_or("no embeddings field")?;
            for item in embs {
                let arr = item["values"].as_array().ok_or("no values")?;
                let mut v: Vec<f32> = arr.iter().filter_map(|x| x.as_f64().map(|f| f as f32)).collect();
                let n = v.iter().map(|x| x*x).sum::<f32>().sqrt().max(1e-9);
                v.iter_mut().for_each(|x| *x /= n);
                out.push(v);
            }
        }
        Ok(out)
    }
}

pub fn pick_provider(flag: &str) -> Box<dyn Provider> {
    match flag {
        "hash" => Box::new(HashProvider),
        "ollama" => Box::new(OllamaProvider::new("http://localhost:11434", "nomic-embed-text")),
        "google" => {
            let key = std::env::var("GOOGLE_API_KEY").unwrap_or_default();
            if key.is_empty() { eprintln!("GOOGLE_API_KEY not set, falling back to hash"); Box::new(HashProvider) }
            else { Box::new(GoogleProvider{ api_key: key }) }
        },
        "openai" => {
            let key = std::env::var("OPENAI_API_KEY").unwrap_or_default();
            if key.is_empty() { eprintln!("OPENAI_API_KEY not set, falling back to hash"); Box::new(HashProvider) }
            else { Box::new(OpenAIProvider{ api_key: key, model: "text-embedding-3-large".into() }) }
        },
        _ => {
            let oll = OllamaProvider::new("http://localhost:11434", "nomic-embed-text");
            if oll.health().is_ok() {
                eprintln!("[bench] using ollama/nomic-embed-text 768d");
                Box::new(oll)
            } else if std::env::var("OPENAI_API_KEY").map(|k| !k.is_empty()).unwrap_or(false) {
                eprintln!("[bench] ollama down, using openai 3072d");
                Box::new(OpenAIProvider{ api_key: std::env::var("OPENAI_API_KEY").unwrap(), model: "text-embedding-3-large".into() })
            } else {
                eprintln!("[bench] no real embed available, using hash 64d (fake)");
                Box::new(HashProvider)
            }
        }
    }
}

pub fn cosine_generic(a: &[f32], b: &[f32]) -> f32 {
    cosine(a, b)
}
