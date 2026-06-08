//! Reads RAG configuration from environment variables loaded by `dotenvy`.
//! All variables have safe defaults so the server starts even without an API key
//! (RAG endpoints will return errors on call, not at startup).

use std::{env, path::PathBuf};

#[derive(Debug, Clone)]
pub struct RagConfig {
    /// `OPENAI_API_KEY` — empty string if not set (causes API calls to fail gracefully).
    pub api_key: String,
    /// `RAG_EMBED_MODEL` — default: `text-embedding-3-small`
    pub embed_model: String,
    /// `RAG_LLM_MODEL` — default: `gpt-4o-mini` (handles text + vision in one model)
    pub chat_model: String,
    /// `RAG_LLM_TEMPERATURE` — default: `0.2`
    pub temperature: f32,
    /// `RAG_DATA_DIR` — directory where uploaded files are saved
    pub data_dir: PathBuf,
    /// `RAG_TOP_K` — number of chunks to retrieve per query, default: `5`
    pub top_k: usize,
    /// `RAG_CHUNK_SIZE` — characters per chunk, default: `512`
    pub chunk_size: usize,
    /// `RAG_CHUNK_OVERLAP` — overlap characters between consecutive chunks, default: `64`
    pub chunk_overlap: usize,
}

impl RagConfig {
    pub fn from_env() -> Self {
        Self {
            api_key: env::var("OPENAI_API_KEY").unwrap_or_default(),
            embed_model: env::var("RAG_EMBED_MODEL")
                .unwrap_or_else(|_| "text-embedding-3-small".into()),
            chat_model: env::var("RAG_LLM_MODEL")
                .unwrap_or_else(|_| "gpt-4o-mini".into()),
            temperature: env::var("RAG_LLM_TEMPERATURE")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.2),
            data_dir: PathBuf::from(
                env::var("RAG_DATA_DIR").unwrap_or_else(|_| "./uploads".into()),
            ),
            top_k: env::var("RAG_TOP_K")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(5),
            chunk_size: env::var("RAG_CHUNK_SIZE")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(512),
            chunk_overlap: env::var("RAG_CHUNK_OVERLAP")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(64),
        }
    }
}
