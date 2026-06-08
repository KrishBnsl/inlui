//! RAG (Retrieval-Augmented Generation) pipeline — integrated into the sim_engine server.
//!
//! Sub-modules follow the same pattern as `predictor/`:
//!
//! - [`config`]       — typed env-var configuration
//! - [`vector_store`] — in-memory cosine-similarity chunk store
//! - [`embeddings`]   — OpenAI text-embedding-3-small client
//! - [`ingestion`]    — PDF text extraction, image OCR via vision, chunking
//! - [`prompts`]      — system / user prompt templates
//! - [`retrieval`]    — top-k lookup + GPT-4o-mini answer generation

pub mod config;
pub mod embeddings;
pub mod ingestion;
pub mod prompts;
pub mod retrieval;
pub mod vector_store;

use std::sync::Arc;

use async_openai::{config::OpenAIConfig, Client};
use tokio::sync::RwLock;

use config::RagConfig;
use vector_store::VectorStore;

// ── Service handle ─────────────────────────────────────────────────────────────

/// Central handle shared via `Arc` on `AppState`.
/// Holds the OpenAI client, the in-memory vector store, and all configuration.
pub struct RagService {
    pub config: RagConfig,
    /// Thread-safe in-memory vector store (no disk persistence — resets on restart).
    pub store: Arc<RwLock<VectorStore>>,
    /// Async OpenAI client (embeddings + chat completions).
    pub openai: Client<OpenAIConfig>,
}

impl RagService {
    pub fn new(config: RagConfig) -> Self {
        let oa_config = OpenAIConfig::new().with_api_key(&config.api_key);
        Self {
            openai: Client::with_config(oa_config),
            store: Arc::new(RwLock::new(VectorStore::new())),
            config,
        }
    }
}
