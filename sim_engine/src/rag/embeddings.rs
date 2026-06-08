//! OpenAI text-embedding-3-small client.
//! Batches all texts in a single API call; results are re-sorted by index
//! to guarantee alignment with the input slice.

use async_openai::{config::OpenAIConfig, Client};
use async_openai::types::CreateEmbeddingRequestArgs;

use super::config::RagConfig;

/// Embed one or more texts. Returns a `Vec<Vec<f32>>` in the same order as `texts`.
pub async fn embed_texts(
    client: &Client<OpenAIConfig>,
    config: &RagConfig,
    texts: Vec<String>,
) -> Result<Vec<Vec<f32>>, Box<dyn std::error::Error + Send + Sync>> {
    if texts.is_empty() {
        return Ok(vec![]);
    }

    let request = CreateEmbeddingRequestArgs::default()
        .model(&config.embed_model)
        .input(texts)
        .build()?;

    let response = client.embeddings().create(request).await?;

    // The API may return embeddings out of order — sort by index to be safe.
    let mut indexed: Vec<(usize, Vec<f32>)> = response
        .data
        .into_iter()
        .map(|e| (e.index as usize, e.embedding))
        .collect();
    indexed.sort_by_key(|(i, _)| *i);

    Ok(indexed.into_iter().map(|(_, emb)| emb).collect())
}
