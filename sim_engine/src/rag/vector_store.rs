//! In-memory vector store with cosine-similarity search.
//! No disk persistence — the store resets when the server restarts.
//! Wrapped in `Arc<RwLock<VectorStore>>` on `RagService` for thread safety.

use std::collections::HashSet;

use uuid::Uuid;

// ── Data types ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct Chunk {
    pub id: Uuid,
    /// Original filename that produced this chunk.
    pub source: String,
    /// Raw text content of the chunk.
    pub text: String,
    /// Embedding vector from text-embedding-3-small (1536 dims).
    pub embedding: Vec<f32>,
}

// ── Store ──────────────────────────────────────────────────────────────────────

#[derive(Debug, Default)]
pub struct VectorStore {
    chunks: Vec<Chunk>,
}

impl VectorStore {
    pub fn new() -> Self {
        Self { chunks: Vec::new() }
    }

    /// Append chunks to the store.
    pub fn insert(&mut self, new_chunks: Vec<Chunk>) {
        self.chunks.extend(new_chunks);
    }

    /// Return the `top_k` most similar chunks to `query_embedding`.
    pub fn search(&self, query_embedding: &[f32], top_k: usize) -> Vec<&Chunk> {
        if self.chunks.is_empty() {
            return vec![];
        }

        let mut scored: Vec<(f32, &Chunk)> = self
            .chunks
            .iter()
            .map(|c| (cosine_similarity(query_embedding, &c.embedding), c))
            .collect();

        // Sort descending by similarity
        scored.sort_by(|a, b| {
            b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal)
        });

        scored.into_iter().take(top_k).map(|(_, c)| c).collect()
    }

    pub fn chunk_count(&self) -> usize {
        self.chunks.len()
    }

    /// Unique document names currently indexed.
    pub fn documents(&self) -> Vec<String> {
        let mut docs: Vec<String> = self
            .chunks
            .iter()
            .map(|c| c.source.clone())
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
        docs.sort();
        docs
    }
}

// ── Math ───────────────────────────────────────────────────────────────────────

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_chunk(text: &str, embedding: Vec<f32>) -> Chunk {
        Chunk {
            id: Uuid::new_v4(),
            source: "test.pdf".into(),
            text: text.into(),
            embedding,
        }
    }

    #[test]
    fn cosine_similarity_identical() {
        let v = vec![1.0f32, 0.0, 0.0];
        assert!((cosine_similarity(&v, &v) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn cosine_similarity_orthogonal() {
        let a = vec![1.0f32, 0.0];
        let b = vec![0.0f32, 1.0];
        assert!((cosine_similarity(&a, &b)).abs() < 1e-5);
    }

    #[test]
    fn search_returns_closest() {
        let mut store = VectorStore::new();
        store.insert(vec![
            make_chunk("hello", vec![1.0, 0.0]),
            make_chunk("world", vec![0.0, 1.0]),
        ]);
        let results = store.search(&[1.0, 0.0], 1);
        assert_eq!(results[0].text, "hello");
    }
}
