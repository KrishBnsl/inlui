//! Document ingestion: PDF text extraction, image OCR via GPT-4o-mini vision,
//! character-based chunking, embedding, and insertion into the vector store.

use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
use uuid::Uuid;

use super::{embeddings::embed_texts, vector_store::Chunk, RagService};

// ── Public entry point ─────────────────────────────────────────────────────────

/// Ingest a single uploaded file.
///
/// - `.pdf`  → pure-Rust text extraction via `pdf-extract`
/// - `.png` / `.jpg` / `.jpeg` / `.webp` → GPT-4o-mini vision OCR
///
/// Returns the number of chunks added to the vector store.
pub async fn ingest_document(
    rag: &RagService,
    filename: &str,
    bytes: &[u8],
) -> Result<usize, Box<dyn std::error::Error + Send + Sync>> {
    let lower = filename.to_lowercase();

    let text = if lower.ends_with(".pdf") {
        pdf_extract::extract_text_from_mem(bytes)
            .map_err(|e| format!("PDF extraction failed: {e}"))?
    } else if lower.ends_with(".png")
        || lower.ends_with(".jpg")
        || lower.ends_with(".jpeg")
        || lower.ends_with(".webp")
    {
        ocr_image(rag, bytes, &lower).await?
    } else {
        return Err(format!(
            "Unsupported file type for '{filename}'. \
             Accepted: PDF, PNG, JPG, JPEG, WEBP."
        )
        .into());
    };

    let text = text.trim().to_string();
    if text.is_empty() {
        tracing::warn!("Ingested '{}' but extracted no text", filename);
        return Ok(0);
    }

    let raw_chunks = chunk_text(&text, rag.config.chunk_size, rag.config.chunk_overlap);
    if raw_chunks.is_empty() {
        return Ok(0);
    }

    tracing::info!(
        "Ingesting '{}': {} chunks → embedding…",
        filename,
        raw_chunks.len()
    );

    let embeddings = embed_texts(&rag.openai, &rag.config, raw_chunks.clone()).await?;

    let chunks: Vec<Chunk> = raw_chunks
        .into_iter()
        .zip(embeddings)
        .map(|(text, embedding)| Chunk {
            id: Uuid::new_v4(),
            source: filename.to_string(),
            text,
            embedding,
        })
        .collect();

    let count = chunks.len();
    rag.store.write().await.insert(chunks);
    tracing::info!("Inserted {} chunks from '{}'", count, filename);

    Ok(count)
}

// ── Chunking ───────────────────────────────────────────────────────────────────

/// Split `text` into overlapping character-based chunks.
fn chunk_text(text: &str, chunk_size: usize, overlap: usize) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let total = chars.len();
    if total == 0 {
        return vec![];
    }

    let step = chunk_size.saturating_sub(overlap).max(1);
    let mut chunks = Vec::new();
    let mut start = 0;

    while start < total {
        let end = (start + chunk_size).min(total);
        let chunk: String = chars[start..end].iter().collect();
        let trimmed = chunk.trim().to_string();
        if !trimmed.is_empty() {
            chunks.push(trimmed);
        }
        if end >= total {
            break;
        }
        start += step;
    }

    chunks
}

// ── Image OCR via GPT-4o-mini vision ──────────────────────────────────────────

async fn ocr_image(
    rag: &RagService,
    bytes: &[u8],
    lower_name: &str,
) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    use async_openai::types::{
        ChatCompletionRequestUserMessageArgs,
        ChatCompletionRequestMessageContentPartImageArgs,
        ChatCompletionRequestMessageContentPartTextArgs,
        CreateChatCompletionRequestArgs,
        ImageDetail, ImageUrlArgs,
    };

    let mime = if lower_name.ends_with(".png") {
        "image/png"
    } else if lower_name.ends_with(".webp") {
        "image/webp"
    } else {
        "image/jpeg"
    };

    let data_url = format!("data:{mime};base64,{}", BASE64.encode(bytes));

    let text_part = ChatCompletionRequestMessageContentPartTextArgs::default()
        .text(
            "Extract ALL visible text from this image exactly as it appears — \
             including labels, statuses, dates, names, rank numbers, institute \
             names, document lists, and any other readable content. \
             Output only the extracted text, nothing else.",
        )
        .build()?;

    let image_part = ChatCompletionRequestMessageContentPartImageArgs::default()
        .image_url(
            ImageUrlArgs::default()
                .url(data_url)
                .detail(ImageDetail::Auto)
                .build()?,
        )
        .build()?;

    let message = ChatCompletionRequestUserMessageArgs::default()
        .content(vec![text_part.into(), image_part.into()])
        .build()?;

    let request = CreateChatCompletionRequestArgs::default()
        .model(&rag.config.chat_model)
        .max_tokens(1024u32)
        .messages(vec![message.into()])
        .build()?;

    let response = rag.openai.chat().create(request).await?;

    Ok(response
        .choices
        .into_iter()
        .next()
        .and_then(|c| c.message.content)
        .unwrap_or_default())
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_text_basic() {
        let text = "abcde";
        let chunks = chunk_text(text, 3, 1);
        // [abc] [bcd] [cde]  — but step = 3-1=2: [abc] start=2 [cde]
        assert!(!chunks.is_empty());
        assert!(chunks[0].len() <= 3);
    }

    #[test]
    fn chunk_text_empty() {
        assert!(chunk_text("", 512, 64).is_empty());
    }

    #[test]
    fn chunk_text_shorter_than_size() {
        let chunks = chunk_text("hello", 512, 64);
        assert_eq!(chunks, vec!["hello"]);
    }
}
