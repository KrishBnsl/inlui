//! RAG answer generation: embed the query → retrieve top-k chunks → call GPT-4o-mini.
//! Optionally accepts an inline image (base64) that the user attaches to their question
//! (e.g. a screenshot of their counselling portal showing their current allotment).

use serde::Serialize;

use super::{embeddings::embed_texts, prompts, RagService};

// ── Response types (serialized to JSON by Axum) ────────────────────────────────

#[derive(Debug, Serialize, Clone)]
pub struct Source {
    /// Original filename that produced this excerpt.
    pub source: String,
    /// First 200 characters of the chunk, for display in the UI.
    pub excerpt: String,
}

#[derive(Debug, Serialize)]
pub struct AskResponse {
    pub answer: String,
    pub sources: Vec<Source>,
}

// ── Main function ──────────────────────────────────────────────────────────────

/// Answer a user's `question` using retrieved context from the vector store.
///
/// `image_b64` is an optional base64-encoded image (JPEG/PNG) that the user
/// attaches directly to their message — GPT-4o-mini will interpret it alongside
/// the retrieved text context.
pub async fn answer(
    rag: &RagService,
    question: &str,
    image_b64: Option<&str>,
) -> Result<AskResponse, Box<dyn std::error::Error + Send + Sync>> {
    // 1. Embed the question text
    let mut embs = embed_texts(&rag.openai, &rag.config, vec![question.to_string()]).await?;
    let query_embedding = embs.remove(0);

    // 2. Retrieve top-k chunks (clone data before releasing the lock)
    let (context_texts, sources) = {
        let store = rag.store.read().await;
        let top = store.search(&query_embedding, rag.config.top_k);
        let texts: Vec<String> = top.iter().map(|c| c.text.clone()).collect();
        let srcs: Vec<Source> = top
            .iter()
            .map(|c| {
                let excerpt: String = c.text.chars().take(200).collect();
                let excerpt = if c.text.len() > 200 {
                    format!("{excerpt}…")
                } else {
                    excerpt
                };
                Source { source: c.source.clone(), excerpt }
            })
            .collect();
        (texts, srcs)
    }; // RwLock released here

    // 3. Build the prompt
    let system_text = prompts::system_prompt();
    let user_text = prompts::user_prompt(&context_texts, question);

    // 4. Call GPT-4o-mini (with optional inline image)
    let answer_text = call_llm(rag, system_text, &user_text, image_b64).await?;

    Ok(AskResponse {
        answer: answer_text,
        sources,
    })
}

// ── LLM call (text-only or multimodal) ────────────────────────────────────────

async fn call_llm(
    rag: &RagService,
    system_text: &str,
    user_text: &str,
    image_b64: Option<&str>,
) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    use async_openai::types::{
        ChatCompletionRequestSystemMessageArgs,
        ChatCompletionRequestUserMessageArgs,
        ChatCompletionRequestMessageContentPartImageArgs,
        ChatCompletionRequestMessageContentPartTextArgs,
        CreateChatCompletionRequestArgs,
        ImageDetail, ImageUrlArgs,
    };

    let system_msg = ChatCompletionRequestSystemMessageArgs::default()
        .content(system_text)
        .build()?;

    let user_msg = if let Some(b64) = image_b64 {
        // Multimodal: text + image
        let text_part = ChatCompletionRequestMessageContentPartTextArgs::default()
            .text(user_text)
            .build()?;

        let image_part = ChatCompletionRequestMessageContentPartImageArgs::default()
            .image_url(
                ImageUrlArgs::default()
                    .url(format!("data:image/jpeg;base64,{b64}"))
                    .detail(ImageDetail::Auto)
                    .build()?,
            )
            .build()?;

        ChatCompletionRequestUserMessageArgs::default()
            .content(vec![
                text_part.into(),
                image_part.into(),
            ])
            .build()?
    } else {
        // Text only
        ChatCompletionRequestUserMessageArgs::default()
            .content(user_text)
            .build()?
    };

    let request = CreateChatCompletionRequestArgs::default()
        .model(&rag.config.chat_model)
        .temperature(rag.config.temperature)
        .max_tokens(1024u32)
        .messages(vec![system_msg.into(), user_msg.into()])
        .build()?;

    let response = rag.openai.chat().create(request).await?;

    Ok(response
        .choices
        .into_iter()
        .next()
        .and_then(|c| c.message.content)
        .unwrap_or_else(|| "I'm sorry, I couldn't generate a response.".into()))
}
