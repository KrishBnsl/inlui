//! Prompt templates for the counselling advisor.

/// System prompt injected at the start of every LLM conversation.
pub fn system_prompt() -> &'static str {
    "You are a helpful JoSAA counselling advisor for students appearing in JEE \
     (Joint Entrance Examination) in India. You assist with:\n\
     - The JoSAA counselling and seat allotment process\n\
     - Document requirements for reporting to institutes\n\
     - Choice filling strategy and deadlines\n\
     - Quota rules (AI, HS, EWS, OBC-NCL, SC, ST, PwD)\n\
     - Withdrawal, upgradation, and freeze/float/slide options\n\
     - Reading and interpreting their uploaded counselling documents and screenshots\n\
     \n\
     When answering:\n\
     - Prioritise information from the uploaded context documents\n\
     - If a screenshot of a counselling portal is attached, reference the specific \
       details visible in it\n\
     - If the context is insufficient, clearly say so and supplement with general knowledge\n\
     - Be concise, accurate, and encouraging\n\
     - Format steps and lists clearly\n\
     - Always advise students to verify critical details on the official JoSAA portal \
       (josaa.nic.in)"
}

/// Build the user-facing message containing retrieved context and the question.
pub fn user_prompt(context_chunks: &[String], question: &str) -> String {
    if context_chunks.is_empty() {
        return format!(
            "Question: {question}\n\n\
             Note: No documents have been uploaded yet. \
             I will answer from general knowledge."
        );
    }

    let context = context_chunks
        .iter()
        .enumerate()
        .map(|(i, chunk)| format!("[Excerpt {}]\n{chunk}", i + 1))
        .collect::<Vec<_>>()
        .join("\n\n---\n\n");

    format!(
        "Relevant excerpts from uploaded documents:\n\n\
         {context}\n\n\
         ════════════════════════════════\n\n\
         Question: {question}\n\n\
         Answer using the excerpts above where possible. \
         Note which excerpt supports each point."
    )
}
