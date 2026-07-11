"""Internal RAG errors with stable, sanitized public messages."""


class RagServiceError(RuntimeError):
    public_message = "The RAG request could not be completed."


class ProviderUnavailableError(RagServiceError):
    public_message = "Gemini is not configured. Set GOOGLE_API_KEY and restart the RAG service."


class UpstreamProviderError(RagServiceError):
    public_message = "The Gemini provider could not complete the request."


class VectorStoreUnavailableError(RagServiceError):
    public_message = "The FAISS vector store is unavailable."


class DocumentValidationError(RagServiceError, ValueError):
    public_message = "The uploaded document is invalid."


class EmptyDocumentError(DocumentValidationError):
    public_message = "The uploaded file is empty."


class UploadTooLargeError(DocumentValidationError):
    public_message = "The uploaded file exceeds the configured size limit."


class UnsupportedDocumentError(DocumentValidationError):
    public_message = "Unsupported document type. Accepted types: PDF, PNG, JPG, JPEG, WEBP."


class DocumentExtractionError(DocumentValidationError):
    public_message = "Text could not be extracted from the uploaded document."
