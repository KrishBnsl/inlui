from __future__ import annotations

import pytest
from langchain_core.documents import Document

from rag_api.errors import (
    DocumentValidationError,
    EmptyDocumentError,
    UnsupportedDocumentError,
    UploadTooLargeError,
)
from rag_api.rag.ingestion import (
    decode_inline_image,
    extract_text_from_pdf,
    ingest_file,
    validate_upload,
)
from rag_api.rag.text_splitter import create_text_splitter
from rag_api.rag.vector_store import VectorStore
from tests.conftest import FakeChat, FakeEmbeddings, make_pdf, make_png


def test_pdf_extraction_and_chunk_metadata(rag_config):
    content = make_pdf("JoSAA rank document " * 20)
    docs = ingest_file(
        content,
        "brochure.pdf",
        "application/pdf",
        FakeChat(),
        create_text_splitter(rag_config),
        rag_config.rag_max_upload_bytes,
    )
    assert docs
    assert "JoSAA" in extract_text_from_pdf(content)
    assert all(doc.metadata["source"] == "brochure.pdf" for doc in docs)
    assert all(doc.metadata["page"] == 1 for doc in docs)
    assert [doc.metadata["chunk_index"] for doc in docs] == list(range(len(docs)))


def test_supported_image_uses_injected_ocr_gateway(rag_config):
    chat = FakeChat()
    docs = ingest_file(
        make_png(),
        "portal.png",
        "image/png",
        chat,
        create_text_splitter(rag_config),
        rag_config.rag_max_upload_bytes,
    )
    assert docs[0].page_content == "JoSAA document rank 1234"
    assert docs[0].metadata["file_type"] == "image/png"
    assert len(chat.calls) == 1


@pytest.mark.parametrize(
    ("filename", "mime"),
    [("notes.txt", "text/plain"), ("brochure.pdf", "text/plain"), ("image.jpg", "image/png")],
)
def test_extension_and_mime_are_cross_validated(filename, mime):
    with pytest.raises(UnsupportedDocumentError):
        validate_upload(make_pdf(), filename, mime, 4096)


def test_empty_file_is_rejected():
    with pytest.raises(EmptyDocumentError):
        validate_upload(b"", "brochure.pdf", "application/pdf", 4096)


def test_upload_size_limit_is_rejected_before_parsing():
    with pytest.raises(UploadTooLargeError):
        validate_upload(b"%PDF-" + b"x" * 100, "brochure.pdf", "application/pdf", 16)


def test_signature_mismatch_is_rejected():
    with pytest.raises(DocumentValidationError):
        validate_upload(b"%PDF-not-a-png", "image.png", "image/png", 4096)


def test_inline_image_base64_and_mime_validation():
    import base64

    content = make_png()
    normalized, mime = decode_inline_image(base64.b64encode(content).decode(), "image/png", 2048)
    assert base64.b64decode(normalized) == content
    assert mime == "image/png"
    with pytest.raises(DocumentValidationError):
        decode_inline_image(normalized, "image/jpeg", 2048)


def test_faiss_insertion_retrieval_and_source_metadata():
    store = VectorStore(FakeEmbeddings())
    docs = [
        Document(page_content="freeze seat option", metadata={"source": "rules.pdf", "page": 2}),
        Document(page_content="rank document list", metadata={"source": "checklist.pdf", "page": 1}),
    ]
    assert store.add_documents(docs, "rules.pdf") == 2
    assert store.chunk_count == 2
    assert store.documents == ["rules.pdf"]
    result = store.similarity_search("freeze seat", k=1)
    assert len(result) == 1
    assert result[0].metadata["source"] == "rules.pdf"
    assert result[0].metadata["chunk_index"] == 0


def test_empty_vector_store_returns_empty_list():
    store = VectorStore(FakeEmbeddings())
    assert store.similarity_search("rank", k=5) == []
    assert store.chunk_count == 0
