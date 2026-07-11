from __future__ import annotations

from io import BytesIO

import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage
from PIL import Image

from rag_api.config import Settings
from rag_api.rag import RagPipeline


class FakeEmbeddings(Embeddings):
    """Synthetic deterministic embeddings used only by unit tests."""

    @staticmethod
    def _vector(text: str) -> list[float]:
        lowered = text.casefold()
        values = [
            float(lowered.count("rank")),
            float(lowered.count("document")),
            float(lowered.count("freeze")),
            float(lowered.count("seat")),
            float(len(lowered)) / 1000.0,
        ]
        return values if any(values) else [0.0, 0.0, 0.0, 0.0, 0.001]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


class FakeChat:
    def __init__(self, answer: str = "Grounded mocked answer") -> None:
        self.answer = answer
        self.calls: list[list] = []

    def invoke(self, messages):
        self.calls.append(messages)
        system_text = str(getattr(messages[0], "content", ""))
        if "OCR engine" in system_text:
            return AIMessage(content="JoSAA document rank 1234")
        return AIMessage(content=self.answer)


def make_pdf(text: str = "JoSAA rank document freeze seat") -> bytes:
    """Build a tiny one-page PDF with extractable Helvetica text."""
    escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream = f"BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET".encode("latin-1")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    output = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for index, obj in enumerate(objects, start=1):
        offsets.append(len(output))
        output.extend(f"{index} 0 obj\n".encode("ascii"))
        output.extend(obj)
        output.extend(b"\nendobj\n")
    xref = len(output)
    output.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    output.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        output.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    output.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode("ascii")
    )
    return bytes(output)


def make_png() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (8, 8), "white").save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def rag_config() -> Settings:
    return Settings(
        _env_file=None,
        google_api_key=None,
        rag_chunk_size=100,
        rag_chunk_overlap=10,
        rag_max_upload_bytes=4096,
        rag_max_inline_image_bytes=2048,
        rag_request_timeout_seconds=1.0,
        rag_provider_timeout_seconds=1.0,
    )


@pytest.fixture
def fake_chat() -> FakeChat:
    return FakeChat()


@pytest.fixture
def pipeline(rag_config, fake_chat) -> RagPipeline:
    return RagPipeline(
        embeddings=FakeEmbeddings(),
        llm=fake_chat,
        config=rag_config,
    )
