"""
Entrypoint for the RAG service.

Usage:
    uvicorn main:app --host 0.0.0.0 --port 8081

All logic lives in the `app/` package — this file just wires up the factory.
"""

from app import create_app

app = create_app()
