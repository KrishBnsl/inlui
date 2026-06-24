"""
Entry-point for the JoSAA Inference Service.

Run locally:
    uvicorn main:app --host 0.0.0.0 --port 8082 --reload

Or via Docker (see Dockerfile).
"""

from app import create_app

app = create_app()
