"""
Entry-point for the JoSAA Inference Service.

Run locally:
    uvicorn inference_api.main:app --host 0.0.0.0 --port 8082 --reload

Or via Docker (see Dockerfile).
"""

from inference_api import create_app

app = create_app()
