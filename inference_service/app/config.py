"""
Configuration for the JoSAA Inference Service.

All values are read from environment variables (or a .env file).
Defaults are set so the service starts out-of-the-box for local development
when run from the repo root alongside the models/ directory.
"""

from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Inference service configuration — all values can be overridden via env vars."""

    # ── Paths ─────────────────────────────────────────────────────────────────
    models_dir: Path = Path("../models")
    data_dir: Path = Path("../models/data")
    artifacts_dir: Path = Path("../models/model_artifacts")

    # ── Monte Carlo ───────────────────────────────────────────────────────────
    mc_n_sims: int = 1000
    mc_std_dev_default: float = 150.0

    # ── Recommendation scoring weights ────────────────────────────────────────
    score_prob_weight: float = 100.0
    score_margin_weight: float = 20.0

    # ── Probability thresholds for Safe / Moderate / Ambitious ────────────────
    safe_threshold: float = 0.80      # ≥80 % → Safe
    moderate_threshold: float = 0.40  # 40–80 % → Moderate; <40 % → Ambitious

    # ── Server ────────────────────────────────────────────────────────────────
    port: int = 8082
    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:3001"]

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


# Singleton — import this everywhere
settings = Settings()
