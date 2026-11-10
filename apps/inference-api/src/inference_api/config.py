"""
Configuration for the JoSAA inference API.

All values are read from environment variables (or a .env file).
Defaults are set so the service starts out-of-the-box for local development
when run from any directory inside a source checkout.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


def _discover_repository_root() -> Path:
    """Find a source checkout, with the container working directory as fallback."""

    for candidate in Path(__file__).resolve().parents:
        if (candidate / "docker-compose.yml").is_file():
            return candidate
    return Path.cwd().resolve()


REPO_ROOT = _discover_repository_root()
_SOURCE_SERVICE_ROOT = REPO_ROOT / "apps" / "inference-api"
SERVICE_ROOT = _SOURCE_SERVICE_ROOT if _SOURCE_SERVICE_ROOT.is_dir() else REPO_ROOT


class Settings(BaseSettings):
    """Inference service configuration — all values can be overridden via env vars."""

    # ── Paths ─────────────────────────────────────────────────────────────────
    research_dir: Path = REPO_ROOT / "research"
    data_dir: Path = REPO_ROOT / "research" / "data"
    artifacts_dir: Path = REPO_ROOT / "research" / "artifacts"

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
    cors_origins: list[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
    ]

    model_config = SettingsConfigDict(
        env_file=(REPO_ROOT / ".env", SERVICE_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )


# Singleton — import this everywhere
settings = Settings()
