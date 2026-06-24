"""
Artifact loader — loads all trained ML artifacts exactly once at service startup.

Stored in `app.state.artifacts` as an `ArtifactStore` dataclass so every
request handler can access them without reloading.

Loaded artifacts
----------------
model           : sklearn estimator (unwrapped from dict wrapper if needed)
prep_pipeline   : sklearn Pipeline  — transforms the 10-column feature DataFrame
label_encoders  : dict              — categorical → integer mappings from label_encoders.json
universe        : pd.DataFrame      — 2024 Round-6 rows from josaa_cleaned.csv (deduped)
std_map         : pd.DataFrame      — residual std_dev per (institute_type_str, round)
                  derived from residuals_analysis.csv
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from app.config import settings

logger = logging.getLogger("inference_service.artifact_loader")


# ── Data container ─────────────────────────────────────────────────────────────

@dataclass
class ArtifactStore:
    """Immutable container for all loaded artifacts."""

    model: Any
    prep_pipeline: Any
    label_encoders: dict
    universe: pd.DataFrame
    std_map: pd.DataFrame
    universe_size: int = field(init=False)

    def __post_init__(self) -> None:
        self.universe_size = len(self.universe)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _resolve(path: Path) -> Path:
    """Return path as-is if absolute, else resolve relative to CWD."""
    return path if path.is_absolute() else Path.cwd() / path


def _load_model(artifacts_dir: Path) -> Any:
    """Load the final regression model, unwrapping dict wrapper if present."""
    model_path = artifacts_dir / "final_regression_model.joblib"
    logger.info(f"Loading model from {model_path}")
    obj = joblib.load(model_path)
    if isinstance(obj, dict):
        return obj["model"]
    return obj


def _load_prep_pipeline(artifacts_dir: Path) -> Any:
    """Load the preprocessing pipeline used during training."""
    pipeline_path = artifacts_dir / "preprocessing_pipeline.joblib"
    logger.info(f"Loading preprocessing pipeline from {pipeline_path}")
    return joblib.load(pipeline_path)


def _load_label_encoders(artifacts_dir: Path) -> dict:
    """Load the label encoder mappings from label_encoders.json."""
    enc_path = artifacts_dir / "label_encoders.json"
    logger.info(f"Loading label encoders from {enc_path}")
    with open(enc_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_universe(data_dir: Path) -> pd.DataFrame:
    """
    Load the 2024 Round-5 universe of choices from feature_engineered.csv.
    Deduplicates on (institute, program, category, gender, quota).
    """
    csv_path = data_dir / "features" / "feature_engineered.csv"
    logger.info(f"Loading universe from {csv_path} …")
    df = pd.read_csv(csv_path, low_memory=False)

    # Filter to 2024 Round 5 only — this is the most recent final round
    universe = df[(df["year"] == 2024) & (df["round"] == 5)].copy()
    universe = universe.drop_duplicates(
        subset=["institute", "program", "category", "gender", "quota"]
    )
    universe = universe.reset_index(drop=True)
    logger.info(f"Universe loaded: {len(universe)} rows after dedup")
    return universe


def _load_std_map(artifacts_dir: Path) -> pd.DataFrame:
    """
    Derive empirical residual std_dev per (institute_type, round) from
    residuals_analysis.csv.  A floor of 100 is applied to avoid over-confident
    predictions on historically stable programs.
    """
    residuals_path = artifacts_dir / "residuals_analysis.csv"
    logger.info(f"Loading residuals analysis from {residuals_path} …")
    df = pd.read_csv(residuals_path, low_memory=False)

    # The residuals CSV has columns: institute_type_encoded, round, residual, …
    # We need institute_type as a string — load label encoders here for decoding
    enc_path = artifacts_dir / "label_encoders.json"
    with open(enc_path, "r", encoding="utf-8") as f:
        encoders = json.load(f)

    inst_type_inv: dict[int, str] = {
        v: k for k, v in encoders["institute_type"].items()
    }

    if "institute_type_encoded" in df.columns:
        df["institute_type_str"] = df["institute_type_encoded"].map(inst_type_inv)
    elif "institute_type" in df.columns:
        df["institute_type_str"] = df["institute_type"]
    else:
        # Fallback: return a default std_map with a single row
        logger.warning(
            "residuals_analysis.csv has no institute_type column — using global default std_dev"
        )
        return pd.DataFrame(
            [{"institute_type_str": "__default__", "round": -1, "std_dev": settings.mc_std_dev_default}]
        )

    grouped = (
        df.groupby(["institute_type_str", "round"])["residual"]
        .std()
        .reset_index()
        .rename(columns={"residual": "std_dev"})
    )
    grouped["std_dev"] = grouped["std_dev"].clip(lower=100.0).fillna(settings.mc_std_dev_default)
    logger.info(f"Std map computed: {len(grouped)} groups")
    return grouped


# ── Public loader ──────────────────────────────────────────────────────────────

def load_all_artifacts() -> ArtifactStore:
    """
    Load every artifact needed for inference.  Called once at startup.

    Returns
    -------
    ArtifactStore
        Dataclass with model, pipeline, encoders, universe, and std_map.

    Raises
    ------
    FileNotFoundError
        If any required artifact is missing.
    """
    artifacts_dir = _resolve(settings.artifacts_dir)
    data_dir = _resolve(settings.data_dir)

    # Fail fast if the directory doesn't exist
    if not artifacts_dir.exists():
        raise FileNotFoundError(
            f"Artifacts directory not found: {artifacts_dir}. "
            "Set ARTIFACTS_DIR in your .env to point at models/model_artifacts/"
        )

    model = _load_model(artifacts_dir)
    prep_pipeline = _load_prep_pipeline(artifacts_dir)
    label_encoders = _load_label_encoders(artifacts_dir)
    universe = _load_universe(data_dir)
    std_map = _load_std_map(artifacts_dir)

    store = ArtifactStore(
        model=model,
        prep_pipeline=prep_pipeline,
        label_encoders=label_encoders,
        universe=universe,
        std_map=std_map,
    )
    logger.info(
        f"All artifacts loaded — universe_size={store.universe_size}"
    )
    return store
