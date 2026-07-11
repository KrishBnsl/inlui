"""Validated loading of the artifacts required by the inference API.

Serving deliberately requires a past-only program universe, explicit model
metadata, and uncertainty estimated from rolling-origin out-of-fold residuals.
A final-test residual file is never accepted as a production uncertainty source.
"""

from __future__ import annotations

import json
import hashlib
import logging
import math
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

import joblib
import pandas as pd
import josaa_core
from josaa_core import baselines as josaa_baselines

from inference_api.config import settings

logger = logging.getLogger("inference_api.artifact_loader")

ARTIFACT_RECOVERY_COMMAND = (
    ".venv/bin/python research/pipeline/research_evaluation.py "
    "--data research/data/processed/josaa_cleaned.csv "
    "--output-dir research/reports --artifact-dir research/artifacts "
    "--data-cutoff-year 2025 --prediction-year 2026"
)
NORMAL_90_INTERVAL_Z = 1.6448536269514722
REQUIRED_UNIVERSE_COLUMNS = {
    "year",
    "round",
    "institute",
    "institute_type",
    "program",
    "category",
    "gender",
    "quota",
}


class ArtifactLoadError(RuntimeError):
    """Raised when serving artifacts are missing, invalid, or unsafe."""


def _recovery_message(reason: str) -> str:
    return f"{reason} Generate research-safe serving artifacts with: {ARTIFACT_RECOVERY_COMMAND}"


def _install_sklearn_compat_shims() -> None:
    """Patch a small sklearn pickle-compat gap for older runtime packages."""
    try:
        import sklearn.compose._column_transformer as column_transformer
    except Exception:
        return

    if not hasattr(column_transformer, "_RemainderColsList"):
        class _RemainderColsList(list):
            pass

        column_transformer._RemainderColsList = _RemainderColsList


@contextmanager
def _legacy_core_pickle_aliases():
    """Resolve pre-migration model class paths without restoring path hacks.

    Existing, hash-verified local model bundles were serialized while the core
    package lived under ``shared.josaa_core``. New artifacts serialize the same
    class from ``josaa_core``. These temporary module aliases keep old bundles
    readable during the repository migration and are removed immediately after
    deserialization.
    """

    shared_module = ModuleType("shared")
    shared_module.__path__ = []
    shared_module.josaa_core = josaa_core
    aliases = {
        "shared": shared_module,
        "shared.josaa_core": josaa_core,
        "shared.josaa_core.baselines": josaa_baselines,
    }
    previous = {name: sys.modules.get(name) for name in aliases}
    sys.modules.update(aliases)
    try:
        yield
    finally:
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


@dataclass
class ArtifactStore:
    """Container for one internally consistent serving artifact bundle."""

    model: Any
    prep_pipeline: Any
    label_encoders: dict
    feature_schema: list[str]
    universe: pd.DataFrame
    std_map: pd.DataFrame
    metadata: dict[str, Any] = field(default_factory=dict)
    universe_size: int = field(init=False)
    artifact_status: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.universe_size = len(self.universe)
        if not self.artifact_status:
            self.artifact_status = {
                "model": "loaded" if self.model is not None else "missing",
                "preprocessing_pipeline": "bundled_with_model" if self.model is not None else "missing",
                "label_encoders": "not_required",
                "feature_schema": "loaded" if self.feature_schema else "missing",
                "residual_uncertainty": "loaded" if not self.std_map.empty else "missing",
                "program_universe": "loaded" if self.universe_size else "missing",
                "model_metadata": "loaded" if self.metadata else "missing",
            }


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else Path.cwd() / path


def _required_file(directory: Path, filename: str) -> Path:
    path = directory / filename
    if not path.is_file():
        raise ArtifactLoadError(_recovery_message(f"Required artifact is missing: {path}."))
    return path


def _load_json(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactLoadError(_recovery_message(f"{label} is not valid JSON: {path}.")) from exc


def _verify_artifact_hash(path: Path, metadata: dict[str, Any]) -> None:
    record = metadata.get("artifacts", {}).get(path.name)
    expected = record.get("sha256") if isinstance(record, dict) else None
    if not isinstance(expected, str) or len(expected) != 64:
        raise ArtifactLoadError(f"model_metadata.json has no valid SHA-256 for {path.name}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest().casefold() != expected.casefold():
        raise ArtifactLoadError(_recovery_message(f"Artifact hash mismatch: {path}."))


def _load_model(artifacts_dir: Path, metadata: dict[str, Any]) -> Any:
    path = _required_file(artifacts_dir, "pre_counselling_model.joblib")
    _verify_artifact_hash(path, metadata)
    logger.info("Loading model from %s", path)
    try:
        with _legacy_core_pickle_aliases():
            obj = joblib.load(path)
    except Exception as exc:
        raise ArtifactLoadError(_recovery_message(f"Could not deserialize model artifact: {path}.")) from exc
    if isinstance(obj, dict):
        obj = obj.get("model")
    if obj is None or not hasattr(obj, "predict"):
        raise ArtifactLoadError("pre_counselling_model.joblib does not contain a predictor")
    return obj


def _load_feature_schema(artifacts_dir: Path, metadata: dict[str, Any]) -> list[str]:
    path = _required_file(artifacts_dir, "pre_counselling_feature_schema.json")
    _verify_artifact_hash(path, metadata)
    payload = _load_json(path, "pre_counselling_feature_schema.json")
    features = payload.get("features") if isinstance(payload, dict) else None
    if not isinstance(features, list) or not features or not all(isinstance(item, str) for item in features):
        raise ArtifactLoadError("pre_counselling_feature_schema.json must contain a non-empty string feature list")
    return features


def _load_metadata(artifacts_dir: Path) -> dict[str, Any]:
    path = _required_file(artifacts_dir, "model_metadata.json")
    payload = _load_json(path, "model_metadata.json")
    if not isinstance(payload, dict):
        raise ArtifactLoadError("model_metadata.json must contain an object")

    required = {
        "schema_version",
        "model_version",
        "feature_mode",
        "data_cutoff_year",
        "prediction_year",
        "snapshot_year",
        "dataset_sha256",
        "training_years",
        "seed",
        "features",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ArtifactLoadError(_recovery_message(f"model_metadata.json is missing fields: {missing}."))
    if payload["schema_version"] != 1:
        raise ArtifactLoadError("Unsupported model metadata schema_version")
    if payload["feature_mode"] != "pre_counselling":
        raise ArtifactLoadError("Serving requires feature_mode='pre_counselling'")
    try:
        cutoff_year = int(payload["data_cutoff_year"])
        prediction_year = int(payload["prediction_year"])
        snapshot_year = int(payload["snapshot_year"])
        train_years = [int(year) for year in payload["training_years"]]
    except (TypeError, ValueError) as exc:
        raise ArtifactLoadError("Model metadata years must be integers") from exc
    if not train_years or max(train_years) > cutoff_year:
        raise ArtifactLoadError("train_years must not exceed data_cutoff_year")
    if prediction_year <= cutoff_year or snapshot_year > cutoff_year:
        raise ArtifactLoadError("Prediction year must follow the explicit data/snapshot cutoff")
    digest = str(payload["dataset_sha256"])
    if len(digest) != 64 or any(char not in "0123456789abcdefABCDEF" for char in digest):
        raise ArtifactLoadError("model metadata dataset_sha256 must be a 64-character hexadecimal digest")
    if isinstance(payload["seed"], bool) or not isinstance(payload["seed"], int):
        raise ArtifactLoadError("model metadata seed must be an integer")
    if not str(payload["model_version"]).strip():
        raise ArtifactLoadError("model_version must be non-empty")
    if not isinstance(payload["features"], list) or not payload["features"]:
        raise ArtifactLoadError("model metadata features must be a non-empty list")
    return payload


def _load_universe(artifacts_dir: Path, metadata: dict[str, Any]) -> pd.DataFrame:
    path = _required_file(artifacts_dir, "inference_universe.csv")
    _verify_artifact_hash(path, metadata)
    logger.info("Loading past-only inference universe from %s", path)
    try:
        universe = pd.read_csv(path, low_memory=False)
    except Exception as exc:
        raise ArtifactLoadError(_recovery_message(f"Could not read inference universe: {path}.")) from exc

    missing = sorted(REQUIRED_UNIVERSE_COLUMNS - set(universe.columns))
    if missing:
        raise ArtifactLoadError(f"inference_universe.csv is missing columns: {missing}")
    if universe.empty:
        raise ArtifactLoadError("inference_universe.csv contains no candidate rows")
    prediction_year = int(metadata["prediction_year"])
    years = pd.to_numeric(universe["year"], errors="coerce")
    if years.isna().any() or not (years.astype(int) == prediction_year).all():
        raise ArtifactLoadError("Every inference universe row must use metadata prediction_year")
    forbidden_observed = {
        "closing_rank",
        "opening_rank",
        "log_opening_rank",
        "opening_percentile",
        "applicants",
        "prev_round_closing_rank",
        "is_final_round",
    }
    present = sorted(forbidden_observed & set(universe.columns))
    if present:
        raise ArtifactLoadError(
            "Pre-counselling inference universe contains unavailable target-season columns: "
            f"{present}"
        )
    universe = universe.drop_duplicates(
        subset=["round", "institute", "program", "category", "gender", "quota"]
    ).reset_index(drop=True)
    return universe


def _validated_quantile(record: Any, label: str) -> tuple[int, float]:
    if not isinstance(record, dict):
        raise ArtifactLoadError(f"{label} calibration entry must be an object")
    count = record.get("count")
    quantile = record.get("abs_residual_quantile")
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ArtifactLoadError(f"{label} calibration count must be positive")
    if not isinstance(quantile, (int, float)) or not math.isfinite(float(quantile)) or float(quantile) <= 0:
        raise ArtifactLoadError(f"{label} abs_residual_quantile must be finite and positive")
    return count, float(quantile)


def _load_std_map(artifacts_dir: Path, metadata: dict[str, Any]) -> pd.DataFrame:
    path = _required_file(artifacts_dir, "uncertainty_calibration.json")
    _verify_artifact_hash(path, metadata)
    payload = _load_json(path, "uncertainty_calibration.json")
    required = {
        "schema_version",
        "method",
        "source_split",
        "target_coverage",
        "calibration_years",
        "model_training_years",
        "prediction_year",
        "dataset_sha256",
        "seed",
        "global",
        "groups",
    }
    if not isinstance(payload, dict):
        raise ArtifactLoadError("uncertainty_calibration.json must contain an object")
    missing = sorted(required - set(payload))
    if missing:
        raise ArtifactLoadError(_recovery_message(f"Uncertainty calibration is missing fields: {missing}."))
    if payload["schema_version"] != 1 or payload["method"] != "rolling_origin_oof_absolute_residual":
        raise ArtifactLoadError("Unsupported uncertainty calibration schema or method")
    if str(payload["source_split"]).casefold() == "test":
        raise ArtifactLoadError("Final-test residuals cannot calibrate production uncertainty")
    if payload["source_split"] != "rolling_origin_oof":
        raise ArtifactLoadError("Uncertainty source_split must be 'rolling_origin_oof'")
    try:
        target_year = int(payload["prediction_year"])
        calibration_years = [int(year) for year in payload["calibration_years"]]
        training_years = [int(year) for year in payload["model_training_years"]]
    except (TypeError, ValueError) as exc:
        raise ArtifactLoadError("Uncertainty calibration years must be integers") from exc
    if target_year != int(metadata["prediction_year"]):
        raise ArtifactLoadError("Uncertainty prediction_year must match model prediction_year")
    if not calibration_years or max(calibration_years) >= target_year:
        raise ArtifactLoadError("Calibration years must precede prediction_year")
    if not training_years or max(training_years) >= target_year:
        raise ArtifactLoadError("Uncertainty model-training years must precede prediction_year")
    if str(payload["dataset_sha256"]).casefold() != str(metadata["dataset_sha256"]).casefold():
        raise ArtifactLoadError("Model and uncertainty artifacts have different dataset hashes")
    if payload["seed"] != metadata["seed"]:
        raise ArtifactLoadError("Model and uncertainty artifacts have different random seeds")
    coverage = payload["target_coverage"]
    if not isinstance(coverage, (int, float)) or not math.isclose(float(coverage), 0.9):
        raise ArtifactLoadError("Serving currently requires target_coverage=0.9")
    global_count, global_radius = _validated_quantile(payload["global"], "global")
    groups = payload["groups"]
    if not isinstance(groups, list):
        raise ArtifactLoadError("Uncertainty calibration groups must be a list")

    rows = [{
        "institute_type_str": "__default__",
        "round": -1,
        "count": global_count,
        "uncertainty_radius": global_radius,
        "std_dev": global_radius / NORMAL_90_INTERVAL_Z,
    }]
    for index, group in enumerate(groups):
        count, radius = _validated_quantile(group, f"groups[{index}]")
        inst_type = str(group.get("institute_type_str", "")).upper()
        round_value = group.get("round")
        if inst_type not in {"IIT", "NIT", "IIIT", "GFTI"}:
            raise ArtifactLoadError(f"groups[{index}] has an invalid institute_type")
        if isinstance(round_value, bool) or not isinstance(round_value, int) or not 1 <= round_value <= 7:
            raise ArtifactLoadError(f"groups[{index}] has an invalid round")
        rows.append({
            "institute_type_str": inst_type,
            "round": round_value,
            "count": count,
            "uncertainty_radius": radius,
            "std_dev": radius / NORMAL_90_INTERVAL_Z,
        })

    result = pd.DataFrame(rows)
    result.attrs["calibration_metadata"] = payload
    return result


def load_all_artifacts() -> ArtifactStore:
    """Load and cross-check one research-safe serving artifact bundle."""
    artifacts_dir = _resolve(settings.artifacts_dir)
    _install_sklearn_compat_shims()
    if not artifacts_dir.is_dir():
        raise ArtifactLoadError(_recovery_message(f"Artifacts directory not found: {artifacts_dir}."))

    metadata = _load_metadata(artifacts_dir)
    feature_schema = _load_feature_schema(artifacts_dir, metadata)
    if feature_schema != metadata["features"]:
        raise ArtifactLoadError(
            "pre_counselling_feature_schema.json does not match model_metadata.json"
        )
    universe = _load_universe(artifacts_dir, metadata)
    std_map = _load_std_map(artifacts_dir, metadata)
    model = _load_model(artifacts_dir, metadata)

    metadata = dict(metadata)
    metadata["uncertainty"] = std_map.attrs.get("calibration_metadata", {})
    store = ArtifactStore(
        model=model,
        prep_pipeline=None,
        label_encoders={},
        feature_schema=feature_schema,
        universe=universe,
        std_map=std_map,
        metadata=metadata,
    )
    logger.info("All artifacts loaded; universe_size=%s", store.universe_size)
    return store
