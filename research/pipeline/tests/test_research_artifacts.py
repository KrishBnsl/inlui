from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd


REPO_ROOT = Path(__file__).parents[3]
REPORTS = REPO_ROOT / "research" / "reports"
ARTIFACTS = REPO_ROOT / "research" / "artifacts"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_manifest_hashes_every_versioned_evidence_file():
    manifest = _json(REPORTS / "artifact_manifest.json")
    entries = [
        *manifest["versioned_outputs"],
        *[
            entry
            for entry in manifest["generated_serving_artifacts"]
            if entry["committed_expected"]
        ],
    ]
    assert entries
    for entry in entries:
        path = REPO_ROOT / entry["path"]
        assert path.exists(), entry["path"]
        assert path.stat().st_size == entry["bytes"]
        assert _sha256(path) == entry["sha256"]


def test_machine_results_keep_final_test_out_of_selection_and_interval_fit():
    results = _json(REPORTS / "metrics.json")
    final_year = results["evaluation"]["final_test_year"]
    assert results["status"] == "exploratory_unverified_data_provenance"
    assert results["dataset"]["provenance"] == "unverified"
    assert results["dataset"]["license"] == "unresolved"
    assert results["primary_model"] == "naive_last_year"
    assert max(results["model_selection"]["selection_years"]) < final_year
    assert results["uncertainty"]["final_test_residuals_used_for_interval"] is False
    assert results["uncertainty"]["residual_model"] == results["primary_model"]
    assert results["timestamp_source"] == "SOURCE_DATE_EPOCH"
    assert "not the wall-clock" in results["timestamp_semantics"]


def test_generated_evidence_records_one_non_circular_source_snapshot():
    payloads = [
        _json(REPORTS / "metrics.json"),
        _json(REPORTS / "artifact_manifest.json"),
        _json(ARTIFACTS / "model_metadata.json"),
        _json(ARTIFACTS / "uncertainty_calibration.json"),
    ]
    source_commits = {payload["source_commit"] for payload in payloads}
    assert len(source_commits) == 1
    source_commit = source_commits.pop()
    assert isinstance(source_commit, str) and len(source_commit) == 40
    assert all(character in "0123456789abcdef" for character in source_commit)
    for payload in payloads:
        assert payload["generated_from_commit"] == source_commit
        assert isinstance(payload["source_worktree_dirty"], bool)
        assert payload["worktree_dirty"] == payload["source_worktree_dirty"]
        assert payload["git_commit"] == source_commit
        assert payload["git_worktree_dirty"] == payload["source_worktree_dirty"]


def test_pre_final_selection_table_reproduces_selected_model():
    selection = pd.read_csv(REPORTS / "research_model_selection.csv")
    winner = selection.sort_values(["validation_mean_mae", "model"]).iloc[0]
    results = _json(REPORTS / "metrics.json")
    assert winner["model"] == results["primary_model"]
    assert int(winner["validation_origin_count"]) == 4


def test_uncertainty_contract_is_oof_and_strictly_before_prediction_year():
    payload = _json(ARTIFACTS / "uncertainty_calibration.json")
    assert payload["method"] == "rolling_origin_oof_absolute_residual"
    assert payload["source_split"] == "rolling_origin_oof"
    assert payload["formal_coverage_guarantee"] is False
    assert payload["target_coverage"] == 0.9
    assert max(payload["calibration_years"]) < payload["prediction_year"]
    assert payload["global"]["count"] > 0
    assert payload["global"]["abs_residual_quantile"] > 0


def test_serving_contract_selects_past_only_last_observation():
    metadata = _json(ARTIFACTS / "model_metadata.json")
    schema = _json(ARTIFACTS / "pre_counselling_feature_schema.json")
    assert metadata["model_family"] == "LastObservationRegressor"
    assert metadata["selection_used_final_test"] is False
    assert metadata["feature_mode"] == "pre_counselling"
    assert metadata["data_cutoff_year"] < metadata["prediction_year"]
    assert metadata["features"] == ["last_closing_rank"]
    assert schema["features"] == metadata["features"]
    assert schema["availability_mode"] == "pre_counselling"


def test_generated_universe_has_no_observed_target_ranks_when_locally_present():
    # The universe is intentionally absent in a fresh clone. If a developer has
    # generated it, enforce the same leakage contract as the synthetic tests.
    universe_path = ARTIFACTS / "inference_universe.csv"
    if not universe_path.exists():
        return
    universe = pd.read_csv(universe_path, low_memory=False)
    forbidden = {"opening_rank", "closing_rank", "opening_percentile", "applicants"}
    assert not (forbidden & set(universe.columns))
    assert set(universe["year"]) == {2026}
    assert set(universe["snapshot_year"]) == {2025}


def test_primary_docs_quote_generated_hash_and_primary_metric():
    results = _json(REPORTS / "metrics.json")
    dataset_hash = results["dataset"]["sha256"]
    final_mae = results["evaluation"]["final_test_metrics"]["mae"]
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    technical_report = (REPO_ROOT / "docs" / "TECHNICAL_REPORT.md").read_text(
        encoding="utf-8"
    )
    assert dataset_hash in technical_report
    assert f"{final_mae:,.2f}" in readme
    assert f"{final_mae:,.2f}" in technical_report


def test_legacy_headline_benchmark_and_registry_claim_are_replaced():
    invalidated = pd.read_csv(REPORTS / "legacy" / "regression_benchmark.csv")
    assert invalidated.loc[0, "evidence_status"] == "INVALIDATED"
    assert "test_MAE" not in invalidated.columns

    legacy_config = _json(REPORTS / "legacy" / "experiment_config.json")
    assert legacy_config["evidence_status"] == "SUPERSEDED_INVALID_METHODOLOGY"
    assert legacy_config["replacement"] == "research/reports/metrics.json"

    registry = pd.read_csv(REPORTS / "model_registry.csv")
    assert len(registry) == 1
    assert registry.loc[0, "Feature_Mode"] == "pre_counselling"
    assert registry.loc[0, "Evidence_Status"] == "exploratory_unverified_data_provenance"
    assert registry.loc[0, "Deployment_Status"] != "Production"


def test_legacy_cli_entrypoints_fail_before_optional_imports():
    for relative_path in (
        "research/pipeline/benchmark.py",
        "research/pipeline/train_models.py",
        "research/pipeline/evaluation.py",
        "research/pipeline/recommendation.py",
        "research/pipeline/predict.py",
    ):
        completed = subprocess.run(
            [sys.executable, relative_path],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        assert completed.returncode != 0
        assert "BLOCKED:" in completed.stderr
        assert "ModuleNotFoundError" not in completed.stderr
