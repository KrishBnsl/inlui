"""Build the model registry from current machine-readable evidence only."""

from __future__ import annotations

import csv
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def build_registry(
    results_path: Path,
    metadata_path: Path,
    output_path: Path,
) -> None:
    if not results_path.exists() or not metadata_path.exists():
        raise SystemExit(
            "BLOCKED: current metrics.json and model_metadata.json are required. "
            "Run `python -m research.pipeline.research_evaluation` as documented in "
            "docs/TECHNICAL_REPORT.md#reproducibility."
        )

    results = json.loads(results_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    selected = results["primary_model"]
    if selected != "naive_last_year" or metadata["model_family"] != "LastObservationRegressor":
        raise SystemExit("BLOCKED: selected result and serving model metadata disagree")

    selection_path = REPO_ROOT / results["model_selection"]["selection_table_path"]
    with selection_path.open(newline="", encoding="utf-8") as handle:
        selection_rows = list(csv.DictReader(handle))
    selected_row = next(row for row in selection_rows if row["model"] == selected)
    final_metrics = results["evaluation"]["final_test_metrics"]

    record = {
        "Model_ID": metadata["model_version"],
        "Type": "Deterministic baseline",
        "Target": "closing_rank",
        "Data_Cutoff": metadata["data_cutoff_year"],
        "Prediction_Year": metadata["prediction_year"],
        "Selection_Metric": "pre-final rolling-origin mean MAE",
        "Validation_Mean_MAE": selected_row["validation_mean_mae"],
        "Final_Test_Year": results["evaluation"]["final_test_year"],
        "Final_Test_MAE": final_metrics["mae"],
        "Final_Test_RMSE": final_metrics["rmse"],
        "Final_Test_R2": final_metrics["r2"],
        "Feature_Mode": metadata["feature_mode"],
        "Evidence_Status": results["status"],
        "Deployment_Status": "generated_local_assets_required",
    }
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(record), lineterminator="\n")
        writer.writeheader()
        writer.writerow(record)
    print(f"Wrote {output_path.relative_to(REPO_ROOT)} from current evidence")


def main() -> None:
    build_registry(
        REPO_ROOT / "research/reports/metrics.json",
        REPO_ROOT / "research/artifacts/model_metadata.json",
        REPO_ROOT / "research/reports/model_registry.csv",
    )


if __name__ == "__main__":
    main()
