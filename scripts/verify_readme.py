#!/usr/bin/env python3
"""Verify that SeatCraft documentation matches the executable repository contract.

This verifier intentionally uses only public examples and sanitized child
processes. It never opens a real ``.env`` file and never forwards credentials
to Docker Compose or application-import subprocesses.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
TECHNICAL_REPORT = ROOT / "docs" / "TECHNICAL_REPORT.md"
DEVELOPMENT_GUIDE = ROOT / "docs" / "DEVELOPMENT.md"
PRIMARY_MARKDOWN = (README, TECHNICAL_REPORT, DEVELOPMENT_GUIDE)
CITATION = ROOT / "CITATION.cff"

METRICS_PATH = ROOT / "research" / "reports" / "metrics.json"
MANIFEST_PATH = ROOT / "research" / "reports" / "artifact_manifest.json"
METADATA_PATH = ROOT / "research" / "artifacts" / "model_metadata.json"
DATASET_PATH = "research/data/processed/josaa_cleaned.csv"
ARTIFACT_DIR = "research/artifacts"

DEFAULT_SERVICES = {"web", "inference-api", "rag-api"}
SERVICE_PORTS = {
    "web": (3000, 3000),
    "inference-api": (8082, 8082),
    "rag-api": (8081, 8081),
}
SERVICE_CONTEXTS = {
    "web": ROOT / "apps" / "web",
    "inference-api": ROOT,
    "rag-api": ROOT / "apps" / "rag-api",
}
RUST_PROFILE = "rust-sim"

INFERENCE_ENDPOINTS = {
    ("GET", "/api/v1/health"),
    ("GET", "/api/v1/ready"),
    ("POST", "/api/v1/predict"),
    ("POST", "/api/v1/recommend"),
    ("POST", "/api/v1/chat-context"),
}
RAG_ENDPOINTS = {
    ("GET", "/health"),
    ("GET", "/ready"),
    ("POST", "/api/rag/upload"),
    ("POST", "/api/rag/ask"),
    ("GET", "/api/rag/status"),
}

REQUIRED_ARTIFACTS = {
    "pre_counselling_model.joblib",
    "inference_universe.csv",
    "pre_counselling_feature_schema.json",
    "uncertainty_calibration.json",
    "model_metadata.json",
}

COMMAND_ONLY_ENV = {
    "PLAYWRIGHT_BASE_URL",
    "RUN_LIVE_GEMINI_TEST",
    "SOURCE_DATE_EPOCH",
}

SKIP_DIRS = {
    ".git",
    ".next",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "coverage",
    "node_modules",
    "playwright-report",
    "target",
    "test-results",
}

REMOVED_DOC_NAMES = {
    "CONTRIBUTIONS.md",
    "DATA_CARD.md",
    "DEMO_SCRIPT.md",
    "EXPERIMENTS.md",
    "LIMITATIONS.md",
    "MODEL_CARD.md",
    "PROFESSOR_BRIEF.md",
    "PROJECT_AUDIT.md",
    "REPRODUCIBILITY.md",
    "RESEARCH.md",
}

OLD_PATH_PATTERNS = {
    "seatcraft/": re.compile(r"(?<![\w-])seatcraft[\\/]", re.IGNORECASE),
    "inference_service": re.compile(
        r"(?<![\w-])inference_service[\\/]|"
        r"\b(?:from|import)\s+inference_service\b"
    ),
    "rag_service": re.compile(
        r"(?<![\w-])rag_service[\\/]|"
        r"\b(?:from|import)\s+rag_service\b"
    ),
    "models/": re.compile(
        r"(?<![\w-])models[\\/](?!(?:text-embedding-004|gemini-embedding-2)\b)"
    ),
    "root reports/": re.compile(r"(?<!research[\\/])(?<![\w-])reports[\\/]"),
    "shared/": re.compile(r"(?<![\w-])shared[\\/]"),
    "sim_engine/": re.compile(r"(?<![\w-])sim_engine[\\/]"),
}

FENCE_RE = re.compile(r"```([^\n`]*)\n(.*?)```", re.DOTALL)
INLINE_LINK_RE = re.compile(
    r"!?\[([^\]]*)\]\(\s*(<[^>]+>|[^\s)]+)(?:\s+['\"][^'\"]*['\"])?\s*\)"
)
REFERENCE_DEF_RE = re.compile(r"^\s*\[([^\]]+)\]:\s*(<[^>]+>|\S+)", re.MULTILINE)
REFERENCE_USE_RE = re.compile(r"!?\[([^\]]+)\]\[([^\]]+)\]")


@dataclass(frozen=True)
class Failure:
    area: str
    message: str


class ContractVerifier:
    def __init__(self) -> None:
        self.failures: list[Failure] = []
        self.check_count = 0
        self.documents: dict[Path, str] = {}
        self.metrics: dict[str, Any] | None = None
        self.manifest: dict[str, Any] | None = None
        self.metadata: dict[str, Any] | None = None
        self.openapi: dict[str, set[tuple[str, str]]] = {}
        self.compose_default: dict[str, Any] | None = None
        self.compose_full: dict[str, Any] | None = None

    def require(self, condition: bool, area: str, message: str) -> bool:
        self.check_count += 1
        if not condition:
            self.failures.append(Failure(area, message))
        return condition

    def fail(self, area: str, message: str) -> None:
        self.require(False, area, message)

    def run(self) -> int:
        self._load_documents()
        self._check_markdown_inventory()
        self._check_citation()
        self._check_links_and_anchors()
        self._check_removed_references()
        self._load_evidence()
        self._check_metrics()
        self._check_artifacts()
        self._load_openapi_contracts()
        self._check_openapi_and_documented_endpoints()
        self._load_compose_contracts()
        self._check_compose()
        self._check_frontend_wiring()
        self._check_environment_contract()
        self._check_documented_commands()
        self._check_claim_language()
        self._check_default_stack_independence()

        if self.failures:
            print(f"README CONTRACT: FAILED ({len(self.failures)} issue(s))")
            for failure in sorted(self.failures, key=lambda item: (item.area, item.message)):
                print(f"- [{failure.area}] {failure.message}")
            return 1

        print(f"README CONTRACT: PASSED ({self.check_count} assertions)")
        return 0

    def _load_documents(self) -> None:
        for path in PRIMARY_MARKDOWN:
            if self.require(path.is_file(), "documents", f"Missing required document: {_rel(path)}"):
                try:
                    self.documents[path] = path.read_text(encoding="utf-8")
                except (OSError, UnicodeError) as exc:
                    self.fail("documents", f"Could not read {_rel(path)}: {_safe_error(exc)}")

    def _check_markdown_inventory(self) -> None:
        markdown_files: set[Path] = set()
        for directory, dirnames, filenames in os.walk(ROOT):
            dirnames[:] = [name for name in dirnames if name not in SKIP_DIRS]
            for filename in filenames:
                if not filename.casefold().endswith(".md"):
                    continue
                path = Path(directory, filename).resolve()
                relative = path.relative_to(ROOT.resolve())
                markdown_files.add(relative)

        expected = {_rel(path) for path in PRIMARY_MARKDOWN}
        actual = {path.as_posix() for path in markdown_files}
        self.require(
            actual == expected,
            "documents",
            "Repository Markdown must be exactly "
            f"{sorted(expected)}; found {sorted(actual)}",
        )

    def _check_citation(self) -> None:
        if not self.require(CITATION.is_file(), "citation", "Missing CITATION.cff"):
            return
        try:
            text = CITATION.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            self.fail("citation", f"Could not read CITATION.cff: {_safe_error(exc)}")
            return

        expected = {
            "cff-version": "1.2.0",
            "type": "software",
            "version": "0.1.0",
            "license": "MIT",
            "repository-code": "https://github.com/KrishBnsl/inlui",
            "title": (
                "SeatCraft: Uncertainty-Aware Temporal Forecasting and "
                "Retrieval-Augmented Decision Support for JoSAA Counselling"
            ),
        }
        for key, value in expected.items():
            self.require(
                _cff_scalar(text, key) == value,
                "citation",
                f"CITATION.cff must set {key} to {value!r}",
            )
        self.require(
            bool(_cff_scalar(text, "message")),
            "citation",
            "CITATION.cff must include a non-empty citation message",
        )

        author_pattern = re.compile(
            r"(?m)^[ \t]*-[ \t]+family-names:[ \t]+[\"']?([^\"'\r\n]+?)[\"']?[ \t]*\r?\n"
            r"[ \t]+given-names:[ \t]+[\"']?([^\"'\r\n]+?)[\"']?[ \t]*$"
        )
        authors = {
            (family.strip(), given.strip())
            for family, given in author_pattern.findall(text)
        }
        expected_authors = {("Bansal", "Krish"), ("Gupta", "Mehul")}
        self.require(
            authors == expected_authors,
            "citation",
            "CITATION.cff authors must be exactly Krish Bansal and Mehul Gupta",
        )

        for forbidden in ("affiliation", "date-released", "doi", "orcid"):
            self.require(
                re.search(rf"(?m)^[ \t-]*{re.escape(forbidden)}[ \t]*:", text) is None,
                "citation",
                f"CITATION.cff must not invent {forbidden} metadata",
            )

    def _check_links_and_anchors(self) -> None:
        for document, text in self.documents.items():
            definitions = {
                key.casefold(): target
                for key, target in REFERENCE_DEF_RE.findall(_without_fences(text))
            }
            links = [target for _, target in INLINE_LINK_RE.findall(_without_fences(text))]
            for _, reference in REFERENCE_USE_RE.findall(_without_fences(text)):
                target = definitions.get(reference.casefold())
                if target is None:
                    self.fail(
                        "links",
                        f"{_rel(document)} uses undefined Markdown reference [{reference}]",
                    )
                    continue
                links.append(target)

            for raw_target in links:
                target = raw_target.strip().strip("<>")
                if _is_external_link(target):
                    continue
                self._check_local_link(document, target)

    def _check_local_link(self, source: Path, target: str) -> None:
        parsed = urllib.parse.urlsplit(target)
        decoded_path = urllib.parse.unquote(parsed.path)
        fragment = urllib.parse.unquote(parsed.fragment)
        if decoded_path:
            destination = (
                ROOT / decoded_path.lstrip("/")
                if decoded_path.startswith("/")
                else source.parent / decoded_path
            ).resolve()
        else:
            destination = source.resolve()

        try:
            destination.relative_to(ROOT.resolve())
        except ValueError:
            self.fail(
                "links",
                f"{_rel(source)} link escapes the repository: {target}",
            )
            return
        if _is_secret_env_path(destination):
            self.fail(
                "links",
                f"{_rel(source)} must not link to a secret environment file: {target}",
            )
            return

        if not self.require(
            destination.exists(),
            "links",
            f"{_rel(source)} has a missing local link target: {target}",
        ):
            return

        if fragment and destination.suffix.casefold() == ".md":
            try:
                anchors = _markdown_anchors(destination.read_text(encoding="utf-8"))
            except (OSError, UnicodeError) as exc:
                self.fail("links", f"Could not inspect {_rel(destination)}: {_safe_error(exc)}")
                return
            self.require(
                fragment.casefold() in anchors,
                "links",
                f"{_rel(source)} links to missing anchor #{fragment} in {_rel(destination)}",
            )

    def _check_removed_references(self) -> None:
        for path, text in self.documents.items():
            for old_name in sorted(REMOVED_DOC_NAMES):
                self.require(
                    old_name not in text,
                    "obsolete-paths",
                    f"{_rel(path)} still references removed document {old_name}",
                )
            for old_service_name in ("inference_service", "rag_service"):
                self.require(
                    re.search(rf"\b{old_service_name}\b", text) is None,
                    "obsolete-paths",
                    f"{_rel(path)} still names removed service folder {old_service_name}",
                )

        for path in _contract_text_files():
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeError):
                continue
            for label, pattern in OLD_PATH_PATTERNS.items():
                if label == "root reports/" and _is_within(
                    path.resolve().relative_to(ROOT.resolve()),
                    Path("research/pipeline"),
                ):
                    # Research-pipeline docstrings may describe a reports/
                    # directory relative to research/. Canonical manifest and
                    # documentation paths are checked separately.
                    continue
                match = pattern.search(text)
                self.require(
                    match is None,
                    "obsolete-paths",
                    f"{_rel(path)} still contains obsolete path-shaped reference {label}",
                )

    def _load_evidence(self) -> None:
        self.metrics = self._load_json(METRICS_PATH, "metrics")
        self.manifest = self._load_json(MANIFEST_PATH, "manifest")
        self.metadata = self._load_json(METADATA_PATH, "artifacts")

    def _load_json(self, path: Path, area: str) -> dict[str, Any] | None:
        if not self.require(path.is_file(), area, f"Missing machine-readable file: {_rel(path)}"):
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeError) as exc:
            self.fail(area, f"Invalid JSON in {_rel(path)}: {_safe_error(exc)}")
            return None
        if not isinstance(payload, dict):
            self.fail(area, f"{_rel(path)} must contain a JSON object")
            return None
        return payload

    def _check_metrics(self) -> None:
        if self.metrics is None:
            return
        metrics = self.metrics
        try:
            dataset = metrics["dataset"]
            evaluation = metrics["evaluation"]
            final = evaluation["final_test_metrics"]
            selection = metrics["model_selection"]
            uncertainty = metrics["uncertainty"]
        except (KeyError, TypeError) as exc:
            self.fail("metrics", f"metrics.json is missing canonical structure: {_safe_error(exc)}")
            return

        self.require(dataset.get("path") == DATASET_PATH, "metrics", f"dataset.path must be {DATASET_PATH}")
        self.require(
            evaluation.get("fold_metrics_path") == "research/reports/research_fold_metrics.csv",
            "metrics",
            "evaluation.fold_metrics_path is stale",
        )
        self.require(
            evaluation.get("ablation_metrics_path")
            == "research/reports/research_ablation_metrics.csv",
            "metrics",
            "evaluation.ablation_metrics_path is stale",
        )
        self.require(
            evaluation.get("subgroup_metrics_path")
            == "research/reports/research_subgroup_metrics.csv",
            "metrics",
            "evaluation.subgroup_metrics_path is stale",
        )
        self.require(
            selection.get("selection_table_path")
            == "research/reports/research_model_selection.csv",
            "metrics",
            "model_selection.selection_table_path is stale",
        )
        self.require(
            metrics.get("primary_model") == selection.get("selected_model") == final.get("model"),
            "metrics",
            "Primary, selected, and final-test model names disagree",
        )
        self.require(
            evaluation.get("final_test_year") == final.get("year"),
            "metrics",
            "Final-test year fields disagree",
        )
        self.require(
            uncertainty.get("final_test_residuals_used_for_interval") is False,
            "metrics",
            "Final-test residuals must not be used for interval construction",
        )
        self.require(
            _same_number(
                uncertainty.get("final_test_empirical_coverage"),
                final.get("prediction_interval_coverage"),
            ),
            "metrics",
            "Uncertainty and final-test interval coverage disagree",
        )
        self.require(
            uncertainty.get("final_test_calibration_count") == final.get("calibration_count"),
            "metrics",
            "Uncertainty and final-test calibration counts disagree",
        )

        readme = self.documents.get(README)
        technical = self.documents.get(TECHNICAL_REPORT)
        if readme is not None:
            self._check_final_metric_tokens(readme, "README.md", final, metrics)
            self.require(
                "research/reports/metrics.json" in readme,
                "metrics",
                "README.md must link to the canonical metrics.json",
            )
        if technical is not None:
            self._check_final_metric_tokens(technical, "docs/TECHNICAL_REPORT.md", final, metrics)
            self.require(
                "research/reports/metrics.json" in technical,
                "metrics",
                "TECHNICAL_REPORT.md must link to the canonical metrics.json",
            )
            dataset_hash = str(dataset.get("sha256", ""))
            self.require(
                bool(dataset_hash) and dataset_hash in technical,
                "metrics",
                "TECHNICAL_REPORT.md must include the canonical dataset SHA-256",
            )
            dataset_years = dataset.get("years", [])
            if isinstance(dataset_years, list) and dataset_years:
                self.require(
                    _contains_year_range(technical, min(dataset_years), max(dataset_years)),
                    "metrics",
                    "TECHNICAL_REPORT.md does not state the canonical dataset year range",
                )
            selection_years = selection.get("selection_years", [])
            if isinstance(selection_years, list) and selection_years:
                self.require(
                    _contains_year_range(technical, min(selection_years), max(selection_years)),
                    "metrics",
                    "TECHNICAL_REPORT.md does not state the canonical model-selection years",
                )
            self._check_selection_metric_tokens(technical, selection)

        self._check_extended_metric_contract(
            metrics,
            readme or "",
            technical or "",
        )

    def _check_extended_metric_contract(
        self,
        metrics: dict[str, Any],
        readme: str,
        technical: str,
    ) -> None:
        """Protect every professor-facing generated evidence family from drift."""

        self.require(metrics.get("schema_version") == 2, "metrics", "metrics.json schema_version must be 2")
        experiment_id = metrics.get("experiment_id")
        self.require(
            isinstance(experiment_id, str) and bool(experiment_id),
            "metrics",
            "metrics.json must contain a non-empty experiment_id",
        )
        if isinstance(experiment_id, str):
            for label, text in (("README.md", readme), ("docs/TECHNICAL_REPORT.md", technical)):
                self.require(
                    experiment_id in text,
                    "metrics",
                    f"{label} does not contain the canonical experiment_id",
                )

        dataset = metrics.get("dataset", {})
        dataset_version = dataset.get("version") if isinstance(dataset, dict) else None
        self.require(
            isinstance(dataset_version, str) and bool(dataset_version),
            "metrics",
            "metrics.json must contain dataset.version",
        )
        if isinstance(dataset_version, str):
            for label, text in (("README.md", readme), ("docs/TECHNICAL_REPORT.md", technical)):
                self.require(
                    dataset_version in text,
                    "metrics",
                    f"{label} does not contain the canonical dataset version",
                )

        self.require(bool(metrics.get("models")), "metrics", "metrics.json must contain model specifications")
        self.require(bool(metrics.get("code_hashes")), "metrics", "metrics.json must contain experiment code hashes")

        selection_rows = metrics.get("model_selection", {}).get("selection_table", [])
        final_rows = metrics.get("evaluation", {}).get("final_test_model_comparison", [])
        self.require(isinstance(selection_rows, list) and bool(selection_rows), "metrics", "Embedded selection table is missing")
        self.require(isinstance(final_rows, list) and bool(final_rows), "metrics", "Embedded final comparison is missing")
        if isinstance(selection_rows, list):
            for row in selection_rows:
                if not isinstance(row, dict):
                    continue
                value = row.get("validation_mean_mae")
                model = row.get("model", "unknown")
                for label, text in (("README.md", readme), ("docs/TECHNICAL_REPORT.md", technical)):
                    self.require(
                        _contains_formatted_number(text, value, 2),
                        "metrics",
                        f"{label} omits validation MAE for {model}",
                    )
        if isinstance(final_rows, list):
            for row in final_rows:
                if not isinstance(row, dict):
                    continue
                model = row.get("model", "unknown")
                for key, decimals in (("mae", 2), ("rmse", 2), ("median_absolute_error", 2), ("r2", 5)):
                    for label, text in (("README.md", readme), ("docs/TECHNICAL_REPORT.md", technical)):
                        self.require(
                            _contains_formatted_number(text, row.get(key), decimals),
                            "metrics",
                            f"{label} omits final {key} for {model}",
                        )

        ablations = metrics.get("ablation_results", [])
        self.require(isinstance(ablations, list) and bool(ablations), "metrics", "Canonical ablation results are missing")
        if isinstance(ablations, list):
            for row in ablations:
                if not isinstance(row, dict) or row.get("mae") is None:
                    continue
                variant = row.get("variant", "unknown")
                for label, text in (("README.md", readme), ("docs/TECHNICAL_REPORT.md", technical)):
                    self.require(
                        _contains_formatted_number(text, row.get("mae"), 2),
                        "metrics",
                        f"{label} omits ablation MAE for {variant}",
                    )

        in_round = metrics.get("in_round_diagnostic", [])
        self.require(isinstance(in_round, list) and len(in_round) == 2, "metrics", "Canonical in-round comparison must have two rows")
        if isinstance(in_round, list):
            for row in in_round:
                if not isinstance(row, dict):
                    continue
                variant = row.get("variant", "unknown")
                self.require(
                    _contains_formatted_number(readme, row.get("mae"), 2),
                    "metrics",
                    f"README.md omits in-round MAE for {variant}",
                )
                for key, decimals in (("mae", 2), ("rmse", 2), ("median_absolute_error", 2), ("r2", 5)):
                    self.require(
                        _contains_formatted_number(technical, row.get(key), decimals),
                        "metrics",
                        f"TECHNICAL_REPORT.md omits in-round {key} for {variant}",
                    )

        figures = metrics.get("figures", [])
        self.require(isinstance(figures, list) and len(figures) == 6, "metrics", "Exactly six canonical strict figures are required")
        if isinstance(figures, list):
            canonical_figure_paths: set[str] = set()
            for entry in figures:
                if not isinstance(entry, dict):
                    continue
                path = entry.get("path")
                if isinstance(path, str):
                    canonical_figure_paths.add(path)
                self.require(
                    isinstance(path, str) and path in readme,
                    "metrics",
                    f"README.md does not embed canonical figure {path}",
                )
            primary_figure_dir = ROOT / "research" / "reports" / "figures"
            actual_figure_paths = {
                _rel(path)
                for path in primary_figure_dir.glob("*.png")
                if path.is_file()
            }
            self.require(
                actual_figure_paths == canonical_figure_paths,
                "metrics",
                "Primary figure directory contains non-canonical or missing plots",
            )

        case_study = metrics.get("case_study", {})
        self.require(
            isinstance(case_study, dict) and case_study.get("status") == "generated_from_real_inference_api",
            "metrics",
            "Canonical case study was not generated through the real inference API",
        )
        if isinstance(case_study, dict):
            for row in case_study.get("selected_recommendations", []):
                if not isinstance(row, dict):
                    continue
                institute_tail = str(row.get("institute_name", "")).split()[-1:]
                if institute_tail:
                    self.require(institute_tail[0] in readme, "metrics", f"README omits case-study institute {institute_tail[0]}")
                self.require(str(row.get("program_name", "")) in readme, "metrics", "README omits a case-study programme")
                for key, decimals, percentage in (
                    ("rank_used", 0, False),
                    ("projected_closing_rank", 0, False),
                    ("uncertainty_lower", 0, False),
                    ("uncertainty_upper", 0, False),
                    ("model_probability_percent", 1, False),
                    ("recommendation_score", 4, False),
                    ("safety_margin", 0, False),
                ):
                    value = row.get(key)
                    if key == "safety_margin" and isinstance(value, (int, float)):
                        value = abs(value)
                    self.require(
                        _contains_formatted_number(readme, value, decimals, percentage=percentage),
                        "metrics",
                        f"README omits case-study field {key}",
                    )

    def _check_final_metric_tokens(
        self,
        text: str,
        label: str,
        final: dict[str, Any],
        metrics: dict[str, Any],
    ) -> None:
        requirements = [
            ("matched rows", final.get("matched_rows", final.get("n")), 0, False),
            ("MAE", final.get("mae"), 2, False),
            ("RMSE", final.get("rmse"), 2, False),
            ("median absolute error", final.get("median_absolute_error"), 0, False),
            ("R-squared", final.get("r2"), 5, False),
            ("prediction-interval coverage", final.get("prediction_interval_coverage"), 2, True),
            ("universe coverage", final.get("observed_universe_coverage"), 2, True),
            ("prediction-interval mean width", final.get("prediction_interval_mean_width"), 2, False),
        ]
        for name, value, decimals, percentage in requirements:
            self.require(
                _contains_formatted_number(text, value, decimals, percentage=percentage),
                "metrics",
                f"{label} does not contain canonical {name}",
            )
        final_year = final.get("year")
        self.require(
            isinstance(final_year, int) and re.search(rf"\b{final_year}\b", text) is not None,
            "metrics",
            f"{label} does not identify the final-test year",
        )
        primary_model = str(metrics.get("primary_model", ""))
        model_pattern = r"naive[_ -]last[_ -]year|last[- ]year"
        self.require(
            bool(primary_model) and re.search(model_pattern, text, re.IGNORECASE) is not None,
            "metrics",
            f"{label} does not identify the selected last-year model",
        )

    def _check_selection_metric_tokens(self, technical: str, selection: dict[str, Any]) -> None:
        raw_path = selection.get("selection_table_path")
        if not isinstance(raw_path, str):
            return
        path = ROOT / raw_path
        if not self.require(path.is_file(), "metrics", f"Missing selection table: {raw_path}"):
            return
        try:
            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
        except (OSError, UnicodeError, csv.Error) as exc:
            self.fail("metrics", f"Could not read {raw_path}: {_safe_error(exc)}")
            return
        self.require(bool(rows), "metrics", f"Selection table is empty: {raw_path}")
        for row in rows:
            model = row.get("model", "unknown")
            try:
                value = float(row["validation_mean_mae"])
            except (KeyError, TypeError, ValueError):
                self.fail("metrics", f"Selection row {model} has no numeric validation_mean_mae")
                continue
            self.require(
                _contains_formatted_number(technical, value, 2),
                "metrics",
                f"TECHNICAL_REPORT.md omits selection MAE for {model}",
            )

    def _check_artifacts(self) -> None:
        if self.manifest is None:
            return
        manifest = self.manifest
        serving = manifest.get("generated_serving_artifacts", [])
        source_assets = manifest.get("source_assets", [])
        outputs = manifest.get("versioned_outputs", [])
        if not isinstance(serving, list) or not isinstance(source_assets, list) or not isinstance(outputs, list):
            self.fail("artifacts", "artifact_manifest.json asset collections must be lists")
            return

        serving_paths = {
            str(entry.get("path"))
            for entry in serving
            if isinstance(entry, dict)
        }
        expected_paths = {f"{ARTIFACT_DIR}/{name}" for name in REQUIRED_ARTIFACTS}
        self.require(
            expected_paths <= serving_paths,
            "artifacts",
            f"Manifest is missing serving artifacts: {sorted(expected_paths - serving_paths)}",
        )

        source_paths = {
            str(entry.get("path"))
            for entry in source_assets
            if isinstance(entry, dict)
        }
        self.require(
            DATASET_PATH in source_paths,
            "artifacts",
            f"Manifest source_assets must contain {DATASET_PATH}",
        )
        output_paths = {
            str(entry.get("path"))
            for entry in outputs
            if isinstance(entry, dict)
        }
        self.require(
            "research/reports/metrics.json" in output_paths,
            "artifacts",
            "Manifest versioned_outputs must contain research/reports/metrics.json",
        )

        for collection_name, entries in (
            ("generated_serving_artifacts", serving),
            ("source_assets", source_assets),
            ("versioned_outputs", outputs),
        ):
            allowed_root = {
                "generated_serving_artifacts": ROOT / "research" / "artifacts",
                "source_assets": ROOT / "research" / "data",
                "versioned_outputs": ROOT / "research" / "reports",
            }[collection_name]
            for entry in entries:
                if not isinstance(entry, dict):
                    self.fail("artifacts", f"{collection_name} contains a non-object entry")
                    continue
                raw_path = entry.get("path")
                if not isinstance(raw_path, str):
                    self.fail("artifacts", f"{collection_name} contains an entry without path")
                    continue
                self.require(
                    not _contains_old_repository_path(raw_path),
                    "artifacts",
                    f"Manifest contains stale path: {raw_path}",
                )
                path = (ROOT / raw_path).resolve()
                try:
                    path.relative_to(allowed_root.resolve())
                    path_is_allowed = not _is_secret_env_path(path)
                except ValueError:
                    path_is_allowed = False
                if not self.require(
                    path_is_allowed,
                    "artifacts",
                    f"Manifest {collection_name} path escapes its allowed directory: {raw_path}",
                ):
                    continue
                should_exist = collection_name == "versioned_outputs" or entry.get("committed_expected") is True
                if should_exist:
                    exists = self.require(path.is_file(), "artifacts", f"Manifest file is missing: {raw_path}")
                    expected_hash = entry.get("sha256")
                    hash_is_valid = isinstance(expected_hash, str) and re.fullmatch(
                        r"[0-9a-fA-F]{64}", expected_hash
                    ) is not None
                    self.require(
                        hash_is_valid,
                        "artifacts",
                        f"Manifest has no valid SHA-256 for {raw_path}",
                    )
                    if exists and hash_is_valid:
                        self.require(
                            _sha256_file(path).casefold() == expected_hash.casefold(),
                            "artifacts",
                            f"Manifest SHA-256 does not match {raw_path}",
                        )

        loader = ROOT / "apps" / "inference-api" / "src" / "inference_api" / "services" / "artifact_loader.py"
        if self.require(loader.is_file(), "artifacts", f"Missing artifact loader: {_rel(loader)}"):
            loader_text = loader.read_text(encoding="utf-8")
            for filename in sorted(REQUIRED_ARTIFACTS):
                self.require(
                    filename in loader_text,
                    "artifacts",
                    f"Artifact loader does not name required file {filename}",
                )

        combined_docs = "\n".join(self.documents.values())
        self.require(DATASET_PATH in combined_docs, "artifacts", f"Documentation must name {DATASET_PATH}")
        for filename in sorted(REQUIRED_ARTIFACTS):
            self.require(
                filename in combined_docs,
                "artifacts",
                f"Documentation does not name required artifact {filename}",
            )

        if self.metadata is not None and self.metrics is not None:
            source_commit = self.metrics.get("source_commit")
            self.require(
                isinstance(source_commit, str)
                and re.fullmatch(r"[0-9a-f]{40}", source_commit) is not None,
                "artifacts",
                "metrics.json has no valid committed source snapshot",
            )
            for label, payload in (
                ("metrics.json", self.metrics),
                ("artifact_manifest.json", manifest),
                ("model_metadata.json", self.metadata),
            ):
                self.require(
                    payload.get("source_commit") == source_commit
                    and payload.get("generated_from_commit") == source_commit,
                    "artifacts",
                    f"{label} does not identify the canonical source snapshot",
                )
                source_dirty = payload.get("source_worktree_dirty")
                self.require(
                    isinstance(source_dirty, bool)
                    and payload.get("worktree_dirty") == source_dirty,
                    "artifacts",
                    f"{label} has invalid source-worktree metadata",
                )

            metric_hash = str(self.metrics.get("dataset", {}).get("sha256", ""))
            manifest_hashes = {
                str(entry.get("sha256"))
                for entry in source_assets
                if isinstance(entry, dict) and entry.get("path") == DATASET_PATH
            }
            self.require(
                bool(metric_hash)
                and self.metadata.get("dataset_sha256") == metric_hash
                and metric_hash in manifest_hashes,
                "artifacts",
                "Dataset SHA-256 disagrees across metrics, metadata, and manifest",
            )
            metadata_artifacts = self.metadata.get("artifacts", {})
            self.require(
                isinstance(metadata_artifacts, dict)
                and (REQUIRED_ARTIFACTS - {"model_metadata.json"}) <= set(metadata_artifacts),
                "artifacts",
                "model_metadata.json does not enumerate the serving bundle",
            )
            cutoff_year = self.metrics.get("dataset", {}).get("data_cutoff_year")
            prediction_year = self.metadata.get("prediction_year")
            self.require(
                self.metadata.get("data_cutoff_year") == cutoff_year,
                "artifacts",
                "Model metadata and metrics disagree on data cutoff year",
            )
            self.require(
                isinstance(prediction_year, int)
                and isinstance(cutoff_year, int)
                and prediction_year > cutoff_year,
                "artifacts",
                "Model metadata prediction year must follow the data cutoff",
            )
            for path in (README, TECHNICAL_REPORT):
                text = self.documents.get(path)
                if text is None:
                    continue
                self.require(
                    isinstance(prediction_year, int)
                    and re.search(rf"\b{prediction_year}\b", text) is not None,
                    "artifacts",
                    f"{_rel(path)} does not state the serving prediction year",
                )

    def _load_openapi_contracts(self) -> None:
        for package in ("inference_api", "rag_api"):
            contract = _installed_openapi_contract(package)
            if isinstance(contract, str):
                self.fail("openapi", f"Could not inspect installed {package}: {contract}")
            else:
                self.openapi[package] = contract

    def _check_openapi_and_documented_endpoints(self) -> None:
        inference = self.openapi.get("inference_api")
        rag = self.openapi.get("rag_api")
        if inference is not None:
            self.require(
                INFERENCE_ENDPOINTS <= inference,
                "openapi",
                f"Inference API is missing endpoints: {sorted(INFERENCE_ENDPOINTS - inference)}",
            )
        if rag is not None:
            self.require(
                RAG_ENDPOINTS <= rag,
                "openapi",
                f"RAG API is missing endpoints: {sorted(RAG_ENDPOINTS - rag)}",
            )
        if inference is None or rag is None:
            return

        union_paths = {path for _, path in inference | rag}
        for document, text in self.documents.items():
            for endpoint in sorted(_documented_endpoints(text)):
                self.require(
                    endpoint in union_paths,
                    "openapi",
                    f"{_rel(document)} documents endpoint not present in OpenAPI: {endpoint}",
                )

        readme = self.documents.get(README, "")
        for _, endpoint in sorted(INFERENCE_ENDPOINTS | RAG_ENDPOINTS):
            if endpoint in {"/api/v1/ready", "/api/v1/recommend", "/api/v1/chat-context", "/ready", "/api/rag/status"}:
                continue
            self.require(
                endpoint in readme,
                "openapi",
                f"README.md does not document primary endpoint {endpoint}",
            )

    def _load_compose_contracts(self) -> None:
        default = _compose_contract(profile=None)
        if isinstance(default, str):
            self.fail("compose", f"Could not inspect default Compose config: {default}")
        else:
            self.compose_default = default
        full = _compose_contract(profile=RUST_PROFILE)
        if isinstance(full, str):
            self.fail("compose", f"Could not inspect {RUST_PROFILE} Compose config: {full}")
        else:
            self.compose_full = full

    def _check_compose(self) -> None:
        if self.compose_default is None:
            return
        services = self.compose_default.get("services", {})
        if not isinstance(services, dict):
            self.fail("compose", "Compose config has no services object")
            return
        self.require(
            set(services) == DEFAULT_SERVICES,
            "compose",
            f"Default services must be {sorted(DEFAULT_SERVICES)}; got {sorted(services)}",
        )

        for service_name in sorted(DEFAULT_SERVICES & set(services)):
            service = services[service_name]
            if not isinstance(service, dict):
                self.fail("compose", f"Service {service_name} config is not an object")
                continue
            published, target = SERVICE_PORTS[service_name]
            self.require(
                _has_port(service, published, target),
                "compose",
                f"Service {service_name} must publish {published}:{target}",
            )
            context = _build_context(service)
            expected_context = SERVICE_CONTEXTS[service_name].resolve()
            self.require(
                context == expected_context,
                "compose",
                f"Service {service_name} build context must be {_rel(expected_context)}; got "
                f"{_display_path(context)}",
            )
            self.require(
                not service.get("profiles"),
                "compose",
                f"Primary service {service_name} must not require a profile",
            )
            serialized = json.dumps(service, sort_keys=True).casefold()
            self.require(
                "optional/rust-sim" not in serialized
                and "database_url" not in serialized
                and "postgres://" not in serialized,
                "compose",
                f"Primary service {service_name} depends on optional Rust/PostgreSQL configuration",
            )

        web = services.get("web")
        if isinstance(web, dict):
            dependencies = set((web.get("depends_on") or {}).keys())
            self.require(
                dependencies == {"inference-api", "rag-api"},
                "compose",
                "web must depend only on inference-api and rag-api health",
            )

        if self.compose_full is None:
            return
        full_services = self.compose_full.get("services", {})
        if not isinstance(full_services, dict):
            self.fail("compose", "Profiled Compose config has no services object")
            return
        optional_services = set(full_services) - DEFAULT_SERVICES
        self.require(bool(optional_services), "compose", "rust-sim profile contains no optional services")
        for name in sorted(optional_services):
            service = full_services.get(name, {})
            profiles = set(service.get("profiles") or []) if isinstance(service, dict) else set()
            self.require(
                RUST_PROFILE in profiles,
                "compose",
                f"Optional service {name} is not guarded by profile {RUST_PROFILE}",
            )
        self.require(
            any(
                _build_context(full_services[name]) == (ROOT / "optional" / "rust-sim").resolve()
                for name in optional_services
                if isinstance(full_services.get(name), dict) and full_services[name].get("build")
            ),
            "compose",
            "rust-sim profile does not build from optional/rust-sim",
        )

        readme = self.documents.get(README, "")
        for service_name in DEFAULT_SERVICES:
            self.require(
                service_name in readme,
                "compose",
                f"README.md does not name Compose service {service_name}",
            )
        for port in (3000, 8081, 8082):
            self.require(
                f"localhost:{port}" in readme or f"127.0.0.1:{port}" in readme,
                "compose",
                f"README.md does not document default port {port}",
            )

    def _check_frontend_wiring(self) -> None:
        ml_path = ROOT / "apps" / "web" / "src" / "lib" / "api.ts"
        rag_path = ROOT / "apps" / "web" / "src" / "lib" / "rag-api.ts"
        if not self.require(ml_path.is_file(), "frontend", f"Missing {_rel(ml_path)}"):
            return
        if not self.require(rag_path.is_file(), "frontend", f"Missing {_rel(rag_path)}"):
            return
        ml = ml_path.read_text(encoding="utf-8")
        rag = rag_path.read_text(encoding="utf-8")
        self.require("process.env.NEXT_PUBLIC_ML_URL" in ml, "frontend", "ML client does not use NEXT_PUBLIC_ML_URL")
        self.require("http://localhost:8082" in ml, "frontend", "ML client default is not localhost:8082")
        for endpoint in ("/api/v1/predict", "/api/v1/health", "/api/v1/chat-context"):
            self.require(endpoint in ml, "frontend", f"ML client does not call {endpoint}")
        self.require("process.env.NEXT_PUBLIC_RAG_URL" in rag, "frontend", "RAG client does not use NEXT_PUBLIC_RAG_URL")
        self.require("http://localhost:8081" in rag, "frontend", "RAG client default is not localhost:8081")
        for endpoint in ("/health", "/api/rag/upload", "/api/rag/ask", "/api/rag/status"):
            self.require(endpoint in rag, "frontend", f"RAG client does not call {endpoint}")

        frontend_source = "\n".join(
            path.read_text(encoding="utf-8")
            for path in (ROOT / "apps" / "web" / "src").rglob("*")
            if path.is_file() and path.suffix in {".ts", ".tsx"}
        )
        for forbidden in ("NEXT_PUBLIC_API_URL", "sim_engine", "optional/rust-sim", "localhost:8080"):
            self.require(
                forbidden not in frontend_source,
                "frontend",
                f"Normal frontend source still references forbidden Rust/legacy wiring: {forbidden}",
            )

    def _check_environment_contract(self) -> None:
        example = ROOT / ".env.example"
        if not self.require(example.is_file(), "environment", "Missing root .env.example"):
            return
        try:
            example_text = example.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            self.fail("environment", f"Could not read .env.example: {_safe_error(exc)}")
            return
        names = re.findall(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=", example_text, re.MULTILINE)
        unique_names = set(names)
        self.require(len(names) == len(unique_names), "environment", ".env.example contains duplicate names")
        self.require("GOOGLE_API_KEY" in unique_names, "environment", ".env.example omits GOOGLE_API_KEY")
        self.require("NEXT_PUBLIC_ML_URL" in unique_names, "environment", ".env.example omits NEXT_PUBLIC_ML_URL")
        self.require("NEXT_PUBLIC_RAG_URL" in unique_names, "environment", ".env.example omits NEXT_PUBLIC_RAG_URL")
        self.require("NEXT_PUBLIC_API_URL" not in unique_names, "environment", ".env.example contains obsolete NEXT_PUBLIC_API_URL")

        compose_source = ROOT / "docker-compose.yml"
        compose_text = compose_source.read_text(encoding="utf-8") if compose_source.is_file() else ""
        compose_names = set(re.findall(r"\$\{([A-Za-z_][A-Za-z0-9_]*)", compose_text))
        self.require(
            compose_names <= unique_names,
            "environment",
            f"Compose variables missing from .env.example: {sorted(compose_names - unique_names)}",
        )

        frontend_text = "\n".join(
            path.read_text(encoding="utf-8")
            for path in (ROOT / "apps" / "web" / "src").rglob("*")
            if path.is_file() and path.suffix in {".ts", ".tsx"}
        )
        frontend_names = set(re.findall(r"process\.env\.([A-Za-z_][A-Za-z0-9_]*)", frontend_text))
        self.require(
            frontend_names <= unique_names,
            "environment",
            f"Frontend variables missing from .env.example: {sorted(frontend_names - unique_names)}",
        )

        readme = self.documents.get(README, "")
        env_section = _heading_section(readme, "environment")
        self.require(bool(env_section), "environment", "README.md needs an Environment variables section")
        combined = "\n".join(self.documents.values())
        for name in sorted(unique_names):
            self.require(
                re.search(rf"\b{re.escape(name)}\b", combined) is not None,
                "environment",
                f"Root environment variable {name} is not documented",
            )
            if not name.startswith("RUST_"):
                self.require(
                    re.search(rf"\b{re.escape(name)}\b", env_section) is not None,
                    "environment",
                    f"Primary environment variable {name} is absent from README's environment section",
                )

        documented_env = set(re.findall(r"\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+\b", env_section))
        unknown = documented_env - unique_names - COMMAND_ONLY_ENV - {"JEE_MAIN", "JEE_ADVANCED"}
        self.require(
            not unknown,
            "environment",
            f"README environment section names variables absent from .env.example: {sorted(unknown)}",
        )

    def _check_documented_commands(self) -> None:
        package_path = ROOT / "apps" / "web" / "package.json"
        package: dict[str, Any] = {}
        if self.require(package_path.is_file(), "commands", f"Missing {_rel(package_path)}"):
            try:
                package = json.loads(package_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                self.fail("commands", f"Invalid apps/web/package.json: {_safe_error(exc)}")
        scripts = package.get("scripts", {}) if isinstance(package, dict) else {}

        command_blocks: list[tuple[Path, str]] = []
        for document, text in self.documents.items():
            for language, block in FENCE_RE.findall(text):
                if language.strip().casefold().split(" ", 1)[0] in {"bash", "sh", "shell", "console", "zsh"}:
                    command_blocks.append((document, block))

        combined_commands = "\n".join(block for _, block in command_blocks)
        readme_commands = "\n".join(
            block for document, block in command_blocks if document == README
        )
        self.require("scripts/verify_readme.py" in readme_commands, "commands", "README does not document scripts/verify_readme.py")
        self.require("scripts/verify_project.py" in readme_commands, "commands", "README does not document scripts/verify_project.py")
        self.require(
            re.search(r"docker\s+compose\s+up\s+--build|docker\s+compose\s+--profile\s+\S+\s+up\s+--build", readme_commands) is not None,
            "commands",
            "README does not document default docker compose up --build",
        )
        self.require(RUST_PROFILE in combined_commands, "commands", "Optional rust-sim command is not documented")

        for document, raw_block in command_blocks:
            block = re.sub(r"\\\s*\n\s*", " ", raw_block)
            if re.search(r"\bnpm\s+(?:ci|install|test|audit|run)\b", block):
                self.require(
                    "apps/web" in block or re.search(r"npm\s+--prefix\s+apps/web\b", block) is not None,
                    "commands",
                    f"{_rel(document)} has an npm command without an explicit apps/web context",
                )
            for match in re.finditer(
                r"(?:^|[;&|]\s*|\s)(?:python(?:3)?|\.venv/bin/python)\s+(?!-m\b)([^\s\\]+\.py)",
                block,
                re.MULTILINE,
            ):
                raw = match.group(1).strip("'\"")
                path = ROOT / raw
                self.require(path.is_file(), "commands", f"{_rel(document)} references missing script {raw}")
            for raw in re.findall(r"(?:^|\s)-r\s+([^\s\\]+)", block):
                cleaned = raw.strip("'\"")
                self.require((ROOT / cleaned).is_file(), "commands", f"{_rel(document)} references missing requirements file {cleaned}")
            for invocation in re.findall(
                r"(?:python(?:3)?|\.venv/bin/python)\s+-m\s+pytest\s+([^\n;&|]+)|"
                r"(?:^|\s)(?:pytest|\.venv/bin/pytest)\s+([^\n;&|]+)",
                block,
                re.MULTILINE,
            ):
                arguments = next((value for value in invocation if value), "")
                for token in re.findall(r"(?:^|\s)([^\s]+)", arguments):
                    cleaned = token.strip("'\"")
                    if cleaned.startswith("-") or cleaned in {"and", "or"}:
                        continue
                    if cleaned.startswith(("apps/", "research/", "packages/", "scripts/")):
                        test_path = ROOT / cleaned.split("::", 1)[0]
                        self.require(
                            test_path.exists(),
                            "commands",
                            f"{_rel(document)} references missing pytest target {cleaned}",
                        )
            for raw in re.findall(r"(?:^|[;&|]\s*|\s)cd\s+([^\s;&|\\]+)", block):
                cleaned = raw.strip("'\"")
                if cleaned in {"..", "-", "inlui"} or cleaned.startswith("$HOME"):
                    continue
                self.require((ROOT / cleaned).is_dir(), "commands", f"{_rel(document)} references missing directory {cleaned}")
            for script_name in re.findall(r"\bnpm\s+run\s+([A-Za-z0-9:_-]+)", block):
                self.require(script_name in scripts, "commands", f"{_rel(document)} references undefined npm script {script_name}")
            if re.search(r"\bnpm\s+test\b", block):
                self.require("test" in scripts, "commands", f"{_rel(document)} documents npm test but package.json has no test script")
            for profile in re.findall(r"docker\s+compose\s+--profile\s+([A-Za-z0-9_.-]+)", block):
                self.require(profile == RUST_PROFILE, "commands", f"{_rel(document)} documents unknown Compose profile {profile}")
            for module in re.findall(r"(?:uvicorn|python\s+-m\s+uvicorn)\s+([A-Za-z_][\w.]*):app", block):
                self.require(
                    module in {"inference_api.main", "rag_api.main"},
                    "commands",
                    f"{_rel(document)} documents obsolete Uvicorn module {module}:app",
                )

        for required_script in ("test", "lint", "build", "test:e2e"):
            self.require(
                re.search(rf"\bnpm\s+(?:run\s+)?{re.escape(required_script)}\b", combined_commands) is not None,
                "commands",
                f"Documentation omits npm command {required_script}",
            )

    def _check_claim_language(self) -> None:
        for path, text in self.documents.items():
            lowered = text.casefold()
            for phrase in (
                "oof residual distribution",
                "empirical residual distribution",
                "statistical admission probability",
                "statistical admission-probability",
                "untouched 2025",
            ):
                self.require(
                    phrase not in lowered,
                    "claims",
                    f"{_rel(path)} contains misleading phrase: {phrase}",
                )
            for paragraph in re.split(r"\n\s*\n", lowered):
                if "confidence interval" in paragraph:
                    self.require(
                        re.search(
                            r"\b(?:not|never)\s+(?:a\s+)?confidence intervals?\b|"
                            r"\brather than\s+(?:a\s+)?confidence intervals?\b",
                            paragraph,
                        )
                        is not None,
                        "claims",
                        f"{_rel(path)} uses confidence-interval wording without an explicit rejection",
                    )
                positive_guarantee = re.search(
                    r"\b(?:seatcraft|the model|the system|the interval|this model|this system|"
                    r"this interval)\b[^.\n]{0,120}\bguarantee(?:d|s)?\b|"
                    r"\bguarantee(?:d|s)?\s+(?:admission|coverage|results?)\b|"
                    r"\b(?:admission|coverage|results?)\s+(?:is|are)\s+guaranteed\b",
                    paragraph,
                )
                if positive_guarantee:
                    self.require(
                        re.search(
                            r"\b(?:not|no|never|cannot|without|lacks?|unavailable|false|"
                            r"excludes?|rejects?)\b",
                            paragraph,
                        )
                        is not None,
                        "claims",
                        f"{_rel(path)} uses guarantee wording without an explicit limitation",
                    )
                if re.search(r"\b(?:empirically\s+)?calibrated\s+(?:admission[- ]?)?probabilit", paragraph):
                    self.require(
                        re.search(r"\b(?:not|un|never|future|reserved)\w*\b", paragraph) is not None,
                        "claims",
                        f"{_rel(path)} describes probability as calibrated without a qualification",
                    )

        for path in (README, TECHNICAL_REPORT):
            text = self.documents.get(path)
            if text is None:
                continue
            lowered = text.casefold()
            self.require("normal approximation" in lowered, "claims", f"{_rel(path)} must disclose the normal approximation")
            self.require(
                "not calibrated" in lowered or "uncalibrated" in lowered,
                "claims",
                f"{_rel(path)} must state that modeled probability is uncalibrated",
            )
            self.require("empirical prediction interval" in lowered, "claims", f"{_rel(path)} must label the empirical prediction interval")
            self.require("not an admission guarantee" in lowered, "claims", f"{_rel(path)} must retain the admission disclaimer")

        readme = self.documents.get(README, "").casefold()
        self.require(
            re.search(r"required[^\n]{0,80}next_public_(?:ml|rag)_url", readme) is None,
            "claims",
            "README must describe NEXT_PUBLIC service URLs as optional build-time overrides, not required variables",
        )

    def _check_default_stack_independence(self) -> None:
        primary_files = [
            ROOT / "requirements-dev.txt",
            ROOT / "apps" / "web" / "package.json",
            ROOT / "apps" / "web" / "Dockerfile",
            ROOT / "apps" / "inference-api" / "Dockerfile",
            ROOT / "apps" / "rag-api" / "Dockerfile",
            ROOT / "apps" / "inference-api" / "pyproject.toml",
            ROOT / "apps" / "rag-api" / "pyproject.toml",
            ROOT / "packages" / "josaa-core" / "pyproject.toml",
        ]
        forbidden = ("cargo", "postgres", "database_url", "optional/rust-sim")
        for path in primary_files:
            if not self.require(path.is_file(), "rust-boundary", f"Missing primary configuration {_rel(path)}"):
                continue
            text = path.read_text(encoding="utf-8").casefold()
            for term in forbidden:
                self.require(
                    term not in text,
                    "rust-boundary",
                    f"Primary configuration {_rel(path)} depends on optional stack term {term}",
                )

        if self.compose_default is not None:
            serialized = json.dumps(self.compose_default.get("services", {}), sort_keys=True).casefold()
            for term in ("postgres", "cargo", "database_url", "optional/rust-sim", ":8080"):
                self.require(
                    term not in serialized,
                    "rust-boundary",
                    f"Default Compose config contains optional Rust/PostgreSQL term {term}",
                )


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _display_path(path: Path | None) -> str:
    return "missing" if path is None else _rel(path)


def _cff_scalar(text: str, key: str) -> str | None:
    """Read one top-level scalar from the deliberately simple CFF document."""
    match = re.search(rf"(?m)^{re.escape(key)}[ \t]*:[ \t]*(.+?)[ \t]*$", text)
    if match is None:
        return None
    value = match.group(1).strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    return value


def _safe_error(error: BaseException | str) -> str:
    text = str(error).replace("\n", " ")
    text = re.sub(r"AIza[0-9A-Za-z_-]{20,}", "[REDACTED]", text)
    text = re.sub(r"\b(?:KEY|TOKEN|SECRET|PASSWORD)=\S+", "[REDACTED]", text, flags=re.IGNORECASE)
    return text[-800:]


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _without_fences(text: str) -> str:
    return FENCE_RE.sub("", text)


def _is_external_link(target: str) -> bool:
    lowered = target.casefold()
    return lowered.startswith(("http://", "https://", "mailto:", "tel:", "data:", "app://"))


def _markdown_anchors(text: str) -> set[str]:
    anchors: set[str] = set()
    counts: dict[str, int] = {}
    clean = _without_fences(text)
    for raw_heading in re.findall(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$", clean, re.MULTILINE):
        heading = re.sub(r"<[^>]+>", "", raw_heading)
        heading = re.sub(r"!?\[([^\]]+)\]\([^)]*\)", r"\1", heading)
        heading = re.sub(r"[`*_~]", "", heading).strip().casefold()
        heading = re.sub(r"[^\w\s-]", "", heading, flags=re.UNICODE)
        slug = re.sub(r"\s+", "-", heading).strip("-")
        if not slug:
            continue
        duplicate = counts.get(slug, 0)
        counts[slug] = duplicate + 1
        anchors.add(slug if duplicate == 0 else f"{slug}-{duplicate}")
    anchors.update(value.casefold() for value in re.findall(r"\bid=['\"]([^'\"]+)['\"]", clean, re.IGNORECASE))
    return anchors


def _contract_text_files() -> list[Path]:
    files: set[Path] = set(PRIMARY_MARKDOWN)
    for name in (
        ".dockerignore",
        ".env.example",
        ".gitignore",
        "docker-compose.yml",
        "pytest.ini",
        "requirements-dev.txt",
    ):
        path = ROOT / name
        if path.is_file():
            files.add(path)
    for base in (
        ROOT / ".github",
        ROOT / "apps",
        ROOT / "packages",
        ROOT / "research" / "pipeline",
        ROOT / "scripts",
    ):
        if not base.is_dir():
            continue
        for path in base.rglob("*"):
            if not path.is_file() or any(part in SKIP_DIRS for part in path.parts):
                continue
            if "tests" in path.parts or "e2e" in path.parts or path.name == "verify_readme.py":
                continue
            if path.name == "package-lock.json":
                continue
            if path.suffix.casefold() in {".py", ".ts", ".tsx", ".mjs", ".json", ".yml", ".yaml", ".toml", ".txt", ".ini", ".example"} or path.name in {"Dockerfile", ".dockerignore", ".gitignore"}:
                files.add(path)
    return sorted(files)


def _contains_old_repository_path(value: str) -> bool:
    if value in {"models/text-embedding-004", "models/gemini-embedding-2"}:
        return False
    return any(pattern.search(value) for pattern in OLD_PATH_PATTERNS.values())


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_secret_env_path(path: Path) -> bool:
    name = path.name.casefold()
    return name == ".env" or (name.startswith(".env.") and name != ".env.example")


def _same_number(left: Any, right: Any, tolerance: float = 1e-12) -> bool:
    try:
        return abs(float(left) - float(right)) <= tolerance
    except (TypeError, ValueError):
        return False


def _contains_year_range(text: str, first: Any, last: Any) -> bool:
    try:
        start = int(first)
        end = int(last)
    except (TypeError, ValueError):
        return False
    return re.search(rf"\b{start}\s*(?:-|–|—|to)\s*{end}\b", text, re.IGNORECASE) is not None


def _contains_formatted_number(
    text: str,
    value: Any,
    decimals: int,
    *,
    percentage: bool = False,
) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    if percentage:
        number *= 100.0
    if decimals == 0:
        candidates = {f"{number:,.0f}", f"{number:.0f}", f"{number:,.1f}", f"{number:.1f}", f"{number:,.2f}", f"{number:.2f}"}
    else:
        candidates = {f"{number:,.{decimals}f}", f"{number:.{decimals}f}"}
    suffix = "%" if percentage else ""
    return any(
        re.search(
            rf"(?<![\d.,]){re.escape(candidate)}{re.escape(suffix)}(?![\d.,])",
            text,
        )
        for candidate in candidates
    )


def _safe_child_env() -> dict[str, str]:
    allowed = {
        "HOME",
        "LANG",
        "LC_ALL",
        "PATH",
        "SYSTEMROOT",
        "TEMP",
        "TMP",
        "USERPROFILE",
        "VIRTUAL_ENV",
        "WINDIR",
        "XDG_RUNTIME_DIR",
        "DOCKER_CONTEXT",
        "DOCKER_HOST",
    }
    environment = {name: os.environ[name] for name in allowed if name in os.environ}
    environment["PYTHONIOENCODING"] = "utf-8"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["COMPOSE_DISABLE_ENV_FILE"] = "1"
    return environment


def _installed_openapi_contract(package: str) -> set[tuple[str, str]] | str:
    code = f"""
import builtins
import json
from pathlib import Path
# Refuse any direct secret-file read attempted as an import side effect.
_real_open = builtins.open
_real_path_open = Path.open
def _is_secret_env_path(value):
    try:
        name = Path(value).name.casefold()
    except (TypeError, ValueError):
        return False
    return name == ".env" or (name.startswith(".env.") and name != ".env.example")
def _guarded_open(file, *args, **kwargs):
    if _is_secret_env_path(file):
        raise RuntimeError("README verifier blocked an application .env read")
    return _real_open(file, *args, **kwargs)
def _guarded_path_open(self, *args, **kwargs):
    if _is_secret_env_path(self):
        raise RuntimeError("README verifier blocked an application .env read")
    return _real_path_open(self, *args, **kwargs)
builtins.open = _guarded_open
Path.open = _guarded_path_open
# Prevent application settings from consulting any repository .env path, even
# if a package later changes its SettingsConfigDict to use an absolute file.
try:
    from pydantic_settings.sources import DotEnvSettingsSource
    DotEnvSettingsSource._load_env_vars = lambda self: {{}}
    DotEnvSettingsSource._read_env_files = lambda self: {{}}
    DotEnvSettingsSource.__call__ = lambda self: {{}}
except Exception:
    pass
try:
    import dotenv
    dotenv.load_dotenv = lambda *args, **kwargs: False
    dotenv.dotenv_values = lambda *args, **kwargs: {{}}
except Exception:
    pass
from {package} import create_app
app = create_app()
schema = app.openapi()
methods = {{"get", "post", "put", "patch", "delete", "options", "head"}}
result = sorted(
    (method.upper(), path)
    for path, operations in schema.get("paths", {{}}).items()
    for method in operations
    if method.casefold() in methods
)
print(json.dumps(result))
"""
    with tempfile.TemporaryDirectory(prefix="seatcraft-openapi-") as directory:
        try:
            completed = subprocess.run(
                [sys.executable, "-c", code],
                cwd=directory,
                env=_safe_child_env(),
                text=True,
                capture_output=True,
                timeout=30,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return _safe_error(exc)
    if completed.returncode != 0:
        return _safe_error(completed.stderr or f"exit code {completed.returncode}")
    try:
        payload = json.loads(completed.stdout)
        return {(str(method), str(path)) for method, path in payload}
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        return f"invalid OpenAPI subprocess output ({_safe_error(exc)})"


def _compose_contract(profile: str | None) -> dict[str, Any] | str:
    with tempfile.TemporaryDirectory(prefix="seatcraft-compose-") as directory:
        empty_env = Path(directory) / "empty.env"
        empty_env.write_text("", encoding="utf-8")
        command = ["docker", "compose", "--env-file", str(empty_env)]
        if profile is not None:
            command.extend(["--profile", profile])
        command.extend(["config", "--format", "json"])
        try:
            completed = subprocess.run(
                command,
                cwd=ROOT,
                env=_safe_child_env(),
                text=True,
                capture_output=True,
                timeout=30,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return _safe_error(exc)
    if completed.returncode != 0:
        return _safe_error(completed.stderr or f"exit code {completed.returncode}")
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        return f"invalid JSON from Docker Compose ({_safe_error(exc)})"
    return payload if isinstance(payload, dict) else "Docker Compose returned a non-object"


def _has_port(service: dict[str, Any], published: int, target: int) -> bool:
    for entry in service.get("ports", []) or []:
        if isinstance(entry, dict):
            try:
                if int(entry.get("published")) == published and int(entry.get("target")) == target:
                    return True
            except (TypeError, ValueError):
                continue
        elif isinstance(entry, str):
            match = re.search(r"(?:^|:)(\d+):(\d+)(?:/\w+)?$", entry)
            if match and (int(match.group(1)), int(match.group(2))) == (published, target):
                return True
    return False


def _build_context(service: dict[str, Any]) -> Path | None:
    build = service.get("build")
    if isinstance(build, str):
        return Path(build).resolve()
    if isinstance(build, dict) and isinstance(build.get("context"), str):
        return Path(build["context"]).resolve()
    return None


def _documented_endpoints(text: str) -> set[str]:
    endpoints = set(
        re.findall(
            r"(?<![\w.])/(?:api/(?:v1|rag)/[A-Za-z0-9_./{}-]+|health\b|ready\b)",
            text,
        )
    )
    for path in re.findall(r"https?://(?:localhost|127\.0\.0\.1):\d+(/[A-Za-z0-9_./{}-]+)", text):
        if path.startswith(("/api/", "/health", "/ready")):
            endpoints.add(path)
    return endpoints


def _heading_section(text: str, heading_word: str) -> str:
    matches = list(re.finditer(r"^##\s+(.+?)\s*$", text, re.MULTILINE))
    for index, match in enumerate(matches):
        if heading_word.casefold() not in match.group(1).casefold():
            continue
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        return text[match.end():end]
    return ""


if __name__ == "__main__":
    raise SystemExit(ContractVerifier().run())
