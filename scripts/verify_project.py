#!/usr/bin/env python3
"""Run reproducibility checks and write machine-readable verification evidence."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "research" / "reports" / "verification"


@dataclass
class CheckResult:
    name: str
    command: str
    status: str
    exit_code: int | None
    duration_seconds: float
    stdout: str = ""
    stderr: str = ""
    reason: str | None = None


def _redact(text: str) -> str:
    secret = os.environ.get("GOOGLE_API_KEY", "")
    if secret:
        text = text.replace(secret, "[REDACTED]")
    text = re.sub(r"AIza[0-9A-Za-z_-]{35}", "[REDACTED_GOOGLE_KEY]", text)
    text = re.sub(
        r"(?<![0-9A-Za-z])sk-(?:proj-)?[0-9A-Za-z_-]{20,}",
        "[REDACTED_API_KEY]",
        text,
    )
    return text[-12_000:]


def _display_command(argv: list[str]) -> str:
    return " ".join(f'"{arg}"' if " " in arg else arg for arg in argv)


def run_check(
    name: str,
    argv: list[str],
    *,
    cwd: Path = ROOT,
    timeout: int = 600,
) -> CheckResult:
    command = _display_command(argv)
    executable = argv[0]
    if shutil.which(executable) is None and not Path(executable).exists():
        return CheckResult(name, command, "skipped", None, 0.0, reason=f"{executable} not found")

    started = time.perf_counter()
    try:
        completed = subprocess.run(
            argv,
            cwd=cwd,
            text=True,
            capture_output=True,
            timeout=timeout,
            env=os.environ.copy(),
        )
    except subprocess.TimeoutExpired as exc:
        return CheckResult(
            name,
            command,
            "failed",
            None,
            round(time.perf_counter() - started, 3),
            _redact(exc.stdout or ""),
            _redact(exc.stderr or ""),
            reason=f"timed out after {timeout}s",
        )

    return CheckResult(
        name=name,
        command=command,
        status="passed" if completed.returncode == 0 else "failed",
        exit_code=completed.returncode,
        duration_seconds=round(time.perf_counter() - started, 3),
        stdout=_redact(completed.stdout),
        stderr=_redact(completed.stderr),
    )


def skipped(name: str, command: str, reason: str) -> CheckResult:
    return CheckResult(name, command, "skipped", None, 0.0, reason=reason)


def command_output(argv: list[str], cwd: Path = ROOT) -> str | None:
    try:
        return subprocess.run(
            argv, cwd=cwd, check=True, capture_output=True, text=True, timeout=20
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def environment_versions() -> dict[str, str]:
    versions = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "node": command_output(["node", "--version"]) or "NOT AVAILABLE",
        "npm": command_output(["npm", "--version"]) or "NOT AVAILABLE",
        "docker_compose": command_output(["docker", "compose", "version", "--short"])
        or "NOT AVAILABLE",
    }
    for distribution in ("pandas", "numpy", "scikit-learn", "fastapi", "pydantic", "pytest"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = "NOT INSTALLED"
    return versions


def load_first_json(candidates: list[Path]) -> tuple[str | None, Any]:
    for path in candidates:
        if path.is_file():
            try:
                return path.relative_to(ROOT).as_posix(), json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return path.relative_to(ROOT).as_posix(), {"status": "INVALID JSON"}
    return None, None


def parse_test_counts(results: list[CheckResult]) -> dict[str, int]:
    totals = {"passed": 0, "failed": 0, "skipped": 0}
    patterns = {
        "passed": [r"(\d+) passed", r"^ℹ pass (\d+)$"],
        "failed": [r"(\d+) failed", r"^ℹ fail (\d+)$"],
        "skipped": [r"(\d+) skipped", r"^ℹ skipped (\d+)$"],
    }
    for result in results:
        combined = f"{result.stdout}\n{result.stderr}"
        for key, expressions in patterns.items():
            matches: list[int] = []
            for expression in expressions:
                matches.extend(int(value) for value in re.findall(expression, combined, re.MULTILINE))
            if matches:
                totals[key] += max(matches)
    return totals


def flatten_metrics(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    if isinstance(payload.get("evaluation"), dict):
        evaluation = payload["evaluation"]
        final_metrics = evaluation.get("final_test_metrics", {})
        dataset = payload.get("dataset", {}) if isinstance(payload.get("dataset"), dict) else {}
        uncertainty = (
            payload.get("uncertainty", {})
            if isinstance(payload.get("uncertainty"), dict)
            else {}
        )
        return {
            "status": payload.get("status"),
            "feature_mode": payload.get("primary_feature_mode"),
            "primary_model": payload.get("primary_model"),
            "random_seed": payload.get("random_seed"),
            "data_cutoff_year": dataset.get("data_cutoff_year"),
            "dataset_sha256": dataset.get("sha256"),
            "dataset_provenance": dataset.get("provenance"),
            "dataset_license": dataset.get("license"),
            "final_test_year": evaluation.get("final_test_year"),
            "final_test_metrics": final_metrics,
            "prediction_interval_empirical_coverage": uncertainty.get(
                "final_test_empirical_coverage"
            ),
            "final_test_residuals_used_for_interval": uncertainty.get(
                "final_test_residuals_used_for_interval"
            ),
        }
    preferred = (
        "feature_mode",
        "data_cutoff",
        "test_year",
        "model",
        "mae",
        "rmse",
        "median_absolute_error",
        "r2",
        "prediction_interval_coverage",
        "interval_coverage",
        "random_seed",
        "dataset_sha256",
    )
    summary = {key: payload[key] for key in preferred if key in payload}
    if summary:
        return summary
    for value in payload.values():
        if isinstance(value, dict):
            summary = {key: value[key] for key in preferred if key in value}
            if summary:
                return summary
    return {}


def write_reports(payload: dict[str, Any]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = REPORT_DIR / "verification_report.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {json_path.relative_to(ROOT)}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-frontend-build", action="store_true")
    parser.add_argument("--skip-e2e", action="store_true")
    args = parser.parse_args()

    npm = "npm.cmd" if os.name == "nt" else "npm"
    checks = [
        run_check(
            "Python compile",
            [
                sys.executable,
                "-m",
                "compileall",
                "-q",
                "research",
                "packages",
                "apps/inference-api",
                "apps/rag-api",
                "scripts",
            ],
        ),
        run_check(
            "Python lint",
            [
                sys.executable,
                "-m",
                "ruff",
                "check",
                "research",
                "packages",
                "apps/inference-api",
                "apps/rag-api",
                "scripts",
                "--select",
                "E9,F63,F7,F82",
            ],
        ),
        run_check("Asset manifest", [sys.executable, "scripts/check_assets.py"]),
        run_check("README contract", [sys.executable, "scripts/verify_readme.py"]),
    ]

    python_test_suites = [
        ("Research tests", ROOT / "research" / "pipeline" / "tests"),
        ("Inference API tests", ROOT / "apps" / "inference-api" / "tests"),
        ("RAG mocked tests", ROOT / "apps" / "rag-api" / "tests"),
    ]
    for name, path in python_test_suites:
        checks.append(
            run_check(
                name,
                [sys.executable, "-m", "pytest", str(path.relative_to(ROOT)), "-q"],
            )
            if path.is_dir()
            else skipped(name, f"{sys.executable} -m pytest", f"{path.relative_to(ROOT)} not found")
        )
    checks.extend(
        [
            run_check("Frontend unit tests", [npm, "test"], cwd=ROOT / "apps" / "web"),
            run_check("Frontend lint", [npm, "run", "lint"], cwd=ROOT / "apps" / "web"),
            run_check(
                "Frontend type check",
                [npm, "run", "typecheck"],
                cwd=ROOT / "apps" / "web",
            ),
            run_check(
                "Frontend production dependency audit",
                [npm, "audit", "--omit=dev"],
                cwd=ROOT / "apps" / "web",
            ),
        ]
    )
    checks.append(
        skipped("Frontend build", f"{npm} run build", "--skip-frontend-build")
        if args.skip_frontend_build
        else run_check(
            "Frontend build",
            [npm, "run", "build"],
            cwd=ROOT / "apps" / "web",
            timeout=900,
        )
    )
    checks.append(
        skipped("Playwright E2E", f"{npm} run test:e2e", "--skip-e2e")
        if args.skip_e2e
        else run_check(
            "Playwright E2E",
            [npm, "run", "test:e2e"],
            cwd=ROOT / "apps" / "web",
            timeout=900,
        )
    )
    checks.extend(
        [
            run_check("Docker Compose config", ["docker", "compose", "config", "--quiet"]),
            run_check("Secret scan", [sys.executable, "scripts/scan_secrets.py", "--history"], timeout=180),
        ]
    )

    services = run_check("Default Compose services", ["docker", "compose", "config", "--services"])
    if services.status == "passed":
        actual = set(services.stdout.split())
        expected = {"web", "inference-api", "rag-api"}
        if actual != expected:
            services.status = "failed"
            services.reason = f"expected {sorted(expected)}, got {sorted(actual)}"
            services.exit_code = 1
    checks.append(services)

    if os.environ.get("RUN_LIVE_GEMINI_TEST") == "1":
        live = run_check(
            "Live Gemini",
            [sys.executable, "scripts/verify_gemini.py"],
            timeout=60,
        )
        live_status = "PASSED" if live.status == "passed" else "FAILED"
    else:
        live = skipped(
            "Live Gemini",
            f"RUN_LIVE_GEMINI_TEST=1 {sys.executable} scripts/verify_gemini.py",
            "opt-in flag not set",
        )
        live_status = "NOT RUN"
    checks.append(live)

    manifest_source, manifest = load_first_json(
        [
            ROOT / "research" / "reports" / "artifact_manifest.json",
        ]
    )
    evaluation_source, evaluation = load_first_json(
        [
            ROOT / "research" / "reports" / "metrics.json",
        ]
    )
    current_commit = command_output(["git", "rev-parse", "HEAD"]) or "UNKNOWN"
    verification_worktree_dirty = bool(command_output(["git", "status", "--porcelain"]))
    if isinstance(evaluation, dict):
        source_commit = (
            evaluation.get("source_commit")
            or evaluation.get("generated_from_commit")
            or evaluation.get("git_commit")
            or current_commit
        )
        source_worktree_dirty = evaluation.get(
            "source_worktree_dirty",
            evaluation.get("worktree_dirty", evaluation.get("git_worktree_dirty")),
        )
    else:
        source_commit = current_commit
        source_worktree_dirty = verification_worktree_dirty

    summary = {
        "passed": sum(result.status == "passed" for result in checks),
        "failed": sum(result.status == "failed" for result in checks),
        "skipped": sum(result.status == "skipped" for result in checks),
    }
    blockers: list[str] = []
    if live_status == "NOT RUN":
        blockers.append("Live Gemini verification was not run; set RUN_LIVE_GEMINI_TEST=1 with GOOGLE_API_KEY.")
    if manifest is None:
        blockers.append("No machine-readable data/artifact manifest was found.")
    if evaluation is None:
        blockers.append("No machine-readable pre-counselling evaluation result was found.")
    if isinstance(manifest, dict):
        for blocker in manifest.get("blockers", []):
            blockers.append(str(blocker))
        fresh_clone_status = manifest.get("fresh_clone_status")
        if fresh_clone_status and fresh_clone_status not in {"ready", "reproducible"}:
            recovery = manifest.get("recovery", "No recovery command was recorded.")
            blockers.append(f"Fresh-clone assets: {fresh_clone_status}. {recovery}")
        for source in manifest.get("source_assets", []):
            if not isinstance(source, dict):
                continue
            if source.get("provenance") != "verified" or source.get("license") in {
                None,
                "unresolved",
            }:
                blockers.append(
                    f"Source asset {source.get('path', 'unknown')} has "
                    f"provenance={source.get('provenance')} and license={source.get('license')}."
                )

    payload = {
        "schema_version": 2,
        "overall_status": "FAILED" if summary["failed"] else ("PASSED_WITH_SKIPS" if summary["skipped"] else "PASSED"),
        "source_commit": source_commit,
        "generated_from_commit": source_commit,
        "source_worktree_dirty": source_worktree_dirty,
        "worktree_dirty": source_worktree_dirty,
        "verification_commit": current_commit,
        "verification_worktree_dirty": verification_worktree_dirty,
        # Backward-compatible aliases refer to the experiment source snapshot.
        "git_commit": source_commit,
        "git_worktree_dirty": source_worktree_dirty,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "environment": environment_versions(),
        "commands": [asdict(result) for result in checks],
        "summary": summary,
        "test_counts": parse_test_counts(checks),
        "live_gemini_status": live_status,
        "manifest_source": manifest_source,
        "manifest": manifest or {},
        "model_evaluation_source": evaluation_source,
        "model_evaluation_summary": flatten_metrics(evaluation),
        "known_blockers": list(dict.fromkeys(blockers)),
    }
    write_reports(payload)
    print(
        "VERIFICATION: "
        f"{payload['overall_status']} ({summary['passed']} passed, {summary['failed']} failed, "
        f"{summary['skipped']} skipped; live Gemini {live_status})"
    )
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
