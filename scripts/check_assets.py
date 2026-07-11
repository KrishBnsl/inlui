#!/usr/bin/env python3
"""Validate local research/source and serving assets against the manifest."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "research" / "reports" / "artifact_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    if not MANIFEST.is_file():
        print("ASSET CHECK: BLOCKED")
        print(f"Missing manifest: {MANIFEST.relative_to(ROOT)}")
        print(
            "Run the strict evaluator only after provenance-verified source data is "
            "available; see docs/TECHNICAL_REPORT.md#reproducibility."
        )
        return 2

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assets = [
        *manifest.get("source_assets", []),
        *manifest.get("generated_serving_artifacts", []),
        *manifest.get("versioned_outputs", []),
    ]
    missing: list[str] = []
    mismatched: list[str] = []
    for entry in assets:
        path = ROOT / entry["path"]
        if not path.is_file():
            missing.append(entry["path"])
            continue
        expected = entry.get("sha256")
        if expected and sha256(path) != expected:
            mismatched.append(entry["path"])

    if missing or mismatched:
        print("ASSET CHECK: BLOCKED")
        for path in missing:
            print(f"- missing: {path}")
        for path in mismatched:
            print(f"- SHA-256 mismatch: {path}")
        print()
        print("Automatic download is intentionally disabled: the repository has no verified raw-source URL or redistribution permission.")
        print(
            "Obtain permission, place immutable official JoSAA round files under "
            "research/data/, then run:"
        )
        print("  python -m research.pipeline.preprocessing --data-dir research/data")
        print(
            "  python -m research.pipeline.research_evaluation "
            "--data research/data/processed/josaa_cleaned.csv "
            "--data-cutoff-year 2025 --final-test-year 2025 --prediction-year 2026"
        )
        return 2

    print(f"ASSET CHECK: PASSED ({len(assets)} files match the versioned SHA-256 manifest)")
    provenance = manifest.get("source_assets", [{}])[0].get("provenance", "unknown")
    license_status = manifest.get("source_assets", [{}])[0].get("license", "unknown")
    if provenance != "verified" or license_status == "unresolved":
        print(
            "RESEARCH EVIDENCE STATUS: EXPLORATORY "
            f"(provenance={provenance}, license={license_status})"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
