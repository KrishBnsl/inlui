#!/usr/bin/env python3
"""Lightweight, dependency-free secret scan for the working tree and Git history.

This is intentionally conservative: it looks for high-signal credential formats
and private-key headers, reports only file/line locations, and never echoes the
matched value. It complements (but does not replace) repository-host scanning.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


PATTERNS: dict[str, re.Pattern[str]] = {
    "google_api_key": re.compile(r"AIza[0-9A-Za-z_-]{35}"),
    "github_token": re.compile(r"gh[pousr]_[0-9A-Za-z]{36,255}"),
    "openai_api_key": re.compile(r"(?<![0-9A-Za-z])sk-(?:proj-)?[0-9A-Za-z_-]{32,}"),
    "aws_access_key": re.compile(r"(?:AKIA|ASIA)[0-9A-Z]{16}"),
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
}

MAX_TEXT_BYTES = 10 * 1024 * 1024


def working_tree_files(root: Path) -> list[Path]:
    completed = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    return [root / entry.decode("utf-8") for entry in completed.stdout.split(b"\0") if entry]


def staged_files(root: Path) -> list[str]:
    completed = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    return [entry.decode("utf-8") for entry in completed.stdout.split(b"\0") if entry]


def staged_text(root: Path, path: str) -> str | None:
    size = subprocess.run(
        ["git", "cat-file", "-s", f":{path}"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    if int(size.stdout.strip()) > MAX_TEXT_BYTES:
        return None
    data = subprocess.run(
        ["git", "show", f":{path}"],
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


def history_findings(root: Path) -> tuple[int, list[str]]:
    commits = subprocess.run(
        ["git", "rev-list", "--all"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.splitlines()

    # Scan each reachable blob once. This avoids re-reading an unchanged large
    # file for every commit and makes the history check deterministic in CI.
    object_lines = subprocess.run(
        ["git", "rev-list", "--objects", "--all"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    object_paths: dict[str, str] = {}
    object_ids: list[str] = []
    for line in object_lines:
        object_id, _, path = line.partition(" ")
        object_ids.append(object_id)
        if path:
            object_paths.setdefault(object_id, path)

    check = subprocess.run(
        ["git", "cat-file", "--batch-check=%(objectname) %(objecttype) %(objectsize)"],
        cwd=root,
        input="\n".join(object_ids) + "\n",
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    candidates: list[str] = []
    for line in check:
        object_id, object_type, size_text = line.split()
        if object_type == "blob" and int(size_text) <= MAX_TEXT_BYTES:
            candidates.append(object_id)

    process = subprocess.Popen(
        ["git", "cat-file", "--batch"],
        cwd=root,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stdin is not None and process.stdout is not None
    process.stdin.write(("\n".join(candidates) + "\n").encode("ascii"))
    process.stdin.close()

    findings: list[str] = []
    for expected_id in candidates:
        header = process.stdout.readline().decode("ascii").strip()
        object_id, object_type, size_text = header.split()
        if object_id != expected_id or object_type != "blob":
            raise RuntimeError("Unexpected git cat-file batch response")
        data = process.stdout.read(int(size_text))
        process.stdout.read(1)  # protocol newline
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            continue
        path = object_paths.get(object_id, object_id)
        for name, pattern in PATTERNS.items():
            match = pattern.search(text)
            if match:
                line_number = text.count("\n", 0, match.start()) + 1
                findings.append(f"{path}:{line_number}:{name}")

    stderr = process.stderr.read() if process.stderr is not None else b""
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"git cat-file failed: {stderr.decode('utf-8', 'replace')}")
    return len(commits), findings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", action="store_true", help="also scan every reachable commit")
    parser.add_argument(
        "--staged",
        action="store_true",
        help="scan the exact index snapshot instead of working-tree files",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    findings: list[tuple[str, int, str]] = []

    if args.staged:
        scanned_paths = staged_files(root)
        candidates = ((path, staged_text(root, path)) for path in scanned_paths)
    else:
        working_files = working_tree_files(root)
        scanned_paths = [path.relative_to(root).as_posix() for path in working_files]

        def read_working_file(path: Path) -> str | None:
            try:
                if path.stat().st_size > MAX_TEXT_BYTES:
                    return None
                return path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                return None

        candidates = (
            (path.relative_to(root).as_posix(), read_working_file(path))
            for path in working_files
        )

    for filename, text in candidates:
        if text is None:
            continue

        for line_number, line in enumerate(text.splitlines(), start=1):
            for name, pattern in PATTERNS.items():
                if pattern.search(line):
                    findings.append((filename, line_number, name))

    if findings:
        print("SECRET SCAN: FAILED")
        for filename, line_number, kind in findings:
            print(f"- {filename}:{line_number}: potential {kind} (value redacted)")
        return 1

    history_count = 0
    if args.history:
        history_count, history_hits = history_findings(root)
        if history_hits:
            print("SECRET HISTORY SCAN: FAILED")
            for location in history_hits:
                print(f"- {location}: potential secret (value redacted)")
            return 1

    suffix = f"; {history_count} commits inspected" if args.history else ""
    scope = "staged" if args.staged else "versioned/working-tree"
    print(
        f"SECRET SCAN: PASSED ({len(scanned_paths)} {scope} files inspected{suffix})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
