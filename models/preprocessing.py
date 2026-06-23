"""
JoSAA Cutoff Data Preprocessing Pipeline
==========================================

Day 1 deliverable: takes raw JoSAA round CSVs (2016–2026) plus auxiliary
files, normalizes schemas, cleans ranks, deduplicates, and produces
analysis-ready CSV/Parquet outputs with full provenance tracking.

Outputs:
    data/processed/josaa_cleaned.csv          – cleaned, analysis-ready data
    data/processed/master_cutoffs.csv|parquet  – cleaned + engineered features
    data/processed/branch_timeseries.parquet   – pivoted closing ranks by year
    data/processed/seat_matrix.parquet         – melted seat matrix
    data/processed/train.parquet               – temporal train split
    data/processed/test.parquet                – temporal test split
    data/processed/raw_backup.csv              – raw data before any cleaning
    data/processed/data_dictionary.md          – column documentation
    data/processed/summary_report.md           – cleaning summary
    data/processed/mappings/                   – normalization mapping JSONs

Usage:
    python preprocessing.py                    # run full pipeline
    python preprocessing.py --data-dir ./data  # custom data directory

The module is also fully importable:
    from preprocessing import build_master_dataset
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Columns in the final cleaned output (order matters for readability)
CLEANED_COLUMNS = [
    "year",
    "round",
    "institute",
    "institute_type",
    "program",
    "discipline",
    "duration_years",
    "degree_type",
    "quota",
    "category",
    "is_pwd",
    "gender",
    "opening_rank",
    "closing_rank",
    "is_pwd_rank",
    "source_file",
    "source_row_id",
]

# Extended columns for feature-engineered master output
MASTER_EXTRA_COLUMNS = [
    "opening_percentile",
    "closing_percentile",
    "competitiveness_ratio",
    "applicants",
    "rank_spread",
    "closing_rank_yoy_change",
    "round_progression",
]

# Natural key for deduplication
NATURAL_KEY = [
    "year",
    "round",
    "institute",
    "program",
    "quota",
    "category",
    "is_pwd",
    "gender",
]

LEGACY_HEADER = [
    "Institute",
    "Academic Program Name",
    "Quota",
    "Seat Type",
    "Opening Rank",
    "Closing Rank",
]

MODERN_HEADER = [
    "Institute",
    "Academic Program Name",
    "Quota",
    "Seat Type",
    "Gender",
    "Opening Rank",
    "Closing Rank",
]

# Pattern: "Civil Engineering (4 Years, Bachelor of Technology)"
_PROGRAM_RE = re.compile(
    r"^(?P<name>.+?)\s*"               # program name (greedy-lazy)
    r"\((?P<dur>\d+)\s*Years?,\s*"      # duration
    r"(?P<deg>.+?)\)$",                 # degree type
    re.IGNORECASE,
)

# Pattern for PwD ranks: "123P"
_PWD_RANK_RE = re.compile(r"^(\d+)P$", re.IGNORECASE)

# Pattern to extract year/round from filename: "2024_round_3.csv"
_FILENAME_RE = re.compile(r"^(\d{4})_round_(\d+)\.\w+$")

# Null-like sentinel values to coerce to NaN
_NULL_SENTINELS = {"", "na", "n/a", "null", "none", "-", "--", "---", "nan"}


# =========================================================================
# 1. File Loading & Schema Harmonization
# =========================================================================

def _coerce_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """Replace known null-like string values with pd.NA across all columns."""
    str_cols = df.select_dtypes(include=["object", "string"]).columns
    for col in str_cols:
        df[col] = df[col].apply(
            lambda v: pd.NA if (pd.isna(v) or str(v).strip().lower() in _NULL_SENTINELS) else v
        )
    return df


def load_single_csv(path: Path) -> Optional[pd.DataFrame]:
    """
    Load one round CSV, harmonizing legacy (6-col) and modern (7-col) schemas.

    Adds ``source_file`` and ``source_row_id`` columns for provenance.

    Returns
    -------
    pd.DataFrame or None
        None if the file doesn't match the expected naming pattern.
    """
    match = _FILENAME_RE.match(path.name)
    if not match:
        log.warning("Skipping non-round file: %s", path.name)
        return None

    year, round_num = int(match.group(1)), int(match.group(2))

    # Sniff the header to detect schema version
    with open(path, "r", encoding="utf-8") as f:
        header_line = f.readline().strip()

    num_cols = len(header_line.split(","))
    is_legacy = num_cols <= 6  # 2016–2017: no Gender column

    if is_legacy:
        df = pd.read_csv(path, names=LEGACY_HEADER, header=0, dtype=str)
        df["Gender"] = "Gender-Neutral"
    else:
        df = pd.read_csv(path, names=MODERN_HEADER, header=0, dtype=str)

    # Strip whitespace from all string columns (handles 2026 trailing spaces)
    str_cols = df.select_dtypes(include=["object", "string"]).columns
    for col in str_cols:
        df[col] = df[col].str.strip()

    # Coerce null-like sentinels
    df = _coerce_nulls(df)

    # Provenance
    df["year"] = year
    df["round"] = round_num
    df["source_file"] = path.name
    df["source_row_id"] = range(len(df))

    return df


def _try_load_excel(path: Path) -> Optional[pd.DataFrame]:
    """Attempt to load an Excel file and extract year/round from filename."""
    match = _FILENAME_RE.match(path.name)
    if not match:
        log.warning("Skipping non-round Excel file: %s", path.name)
        return None

    year, round_num = int(match.group(1)), int(match.group(2))

    try:
        df = pd.read_excel(path, dtype=str, engine="openpyxl")
    except ImportError:
        log.warning("openpyxl not installed; skipping Excel file %s", path.name)
        return None
    except Exception as e:
        log.warning("Failed to read Excel file %s: %s", path.name, e)
        return None

    # Strip whitespace from all string columns
    str_cols = df.select_dtypes(include=["object", "string"]).columns
    for col in str_cols:
        df[col] = df[col].str.strip()

    df = _coerce_nulls(df)
    df["year"] = year
    df["round"] = round_num
    df["source_file"] = path.name
    df["source_row_id"] = range(len(df))

    return df


def load_all_round_files(data_dir: Path) -> pd.DataFrame:
    """
    Discover and load all ``{year}_round_{n}.csv|xlsx`` files under *data_dir*.

    Supports both CSV and Excel formats.  Returns a single concatenated
    DataFrame with ``year``, ``round``, ``source_file``, ``source_row_id``
    columns attached.
    """
    csv_files = sorted(data_dir.glob("*_round_*.csv"))
    xlsx_files = sorted(data_dir.glob("*_round_*.xlsx")) + sorted(data_dir.glob("*_round_*.xls"))

    if not csv_files and not xlsx_files:
        raise FileNotFoundError(f"No round CSV/Excel files found in {data_dir}")

    frames: list[pd.DataFrame] = []

    for f in csv_files:
        # Skip Windows Zone.Identifier sidecar files
        if "Zone" in f.name:
            continue
        df = load_single_csv(f)
        if df is not None:
            frames.append(df)
            log.info("  %-28s  →  %6d rows  (year=%d, round=%d)",
                      f.name, len(df), df["year"].iloc[0], df["round"].iloc[0])

    for f in xlsx_files:
        if "Zone" in f.name:
            continue
        df = _try_load_excel(f)
        if df is not None:
            frames.append(df)
            log.info("  %-28s  →  %6d rows  (year=%d, round=%d)  [Excel]",
                      f.name, len(df), df["year"].iloc[0], df["round"].iloc[0])

    combined = pd.concat(frames, ignore_index=True)
    log.info("Loaded %d files → %d total rows", len(frames), len(combined))
    return combined


# =========================================================================
# 2. Rank Cleaning
# =========================================================================

def _parse_rank(val: Any) -> tuple[Optional[int], bool]:
    """
    Parse a single rank value.

    Returns ``(numeric_rank, is_pwd_rank)``.

    Examples
    --------
    >>> _parse_rank("123P")
    (123, True)
    >>> _parse_rank("5057.0")
    (5057, False)
    >>> _parse_rank("")
    (None, False)
    """
    if pd.isna(val) or str(val).strip() == "":
        return None, False

    val_str = str(val).strip().replace(",", "")  # handle comma-formatted ranks

    # PwD rank suffix
    pwd_match = _PWD_RANK_RE.match(val_str)
    if pwd_match:
        return int(pwd_match.group(1)), True

    # Float rank (2018 data has "5057.0")
    try:
        return int(float(val_str)), False
    except (ValueError, OverflowError):
        return None, False


def clean_ranks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse Opening/Closing Rank into integers; flag PwD ranks; drop garbage rows.

    Rows where **both** ranks are null (trailing empty rows in CSVs) are dropped.
    Rows where ranks are ≤ 0 (placeholder data) are also dropped.
    """
    opening = df["Opening Rank"].apply(_parse_rank)
    closing = df["Closing Rank"].apply(_parse_rank)

    df["opening_rank"] = opening.apply(lambda x: x[0]).astype("Int64")
    df["closing_rank"] = closing.apply(lambda x: x[0]).astype("Int64")
    df["is_pwd_rank"] = opening.apply(lambda x: x[1]) | closing.apply(lambda x: x[1])

    # Drop rows where BOTH ranks are null (trailing empty rows in CSVs)
    before = len(df)
    df = df.dropna(subset=["opening_rank", "closing_rank"], how="all")
    dropped = before - len(df)
    if dropped:
        log.info("Dropped %d rows with no rank data", dropped)

    # Drop rows where ranks are <= 0 (bad placeholder data, e.g. 0.01 floats in 2017)
    bad_mask = (
        (df["opening_rank"].notna() & (df["opening_rank"] <= 0)) |
        (df["closing_rank"].notna() & (df["closing_rank"] <= 0))
    )
    n_bad = bad_mask.sum()
    if n_bad:
        log.warning("Dropping %d rows with rank ≤ 0 (placeholder data)", n_bad)
        df = df[~bad_mask]

    total_dropped = before - len(df)
    if total_dropped:
        log.info("Rank cleaning total: dropped %d rows (%d remain)", total_dropped, len(df))

    return df


# =========================================================================
# 3. Categorical Normalization
# =========================================================================

# --- Tracking dicts for normalization mappings ---
_category_mappings: dict[str, str] = {}
_institute_type_mappings: dict[str, str] = {}
_quota_mappings: dict[str, str] = {}
_gender_mappings: dict[str, str] = {}


def _extract_institute_type(name: str) -> str:
    """Classify institute into IIT / NIT / IIIT / GFTI."""
    n = name.upper()
    if "INDIAN INSTITUTE OF TECHNOLOGY" in n:
        result = "IIT"
    elif "NATIONAL INSTITUTE OF TECHNOLOGY" in n:
        result = "NIT"
    elif "INDIAN INSTITUTE OF INFORMATION TECHNOLOGY" in n or "IIIT" in n:
        result = "IIIT"
    else:
        result = "GFTI"

    _institute_type_mappings[name] = result
    return result


def _decompose_seat_type(seat_type: str) -> tuple[str, bool]:
    """
    Decompose seat type into (category, is_pwd).

    Examples
    --------
    >>> _decompose_seat_type("OBC-NCL (PwD)")
    ('OBC-NCL', True)
    >>> _decompose_seat_type("OPEN")
    ('OPEN', False)
    >>> _decompose_seat_type("EWS")
    ('EWS', False)
    """
    if pd.isna(seat_type):
        return "OPEN", False

    s = str(seat_type).upper().strip()
    is_pwd = "(PWD)" in s or "PWD" in s.split()
    # Remove PwD marker to get base category
    base = re.sub(r"\s*\(?\s*PWD\s*\)?\s*", "", s).strip()

    if base.startswith("OPEN") or base.startswith("GEN"):
        result = "OPEN"
    elif base.startswith("OBC"):
        result = "OBC-NCL"
    elif base.startswith("SC"):
        result = "SC"
    elif base.startswith("ST"):
        result = "ST"
    elif base.startswith("EWS"):
        result = "EWS"
    else:
        result = "OPEN"  # fallback

    _category_mappings[str(seat_type).strip()] = result
    return result, is_pwd


def _normalize_gender(gender: Any) -> str:
    """Normalize gender column to 'Gender-Neutral' or 'Female-only'."""
    if pd.isna(gender) or str(gender).strip() == "":
        result = "Gender-Neutral"
    elif "female" in str(gender).lower():
        result = "Female-only"
    else:
        result = "Gender-Neutral"

    raw = str(gender).strip() if not pd.isna(gender) else "(empty)"
    _gender_mappings[raw] = result
    return result


def _normalize_quota(quota: Any) -> str:
    """
    Normalize quota values.

    AI, HS, OS are kept as-is.  State-specific codes are collapsed to STATE.
    """
    if pd.isna(quota) or str(quota).strip() == "":
        return "AI"  # default

    q = str(quota).strip().upper()
    if q in ("AI", "HS", "OS"):
        result = q
    else:
        result = "STATE"

    _quota_mappings[str(quota).strip()] = result
    return result


def _parse_program(program_name: Any) -> tuple[str, Optional[str], Optional[int], Optional[str]]:
    """
    Extract structured fields from program string.

    Returns ``(program_clean, discipline, duration, degree_type)``.

    Example
    -------
    >>> _parse_program("Civil Engineering (4 Years, Bachelor of Technology)")
    ('Civil Engineering', 'Civil Engineering', 4, 'Bachelor of Technology')
    """
    if pd.isna(program_name):
        return "", None, None, None

    raw = str(program_name).strip()
    m = _PROGRAM_RE.match(raw)
    if m:
        name = m.group("name").strip()
        duration = int(m.group("dur"))
        degree = m.group("deg").strip()
        # Discipline is the core program name (before parenthetical)
        return name, name, duration, degree

    return raw, raw, None, None


def normalize_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all categorical normalizations to the DataFrame."""
    # Clear tracking dicts for fresh run
    _category_mappings.clear()
    _institute_type_mappings.clear()
    _quota_mappings.clear()
    _gender_mappings.clear()

    # Institute type
    df["institute"] = df["Institute"].str.strip()
    df["institute_type"] = df["institute"].apply(_extract_institute_type)

    # Program parsing → program, discipline, duration_years, degree_type
    parsed = df["Academic Program Name"].apply(_parse_program)
    df["program"] = parsed.apply(lambda x: x[0])
    df["discipline"] = parsed.apply(lambda x: x[1])
    df["duration_years"] = parsed.apply(lambda x: x[2]).astype("Int64")
    df["degree_type"] = parsed.apply(lambda x: x[3])

    # Seat type → category + is_pwd
    decomposed = df["Seat Type"].apply(_decompose_seat_type)
    df["category"] = decomposed.apply(lambda x: x[0])
    df["is_pwd"] = decomposed.apply(lambda x: x[1])

    # Gender
    df["gender"] = df["Gender"].apply(_normalize_gender)

    # Quota
    df["quota"] = df["Quota"].apply(_normalize_quota)

    return df


# =========================================================================
# 4. Deduplication
# =========================================================================

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate rows based on the natural key.

    When duplicates exist, keeps the first occurrence (preserving source
    provenance). Logs the count of removed duplicates.
    """
    before = len(df)
    df = df.drop_duplicates(subset=NATURAL_KEY, keep="first")
    dropped = before - len(df)
    if dropped:
        log.info("Removed %d duplicate rows (on natural key)", dropped)
    else:
        log.info("No duplicate rows found")
    return df


# =========================================================================
# 5. Auxiliary Data
# =========================================================================

def load_exam_dynamics(data_dir: Path) -> pd.DataFrame:
    """
    Load ``jee_exam_dynamics.csv`` → DataFrame with columns:
    ``year`` (int), ``applicants`` (int), ``total_seats`` (int).

    Handles comma-formatted numbers like ``"15,38,468"``.
    """
    path = data_dir / "jee_exam_dynamics.csv"
    if not path.exists():
        log.warning("jee_exam_dynamics.csv not found; percentile normalization will be skipped")
        return pd.DataFrame(columns=["year", "applicants", "total_seats"])

    df = pd.read_csv(path, dtype=str)
    df.columns = [c.strip() for c in df.columns]

    # Rename to consistent names
    col_map: dict[str, str] = {}
    for c in df.columns:
        cl = c.lower()
        if "year" in cl:
            col_map[c] = "year"
        elif "applicant" in cl or "registered" in cl:
            col_map[c] = "applicants"
        elif "seat" in cl:
            col_map[c] = "total_seats"
    df = df.rename(columns=col_map)

    # Parse comma-separated numbers: "15,38,468" → 1538468
    for col in ["year", "applicants", "total_seats"]:
        if col in df.columns:
            df[col] = df[col].str.replace(",", "", regex=False).astype(int)

    log.info("Exam dynamics: %d years (range %d–%d)",
              len(df), df["year"].min(), df["year"].max())
    return df[["year", "applicants", "total_seats"]]


def load_seat_matrix(data_dir: Path) -> Optional[pd.DataFrame]:
    """
    Load ``josaa_seat_matrix.csv`` into a flat DataFrame.

    The raw file has a messy multi-row layout where Female-only rows lack
    Institute and Program columns. This function forward-fills them.

    Returns
    -------
    pd.DataFrame or None
        DataFrame with columns: ``institute``, ``program``, ``seat_pool``,
        ``category``, ``seat_count``.  None if file doesn't exist.
    """
    path = data_dir / "josaa_seat_matrix.csv"
    if not path.exists():
        log.warning("josaa_seat_matrix.csv not found; skipping seat matrix")
        return None

    try:
        df = pd.read_csv(path, dtype=str, on_bad_lines="skip")
        # Clean column names: strip whitespace, quotes, and embedded newlines
        df.columns = [
            re.sub(r'\s+', ' ', c.strip().strip('"').replace('\n', ' ').replace('\t', ' '))
            for c in df.columns
        ]

        # Rename columns to consistent names
        rename_map: dict[str, str] = {}
        for c in df.columns:
            cl = c.lower()
            if "institute" in cl:
                rename_map[c] = "institute"
            elif "program" in cl and "total" not in cl:
                rename_map[c] = "program"
            elif "seat pool" in cl or "seat_pool" in cl:
                rename_map[c] = "seat_pool"
            elif "state" in cl and "seat" in cl:
                rename_map[c] = "quota_type"
        df = df.rename(columns=rename_map)

        # Forward-fill institute/program for continuation rows
        if "institute" in df.columns:
            df["institute"] = df["institute"].replace("", pd.NA).ffill()
        if "program" in df.columns:
            df["program"] = df["program"].replace("", pd.NA).ffill()

        # Melt category columns (OPEN, SC, ST, etc.) into rows
        target_cats = {
            "OPEN", "OPEN-PWD", "GEN-EWS", "GEN-EWS-PWD",
            "SC", "SC-PWD", "ST", "ST-PWD",
            "OBC-NCL", "OBC-NCL-PWD",
        }
        category_cols = [c for c in df.columns if c.upper().strip() in target_cats]
        # De-duplicate (in case cleaning made two columns identical)
        category_cols = list(dict.fromkeys(category_cols))

        if category_cols and "institute" in df.columns:
            id_cols = [c for c in ["institute", "program", "seat_pool", "quota_type"]
                       if c in df.columns]
            id_cols = [c for c in id_cols if c not in category_cols]
            melted = df.melt(
                id_vars=id_cols,
                value_vars=category_cols,
                var_name="category",
                value_name="seat_count",
            )
            melted["seat_count"] = pd.to_numeric(
                melted["seat_count"].str.replace(",", ""), errors="coerce"
            ).astype("Int64")
            melted = melted.dropna(subset=["seat_count"])
            log.info("Seat matrix: %d rows after melting", len(melted))
            return melted

        log.warning("Could not parse seat matrix columns; skipping")
        return None
    except Exception as e:
        log.warning("Failed to parse seat matrix: %s", e)
        return None


# =========================================================================
# 6. Percentile Normalization
# =========================================================================

def add_percentile_ranks(df: pd.DataFrame, dynamics: pd.DataFrame) -> pd.DataFrame:
    """
    Join with exam dynamics and compute percentile ranks.

    ``percentile = 100 * rank / total_applicants``

    This mirrors the formula in sim_engine's normalization.rs.
    """
    if dynamics.empty:
        log.warning("No exam dynamics data; skipping percentile normalization")
        return df

    df = df.merge(dynamics[["year", "applicants"]], on="year", how="left")

    for rank_col, pct_col in [
        ("opening_rank", "opening_percentile"),
        ("closing_rank", "closing_percentile"),
    ]:
        df[pct_col] = (df[rank_col] / df["applicants"] * 100).round(6)

    # Also add competitiveness ratio as a direct feature
    df["competitiveness_ratio"] = (df["closing_rank"] / df["applicants"]).round(6)

    log.info("Added percentile ranks for %d rows (%.1f%% matched to dynamics)",
              len(df), df["applicants"].notna().mean() * 100)
    return df


# =========================================================================
# 7. Feature Engineering
# =========================================================================

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features useful for modeling."""

    # --- Rank spread (within a single row) ---
    df["rank_spread"] = df["closing_rank"] - df["opening_rank"]

    # --- Group key for time-series features ---
    group_key = ["institute", "program", "category", "is_pwd", "gender", "quota"]

    # --- Year-over-year closing rank change (final round of each year) ---
    # Use the last round of each year for YoY comparison
    last_round = df.groupby(["year"] + group_key)["round"].transform("max")
    final_round_mask = df["round"] == last_round

    yoy = (
        df.loc[final_round_mask]
        .sort_values("year")
        .groupby(group_key)["closing_rank"]
        .diff()
    )
    df["closing_rank_yoy_change"] = pd.NA
    df.loc[yoy.index, "closing_rank_yoy_change"] = yoy.astype("Int64")

    # --- Round progression: closing rank delta from round 1 within a year ---
    round_prog = (
        df.sort_values("round")
        .groupby(["year"] + group_key)["closing_rank"]
        .diff()
    )
    df["round_progression"] = round_prog.astype("Int64")

    log.info("Engineered features: rank_spread, closing_rank_yoy_change, round_progression")
    return df


# =========================================================================
# 8. Validation
# =========================================================================

def validate(df: pd.DataFrame) -> list[str]:
    """
    Run comprehensive validation checks on the processed dataset.

    Returns a list of issues found (empty if all checks pass).
    Raises ``ValueError`` on critical failures.
    """
    issues: list[str] = []
    critical_errors: list[str] = []

    # --- Critical checks (raise on failure) ---

    # No null year/round
    null_years = df["year"].isna().sum()
    if null_years:
        critical_errors.append(f"Found {null_years} null year values")

    null_rounds = df["round"].isna().sum()
    if null_rounds:
        critical_errors.append(f"Found {null_rounds} null round values")

    # Closing ranks should be positive where present
    invalid_ranks = df["closing_rank"].dropna()
    n_bad = (invalid_ranks <= 0).sum()
    if n_bad:
        critical_errors.append(f"Found {n_bad} closing ranks ≤ 0")

    if critical_errors:
        for e in critical_errors:
            log.error("VALIDATION FAILED: %s", e)
        raise ValueError(f"Validation failed with {len(critical_errors)} critical error(s)")

    # --- Non-critical checks (warn and log) ---

    # Duplicate check on natural key
    dup_count = df.duplicated(subset=NATURAL_KEY).sum()
    if dup_count:
        issues.append(f"Found {dup_count} duplicate rows on natural key")

    # Rank columns are numeric (Int64)
    for col in ["opening_rank", "closing_rank"]:
        if col in df.columns and not pd.api.types.is_integer_dtype(df[col]):
            issues.append(f"Column '{col}' is not integer dtype (got {df[col].dtype})")

    # Year range check
    year_min, year_max = df["year"].min(), df["year"].max()
    if year_min < 2010 or year_max > 2030:
        issues.append(f"Suspicious year range: {year_min}–{year_max}")

    # Round range check
    round_min, round_max = df["round"].min(), df["round"].max()
    if round_min < 1 or round_max > 10:
        issues.append(f"Suspicious round range: {round_min}–{round_max}")

    # Categorical columns have known values
    known_categories = {"OPEN", "OBC-NCL", "SC", "ST", "EWS"}
    actual_categories = set(df["category"].dropna().unique())
    unknown_cats = actual_categories - known_categories
    if unknown_cats:
        issues.append(f"Unknown categories found: {unknown_cats}")

    known_genders = {"Gender-Neutral", "Female-only"}
    actual_genders = set(df["gender"].dropna().unique())
    unknown_genders = actual_genders - known_genders
    if unknown_genders:
        issues.append(f"Unknown genders found: {unknown_genders}")

    known_quotas = {"AI", "HS", "OS", "STATE"}
    actual_quotas = set(df["quota"].dropna().unique())
    unknown_quotas = actual_quotas - known_quotas
    if unknown_quotas:
        issues.append(f"Unknown quotas found: {unknown_quotas}")

    known_inst_types = {"IIT", "NIT", "IIIT", "GFTI"}
    actual_inst_types = set(df["institute_type"].dropna().unique())
    unknown_types = actual_inst_types - known_inst_types
    if unknown_types:
        issues.append(f"Unknown institute types found: {unknown_types}")

    # Opening rank <= closing rank (where both are present)
    both_present = df["opening_rank"].notna() & df["closing_rank"].notna()
    inverted = (df.loc[both_present, "opening_rank"] > df.loc[both_present, "closing_rank"]).sum()
    if inverted:
        issues.append(f"Found {inverted} rows where opening_rank > closing_rank")

    # Log results
    if issues:
        log.warning("Validation completed with %d issue(s):", len(issues))
        for issue in issues:
            log.warning("  ⚠ %s", issue)
    else:
        log.info("Validation passed ✓ — all checks clean")

    # Summary stats
    log.info("  Total rows: %d", len(df))
    log.info("  Year range: %d–%d", year_min, year_max)
    log.info("  Unique institutes: %d", df["institute"].nunique())
    log.info("  Unique programs: %d", df["program"].nunique())
    log.info("  Unique disciplines: %d", df["discipline"].nunique())

    year_counts = df.groupby("year").size()
    log.info("  Rows per year:")
    for year, count in year_counts.items():
        log.info("    %d: %6d rows", year, count)

    # Spot-checks
    legacy_gender = df.loc[df["year"].isin([2016, 2017]), "gender"]
    if len(legacy_gender) > 0:
        pct_neutral = (legacy_gender == "Gender-Neutral").mean() * 100
        log.info("  2016–2017 Gender-Neutral fill: %.1f%%", pct_neutral)

    recent = df.loc[df["year"] == 2026]
    if len(recent) > 0:
        has_space = recent["institute"].str.contains(r"\s+$", regex=True).any()
        log.info("  2026 trailing whitespace in institutes: %s",
                  "FOUND (BUG)" if has_space else "none ✓")

    return issues


# =========================================================================
# 9. Report & Documentation Generation
# =========================================================================

def generate_data_dictionary(df: pd.DataFrame, output_dir: Path) -> None:
    """Generate a markdown data dictionary documenting every column."""
    column_docs: dict[str, dict[str, str]] = {
        "year": {
            "description": "Calendar year of the JoSAA counselling round",
            "dtype": "int64",
            "transform": "Extracted from filename pattern `{year}_round_{n}.csv`",
        },
        "round": {
            "description": "Counselling round number within that year (1–7)",
            "dtype": "int64",
            "transform": "Extracted from filename pattern `{year}_round_{n}.csv`",
        },
        "institute": {
            "description": "Full name of the institute (e.g., 'Indian Institute of Technology Bombay')",
            "dtype": "string",
            "transform": "Whitespace stripped from raw `Institute` column",
        },
        "institute_type": {
            "description": "Institute classification: IIT, NIT, IIIT, or GFTI",
            "dtype": "string (categorical)",
            "transform": "Derived from institute name using keyword matching",
        },
        "program": {
            "description": "Program name without duration/degree (e.g., 'Computer Science and Engineering')",
            "dtype": "string",
            "transform": "Parsed from raw `Academic Program Name` via regex, removing parenthetical suffix",
        },
        "discipline": {
            "description": "Core discipline / field of study (currently same as program name)",
            "dtype": "string",
            "transform": "Extracted from `Academic Program Name`; equals program name after parsing",
        },
        "duration_years": {
            "description": "Program duration in years (e.g., 4, 5)",
            "dtype": "Int64 (nullable)",
            "transform": "Parsed from `Academic Program Name` parenthetical: '(4 Years, ...)'",
        },
        "degree_type": {
            "description": "Degree conferred (e.g., 'Bachelor of Technology', 'Master of Technology')",
            "dtype": "string (nullable)",
            "transform": "Parsed from `Academic Program Name` parenthetical: '(..., Bachelor of Technology)'",
        },
        "quota": {
            "description": "Seat quota type: AI (All India), HS (Home State), OS (Other State), STATE (state-specific)",
            "dtype": "string (categorical)",
            "transform": "Normalized from raw `Quota` column; state-specific codes collapsed to STATE",
        },
        "category": {
            "description": "Reservation category: OPEN, OBC-NCL, SC, ST, EWS",
            "dtype": "string (categorical)",
            "transform": "Decomposed from raw `Seat Type`, PwD suffix removed and tracked separately",
        },
        "is_pwd": {
            "description": "Whether this row is for PwD (Persons with Disability) candidates",
            "dtype": "bool",
            "transform": "Derived from `Seat Type` containing 'PwD' or '(PwD)'",
        },
        "gender": {
            "description": "Gender pool: 'Gender-Neutral' or 'Female-only'",
            "dtype": "string (categorical)",
            "transform": "Normalized from raw `Gender` column; legacy files (2016–17) default to Gender-Neutral",
        },
        "opening_rank": {
            "description": "JEE Advanced/Main rank of the first admitted candidate in this slot",
            "dtype": "Int64 (nullable integer)",
            "transform": "Parsed from raw string; commas removed, floats truncated, PwD suffix stripped",
        },
        "closing_rank": {
            "description": "JEE Advanced/Main rank of the last admitted candidate in this slot",
            "dtype": "Int64 (nullable integer)",
            "transform": "Same as opening_rank",
        },
        "is_pwd_rank": {
            "description": "Whether ranks had a PwD suffix ('P') in the raw data",
            "dtype": "bool",
            "transform": "True if either opening or closing rank had trailing 'P'",
        },
        "source_file": {
            "description": "Name of the raw CSV file this row was loaded from",
            "dtype": "string",
            "transform": "Filename only (no directory path), e.g., '2024_round_3.csv'",
        },
        "source_row_id": {
            "description": "0-based row index within the source file (for traceability)",
            "dtype": "int64",
            "transform": "Assigned during loading; preserves original row order",
        },
        # Feature-engineered columns (master output only)
        "opening_percentile": {
            "description": "Opening rank as percentile of total JEE applicants",
            "dtype": "float64 (nullable)",
            "transform": "100 × opening_rank / applicants (from jee_exam_dynamics.csv)",
        },
        "closing_percentile": {
            "description": "Closing rank as percentile of total JEE applicants",
            "dtype": "float64 (nullable)",
            "transform": "100 × closing_rank / applicants",
        },
        "competitiveness_ratio": {
            "description": "Closing rank divided by total applicants (0–1 scale)",
            "dtype": "float64 (nullable)",
            "transform": "closing_rank / applicants",
        },
        "applicants": {
            "description": "Total JEE registered applicants for that year",
            "dtype": "Int64 (nullable)",
            "transform": "Joined from jee_exam_dynamics.csv on year",
        },
        "rank_spread": {
            "description": "Difference between closing and opening rank within a single row",
            "dtype": "Int64 (nullable)",
            "transform": "closing_rank − opening_rank",
        },
        "closing_rank_yoy_change": {
            "description": "Year-over-year change in closing rank (final round only)",
            "dtype": "Int64 (nullable)",
            "transform": "Diff of closing_rank grouped by (institute, program, category, is_pwd, gender, quota)",
        },
        "round_progression": {
            "description": "Round-to-round change in closing rank within a year",
            "dtype": "Int64 (nullable)",
            "transform": "Diff of closing_rank grouped by year and branch key, ordered by round",
        },
    }

    lines = [
        "# JoSAA Cleaned Dataset — Data Dictionary",
        "",
        f"*Generated by the preprocessing pipeline. {len(df):,} rows × {len(df.columns)} columns.*",
        "",
        "---",
        "",
    ]

    # Document columns actually present in the DataFrame
    for col in df.columns:
        doc = column_docs.get(col, {
            "description": "(Undocumented column)",
            "dtype": str(df[col].dtype),
            "transform": "None",
        })
        lines.extend([
            f"## `{col}`",
            "",
            f"- **Description**: {doc['description']}",
            f"- **Data type**: `{doc['dtype']}`",
            f"- **Transformation**: {doc['transform']}",
            f"- **Non-null count**: {df[col].notna().sum():,} / {len(df):,} "
            f"({df[col].notna().mean() * 100:.1f}%)",
            f"- **Unique values**: {df[col].nunique():,}",
            "",
        ])

    path = output_dir / "data_dictionary.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    log.info("Wrote data_dictionary.md")


def generate_summary_report(
    df_raw: pd.DataFrame,
    df_cleaned: pd.DataFrame,
    df_master: pd.DataFrame,
    issues: list[str],
    output_dir: Path,
) -> None:
    """Generate a markdown summary report of the cleaning process."""
    lines = [
        "# JoSAA Preprocessing — Summary Report",
        "",
        "---",
        "",
        "## Dataset Size",
        "",
        f"| Metric | Value |",
        f"|---|---|",
        f"| Raw rows (before cleaning) | {len(df_raw):,} |",
        f"| Cleaned rows | {len(df_cleaned):,} |",
        f"| Rows dropped | {len(df_raw) - len(df_cleaned):,} |",
        f"| Drop rate | {(1 - len(df_cleaned) / len(df_raw)) * 100:.2f}% |",
        f"| Columns (cleaned) | {len(df_cleaned.columns)} |",
        f"| Columns (master w/ features) | {len(df_master.columns)} |",
        "",
        "## Missing Values (cleaned dataset)",
        "",
        "| Column | Missing | % Missing |",
        "|---|---|---|",
    ]

    for col in df_cleaned.columns:
        n_missing = df_cleaned[col].isna().sum()
        pct = n_missing / len(df_cleaned) * 100
        lines.append(f"| `{col}` | {n_missing:,} | {pct:.2f}% |")

    lines.extend([
        "",
        "## Unique Value Counts",
        "",
        f"| Dimension | Unique Count |",
        f"|---|---|",
        f"| Years | {df_cleaned['year'].nunique()} ({df_cleaned['year'].min()}–{df_cleaned['year'].max()}) |",
        f"| Rounds | {df_cleaned['round'].nunique()} (1–{df_cleaned['round'].max()}) |",
        f"| Institutes | {df_cleaned['institute'].nunique():,} |",
        f"| Institute types | {df_cleaned['institute_type'].nunique()} |",
        f"| Programs | {df_cleaned['program'].nunique():,} |",
        f"| Disciplines | {df_cleaned['discipline'].nunique():,} |",
        f"| Categories | {df_cleaned['category'].nunique()} |",
        f"| Quotas | {df_cleaned['quota'].nunique()} |",
    ])

    # Category breakdown
    lines.extend([
        "",
        "## Category Distribution",
        "",
        "| Category | Count | % |",
        "|---|---|---|",
    ])
    for cat, count in df_cleaned["category"].value_counts().items():
        lines.append(f"| {cat} | {count:,} | {count / len(df_cleaned) * 100:.1f}% |")

    # Institute type breakdown
    lines.extend([
        "",
        "## Institute Type Distribution",
        "",
        "| Type | Count | % |",
        "|---|---|---|",
    ])
    for itype, count in df_cleaned["institute_type"].value_counts().items():
        lines.append(f"| {itype} | {count:,} | {count / len(df_cleaned) * 100:.1f}% |")

    # Year distribution
    lines.extend([
        "",
        "## Rows per Year",
        "",
        "| Year | Rows | Rounds |",
        "|---|---|---|",
    ])
    for year, group in df_cleaned.groupby("year"):
        rounds = sorted(group["round"].unique())
        lines.append(f"| {year} | {len(group):,} | {len(rounds)} ({min(rounds)}–{max(rounds)}) |")

    # Quality issues
    lines.extend([
        "",
        "## Quality Issues Found",
        "",
    ])
    if issues:
        for issue in issues:
            lines.append(f"- ⚠️ {issue}")
    else:
        lines.append("✅ No quality issues detected.")

    # Columns kept/dropped
    raw_cols = {"Institute", "Academic Program Name", "Quota", "Seat Type",
                "Gender", "Opening Rank", "Closing Rank"}
    lines.extend([
        "",
        "## Columns",
        "",
        "### Raw columns consumed",
        "",
    ])
    for c in sorted(raw_cols):
        lines.append(f"- `{c}`")

    lines.extend([
        "",
        "### Final columns produced (cleaned dataset)",
        "",
    ])
    for c in df_cleaned.columns:
        lines.append(f"- `{c}`")

    lines.extend([
        "",
        "### Additional columns in master dataset",
        "",
    ])
    extra = set(df_master.columns) - set(df_cleaned.columns)
    for c in sorted(extra):
        lines.append(f"- `{c}`")

    lines.extend([
        "",
        "### Columns dropped during cleaning",
        "",
        "The following raw columns were consumed and not carried to the output:",
        "",
    ])
    for c in sorted(raw_cols):
        lines.append(f"- `{c}` → decomposed into normalized columns")

    # What the data is ready for
    lines.extend([
        "",
        "---",
        "",
        "## Ready for Day 2",
        "",
        "The cleaned dataset is now suitable for:",
        "",
        "1. **Cutoff prediction models** — XGBoost / LightGBM regression on closing ranks",
        "2. **Monte Carlo simulation** — branch time-series data ready for stochastic modeling",
        "3. **Recommendation ranking** — per-candidate admission probability estimation",
        "4. **Trend analysis** — year-over-year and round-progression features already computed",
        "5. **Category-wise analysis** — standardized categories enable clean group-by operations",
        "",
        "The `branch_timeseries.parquet` is directly usable as input for the sim_engine's "
        "`BranchCutoffHistory` structure.",
    ])

    path = output_dir / "summary_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    log.info("Wrote summary_report.md")


def export_normalization_mappings(output_dir: Path) -> None:
    """Save all normalization mappings as JSON files for reproducibility."""
    mappings_dir = output_dir / "mappings"
    mappings_dir.mkdir(parents=True, exist_ok=True)

    mapping_files = {
        "category_mapping.json": dict(sorted(_category_mappings.items())),
        "institute_types.json": dict(sorted(_institute_type_mappings.items())),
        "quota_mapping.json": dict(sorted(_quota_mappings.items())),
        "gender_mapping.json": dict(sorted(_gender_mappings.items())),
    }

    for filename, data in mapping_files.items():
        path = mappings_dir / filename
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        log.info("  Wrote mappings/%s  (%d entries)", filename, len(data))

    log.info("Exported %d normalization mapping files", len(mapping_files))


# =========================================================================
# 10. Branch Time-Series Export
# =========================================================================

def _export_branch_timeseries(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Create a pivoted time-series: one row per branch, closing ranks as columns by year.

    Uses the LAST round of each year (final cutoff).
    Direct input for the Monte Carlo predictor's BranchCutoffHistory.
    """
    group_key = ["institute", "program", "category", "is_pwd", "gender", "quota"]

    # Get the final round per year per branch
    last_round_idx = df.groupby(["year"] + group_key)["round"].idxmax()
    final = df.loc[last_round_idx]

    # Pivot: rows = branches, columns = years
    pivot = final.pivot_table(
        index=group_key,
        columns="year",
        values="closing_rank",
        aggfunc="first",
    )
    pivot.columns = [f"closing_rank_{int(y)}" for y in pivot.columns]
    pivot = pivot.reset_index()

    pivot.to_parquet(output_dir / "branch_timeseries.parquet", index=False, engine="pyarrow")
    log.info("Wrote branch_timeseries.parquet  (%d branches × %d year-columns)",
              len(pivot), sum(1 for c in pivot.columns if c.startswith("closing_rank_")))


# =========================================================================
# 11. Train/Test Split
# =========================================================================

def _split_dataset(df: pd.DataFrame, output_dir: Path, test_start_year: int = 2025) -> None:
    """
    Temporal train/test split.

    Train: years < test_start_year
    Test:  years >= test_start_year

    Avoids data leakage from using future rounds to predict past ones.
    """
    train = df[df["year"] < test_start_year]
    test = df[df["year"] >= test_start_year]

    train.to_parquet(output_dir / "train.parquet", index=False, engine="pyarrow")
    test.to_parquet(output_dir / "test.parquet", index=False, engine="pyarrow")

    log.info("Train/test split (boundary: %d):", test_start_year)
    log.info("  Train: %d rows  (years %d–%d)", len(train),
              train["year"].min(), train["year"].max())
    log.info("  Test:  %d rows  (years %d–%d)", len(test),
              test["year"].min(), test["year"].max())


# =========================================================================
# 12. Master Pipeline
# =========================================================================

def build_master_dataset(data_dir: Path, output_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    Full pipeline: load → backup → clean → normalize → dedup → validate →
    feature-engineer → export.

    Returns the master DataFrame.
    """
    if output_dir is None:
        output_dir = data_dir / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Step 1: Load all round files ---
    log.info("=" * 60)
    log.info("STEP 1: Loading round files")
    log.info("=" * 60)
    df = load_all_round_files(data_dir)
    raw_row_count = len(df)

    # --- Step 2: Save raw backup ---
    log.info("=" * 60)
    log.info("STEP 2: Saving raw backup")
    log.info("=" * 60)
    df.to_csv(output_dir / "raw_backup.csv", index=False)
    log.info("Wrote raw_backup.csv  (%d rows)", len(df))
    df_raw = df.copy()  # keep a copy for the summary report

    # --- Step 3: Clean ranks ---
    log.info("=" * 60)
    log.info("STEP 3: Cleaning ranks")
    log.info("=" * 60)
    df = clean_ranks(df)

    # --- Step 4: Normalize categories ---
    log.info("=" * 60)
    log.info("STEP 4: Normalizing categories")
    log.info("=" * 60)
    df = normalize_categories(df)

    # --- Step 5: Remove duplicates ---
    log.info("=" * 60)
    log.info("STEP 5: Removing duplicates")
    log.info("=" * 60)
    df = remove_duplicates(df)

    # --- Step 6: Select cleaned columns and save ---
    log.info("=" * 60)
    log.info("STEP 6: Saving cleaned dataset (josaa_cleaned.csv)")
    log.info("=" * 60)
    cleaned_cols = [c for c in CLEANED_COLUMNS if c in df.columns]
    df_cleaned = df[cleaned_cols].copy()

    df_cleaned.to_csv(output_dir / "josaa_cleaned.csv", index=False)
    log.info("Wrote josaa_cleaned.csv  (%d rows × %d cols)", len(df_cleaned), len(df_cleaned.columns))

    # --- Step 7: Validation ---
    log.info("=" * 60)
    log.info("STEP 7: Validation")
    log.info("=" * 60)
    issues = validate(df_cleaned)

    # --- Step 8: Load auxiliary data ---
    log.info("=" * 60)
    log.info("STEP 8: Loading auxiliary data")
    log.info("=" * 60)
    dynamics = load_exam_dynamics(data_dir)
    seat_matrix = load_seat_matrix(data_dir)

    # --- Step 9: Percentile normalization ---
    log.info("=" * 60)
    log.info("STEP 9: Percentile normalization")
    log.info("=" * 60)
    df = add_percentile_ranks(df, dynamics)

    # --- Step 10: Feature engineering ---
    log.info("=" * 60)
    log.info("STEP 10: Feature engineering")
    log.info("=" * 60)
    df = engineer_features(df)

    # --- Step 11: Build master output ---
    log.info("=" * 60)
    log.info("STEP 11: Exporting master dataset")
    log.info("=" * 60)

    # Assemble master output columns
    all_master_cols = CLEANED_COLUMNS + MASTER_EXTRA_COLUMNS
    # Only keep columns that actually exist
    output_cols = [c for c in all_master_cols if c in df.columns]
    # De-duplicate
    seen: set[str] = set()
    deduped: list[str] = []
    for c in output_cols:
        if c not in seen:
            deduped.append(c)
            seen.add(c)
    output_cols = deduped

    master = df[output_cols].copy()

    master.to_parquet(output_dir / "master_cutoffs.parquet", index=False, engine="pyarrow")
    master.to_csv(output_dir / "master_cutoffs.csv", index=False)
    log.info("Wrote master_cutoffs.parquet  (%d rows × %d cols)", len(master), len(master.columns))
    log.info("Wrote master_cutoffs.csv")

    # --- Step 12: Branch time-series (pivoted) ---
    _export_branch_timeseries(master, output_dir)

    # --- Step 13: Seat matrix ---
    if seat_matrix is not None:
        seat_matrix.to_parquet(output_dir / "seat_matrix.parquet", index=False, engine="pyarrow")
        log.info("Wrote seat_matrix.parquet  (%d rows)", len(seat_matrix))

    # --- Step 14: Train/test split ---
    _split_dataset(master, output_dir)

    # --- Step 15: Export normalization mappings ---
    log.info("=" * 60)
    log.info("STEP 12: Exporting normalization mappings")
    log.info("=" * 60)
    export_normalization_mappings(output_dir)

    # --- Step 16: Generate data dictionary ---
    log.info("=" * 60)
    log.info("STEP 13: Generating data dictionary")
    log.info("=" * 60)
    generate_data_dictionary(master, output_dir)

    # --- Step 17: Generate summary report ---
    log.info("=" * 60)
    log.info("STEP 14: Generating summary report")
    log.info("=" * 60)
    generate_summary_report(df_raw, df_cleaned, master, issues, output_dir)

    # --- Final summary ---
    log.info("=" * 60)
    log.info("PIPELINE COMPLETE")
    log.info("=" * 60)
    log.info("  Raw rows:      %d", raw_row_count)
    log.info("  Cleaned rows:  %d", len(df_cleaned))
    log.info("  Master rows:   %d", len(master))
    log.info("  Columns:       %d (cleaned) / %d (master)", len(df_cleaned.columns), len(master.columns))
    log.info("  Output dir:    %s", output_dir)
    log.info("")
    log.info("  Files produced:")
    for f in sorted(output_dir.rglob("*")):
        if f.is_file():
            size_mb = f.stat().st_size / (1024 * 1024)
            log.info("    %-40s  %7.2f MB", f.relative_to(output_dir), size_mb)

    return master


# =========================================================================
# CLI
# =========================================================================

def main() -> None:
    """Command-line entry point for the preprocessing pipeline."""
    parser = argparse.ArgumentParser(
        description="JoSAA cutoff data preprocessing pipeline",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).parent / "data",
        help="Directory containing the raw CSV files (default: ./data)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: <data-dir>/processed)",
    )
    args = parser.parse_args()

    build_master_dataset(args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()
