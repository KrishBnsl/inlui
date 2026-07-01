"""
JoSAA Day 2 — Exploratory Data Analysis & Feature Engineering
==============================================================

Reads the Day 1 master dataset, produces:
  • EDA summary report       → reports/eda_summary.md
  • 10+ publication-quality charts → figures/
  • Feature-engineered dataset → data/features/feature_engineered.csv
  • Regression train/valid/test splits → data/features/regression_*.csv
  • Classification framework splits  → data/features/classification_*.csv
  • Encoders & mappings             → model_artifacts/

Usage:
    python eda_features.py                        # defaults
    python eda_features.py --data-dir ./data      # custom

The module is also fully importable:
    from eda_features import run_day2_pipeline
"""

from __future__ import annotations

import argparse
import json
import logging
import warnings
from pathlib import Path
from typing import Any, Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server/CI

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Plot style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
})
sns.set_theme(style="whitegrid", palette="deep")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Group key for branch-level aggregations
BRANCH_KEY = ["institute", "program", "category", "is_pwd", "gender", "quota"]

# Categorical columns to encode
CATEGORICAL_COLS = ["institute_type", "category", "quota", "gender", "degree_type"]

# Temporal split boundaries
TRAIN_END_YEAR = 2023      # train: ≤ 2023
VALID_YEARS = [2024]       # valid: 2024
TEST_YEARS = [2025, 2026]  # test:  2025-2026


# =========================================================================
# 1. Data Loading
# =========================================================================

def load_master(data_dir: Path) -> pd.DataFrame:
    """
    Load the master cutoffs dataset produced by Day 1.

    Tries Parquet first (faster, preserves dtypes), falls back to CSV.
    """
    parquet_path = data_dir / "processed" / "master_cutoffs.parquet"
    csv_path = data_dir / "processed" / "master_cutoffs.csv"

    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
        log.info("Loaded master_cutoffs.parquet → %d rows × %d cols", len(df), len(df.columns))
    elif csv_path.exists():
        df = pd.read_csv(csv_path)
        log.info("Loaded master_cutoffs.csv → %d rows × %d cols", len(df), len(df.columns))
    else:
        raise FileNotFoundError(
            f"Neither {parquet_path} nor {csv_path} found. Run preprocessing.py first."
        )

    return df


# =========================================================================
# 2. EDA Summary
# =========================================================================

def compute_eda_summary(df: pd.DataFrame) -> dict[str, Any]:
    """Compute comprehensive EDA statistics and return as a dictionary."""
    summary: dict[str, Any] = {}

    # --- Basic counts ---
    summary["total_records"] = len(df)
    summary["n_columns"] = len(df.columns)
    summary["columns"] = list(df.columns)

    for col in ["institute", "program", "discipline", "category",
                 "institute_type", "quota", "gender"]:
        if col in df.columns:
            summary[f"unique_{col}"] = int(df[col].nunique())

    summary["unique_years"] = sorted(df["year"].unique().tolist())
    summary["n_years"] = len(summary["unique_years"])
    summary["unique_rounds"] = sorted(df["round"].unique().tolist())
    summary["n_rounds"] = len(summary["unique_rounds"])

    # --- Missingness ---
    missing = df.isnull().sum()
    summary["missing_values"] = {
        col: int(missing[col]) for col in df.columns if missing[col] > 0
    }
    summary["total_missing"] = int(missing.sum())

    # --- Numeric summaries ---
    for col in ["closing_rank", "opening_rank", "rank_spread"]:
        if col in df.columns:
            vals = df[col].dropna()
            summary[f"{col}_stats"] = {
                "mean": round(float(vals.mean()), 1),
                "median": round(float(vals.median()), 1),
                "std": round(float(vals.std()), 1),
                "min": int(vals.min()),
                "max": int(vals.max()),
                "q25": round(float(vals.quantile(0.25)), 1),
                "q75": round(float(vals.quantile(0.75)), 1),
            }

    # --- Top institutes by record count ---
    inst_counts = df["institute"].value_counts().head(15)
    summary["top_institutes"] = {
        name: int(count) for name, count in inst_counts.items()
    }

    # --- Top programs by record count ---
    prog_counts = df["program"].value_counts().head(15)
    summary["top_programs"] = {
        name: int(count) for name, count in prog_counts.items()
    }

    # --- Rows per year ---
    year_counts = df.groupby("year").size()
    summary["rows_per_year"] = {
        int(y): int(c) for y, c in year_counts.items()
    }

    # --- Volatility analysis (final-round closing rank std by branch) ---
    # Use last round of each year for stable YoY comparison
    last_round_idx = df.groupby(["year"] + BRANCH_KEY)["round"].idxmax()
    final_df = df.loc[last_round_idx]

    vol = (
        final_df.groupby(BRANCH_KEY)["closing_rank"]
        .agg(["mean", "std", "count"])
        .dropna()
    )
    vol = vol[vol["count"] >= 3]  # need ≥ 3 years for meaningful std
    vol["cv"] = vol["std"] / vol["mean"]  # coefficient of variation

    if len(vol) > 0:
        most_volatile = vol.nlargest(10, "cv")
        most_stable = vol.nsmallest(10, "cv")
        summary["most_volatile_branches"] = [
            {
                "institute": idx[0],
                "program": idx[1],
                "category": idx[2],
                "cv": round(float(row["cv"]), 4),
                "mean_rank": round(float(row["mean"]), 0),
                "std_rank": round(float(row["std"]), 0),
                "n_years": int(row["count"]),
            }
            for idx, row in most_volatile.iterrows()
        ]
        summary["most_stable_branches"] = [
            {
                "institute": idx[0],
                "program": idx[1],
                "category": idx[2],
                "cv": round(float(row["cv"]), 4),
                "mean_rank": round(float(row["mean"]), 0),
                "std_rank": round(float(row["std"]), 0),
                "n_years": int(row["count"]),
            }
            for idx, row in most_stable.iterrows()
        ]

    # --- Category distribution ---
    cat_counts = df["category"].value_counts()
    summary["category_distribution"] = {
        str(k): int(v) for k, v in cat_counts.items()
    }

    # --- Institute type distribution ---
    itype_counts = df["institute_type"].value_counts()
    summary["institute_type_distribution"] = {
        str(k): int(v) for k, v in itype_counts.items()
    }

    log.info("EDA summary computed")
    return summary


def write_eda_report(summary: dict[str, Any], output_path: Path) -> None:
    """Write a markdown-formatted EDA report from the summary dict."""
    lines: list[str] = []

    lines.extend([
        "# JoSAA Day 2 — Exploratory Data Analysis Report",
        "",
        "---",
        "",
        "## Dataset Overview",
        "",
        f"| Metric | Value |",
        f"|---|---|",
        f"| Total records | {summary['total_records']:,} |",
        f"| Columns | {summary['n_columns']} |",
        f"| Years | {summary['n_years']} ({summary['unique_years'][0]}–{summary['unique_years'][-1]}) |",
        f"| Rounds | {summary['n_rounds']} (1–{max(summary['unique_rounds'])}) |",
    ])

    for dim in ["institute", "program", "discipline", "category",
                 "institute_type", "quota", "gender"]:
        key = f"unique_{dim}"
        if key in summary:
            lines.append(f"| Unique {dim}s | {summary[key]:,} |")

    # Missingness
    lines.extend(["", "## Missing Values", ""])
    if summary["total_missing"] == 0:
        lines.append("✅ No missing values in the dataset.")
    else:
        lines.extend(["| Column | Missing Count |", "|---|---|"])
        for col, cnt in summary["missing_values"].items():
            lines.append(f"| `{col}` | {cnt:,} |")

    # Numeric summaries
    for col in ["closing_rank", "opening_rank", "rank_spread"]:
        key = f"{col}_stats"
        if key in summary:
            s = summary[key]
            lines.extend([
                "",
                f"## `{col}` Statistics",
                "",
                f"| Stat | Value |",
                f"|---|---|",
                f"| Mean | {s['mean']:,.1f} |",
                f"| Median | {s['median']:,.1f} |",
                f"| Std Dev | {s['std']:,.1f} |",
                f"| Min | {s['min']:,} |",
                f"| Max | {s['max']:,} |",
                f"| Q25 | {s['q25']:,.1f} |",
                f"| Q75 | {s['q75']:,.1f} |",
            ])

    # Top institutes
    lines.extend(["", "## Top 15 Institutes by Record Count", "",
                   "| Rank | Institute | Records |", "|---|---|---|"])
    for i, (name, cnt) in enumerate(summary["top_institutes"].items(), 1):
        short = name[:65] + "..." if len(name) > 65 else name
        lines.append(f"| {i} | {short} | {cnt:,} |")

    # Top programs
    lines.extend(["", "## Top 15 Programs by Record Count", "",
                   "| Rank | Program | Records |", "|---|---|---|"])
    for i, (name, cnt) in enumerate(summary["top_programs"].items(), 1):
        short = name[:65] + "..." if len(name) > 65 else name
        lines.append(f"| {i} | {short} | {cnt:,} |")

    # Rows per year
    lines.extend(["", "## Rows per Year", "",
                   "| Year | Rows |", "|---|---|"])
    for year, cnt in summary["rows_per_year"].items():
        lines.append(f"| {year} | {cnt:,} |")

    # Category distribution
    lines.extend(["", "## Category Distribution", "",
                   "| Category | Count | % |", "|---|---|---|"])
    total = summary["total_records"]
    for cat, cnt in summary["category_distribution"].items():
        lines.append(f"| {cat} | {cnt:,} | {cnt / total * 100:.1f}% |")

    # Volatility
    if "most_volatile_branches" in summary:
        lines.extend(["", "## Most Volatile Branches (highest CV of closing rank)", "",
                       "| Institute | Program | Category | CV | Mean Rank | Std | Years |",
                       "|---|---|---|---|---|---|---|"])
        for b in summary["most_volatile_branches"]:
            inst = b["institute"][:40] + "..." if len(b["institute"]) > 40 else b["institute"]
            prog = b["program"][:30] + "..." if len(b["program"]) > 30 else b["program"]
            lines.append(
                f"| {inst} | {prog} | {b['category']} "
                f"| {b['cv']:.4f} | {b['mean_rank']:,.0f} "
                f"| {b['std_rank']:,.0f} | {b['n_years']} |"
            )

    if "most_stable_branches" in summary:
        lines.extend(["", "## Most Stable Branches (lowest CV of closing rank)", "",
                       "| Institute | Program | Category | CV | Mean Rank | Std | Years |",
                       "|---|---|---|---|---|---|---|"])
        for b in summary["most_stable_branches"]:
            inst = b["institute"][:40] + "..." if len(b["institute"]) > 40 else b["institute"]
            prog = b["program"][:30] + "..." if len(b["program"]) > 30 else b["program"]
            lines.append(
                f"| {inst} | {prog} | {b['category']} "
                f"| {b['cv']:.4f} | {b['mean_rank']:,.0f} "
                f"| {b['std_rank']:,.0f} | {b['n_years']} |"
            )

    # Modeling recommendations
    lines.extend([
        "",
        "---",
        "",
        "## Key Patterns & Modeling Recommendations",
        "",
        "### Promising Features for Cutoff Prediction",
        "",
        "1. **Historical closing rank** (group mean/std) — strongest signal for branch-level prediction",
        "2. **Year trend** — cutoffs shift systematically as JEE applicant pool grows",
        "3. **Round number** — later rounds generally have relaxed cutoffs",
        "4. **Institute type** — IIT/NIT/IIIT/GFTI have very different rank ranges",
        "5. **Category** — OPEN vs. reserved categories have distinct distributions",
        "6. **Competitiveness ratio** — normalizes across years with different applicant counts",
        "7. **YoY change & round progression** — capture momentum/trends",
        "",
        "### Recommended Model Families",
        "",
        "1. **Gradient Boosted Trees (XGBoost / LightGBM)** — best for tabular data with mixed types",
        "2. **Random Forest** — solid baseline, handles non-linearity well",
        "3. **Linear models (Ridge/Lasso)** — useful for interpretability and as a baseline",
        "4. **Quantile Regression** — for prediction intervals (important for counselling)",
        "",
        "### Data Quality Notes for Modeling",
        "",
        "- 785 rows have `opening_rank > closing_rank` (legitimate JoSAA data, not errors)",
        "- 2016–2017 data has no gender column (filled as Gender-Neutral)",
        "- 2026 has only 1 round so far — limited for testing",
        "- EWS category only appears from 2019 onwards",
        "- PwD rows have very different rank scales — consider modeling separately",
    ])

    output_path.write_text("\n".join(lines), encoding="utf-8")
    log.info("Wrote EDA report → %s", output_path)


# =========================================================================
# 3. Chart Generation
# =========================================================================

def generate_charts(df: pd.DataFrame, fig_dir: Path) -> None:
    """Generate all EDA charts and save to fig_dir."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    log.info("Generating charts → %s", fig_dir)

    # Filter to non-PwD, Gender-Neutral, AI quota for cleaner plots
    mainstream = df[
        (~df["is_pwd"]) & (df["gender"] == "Gender-Neutral") & (df["quota"] == "AI")
    ].copy()

    # 1. Closing rank distribution
    _plot_rank_distribution(df, fig_dir)

    # 2. Closing rank by year (boxplot)
    _plot_rank_by_year(mainstream, fig_dir)

    # 3. Closing rank by round (boxplot)
    _plot_rank_by_round(mainstream, fig_dir)

    # 4. Category-wise cutoff comparison
    _plot_category_comparison(df, fig_dir)

    # 5. Institute type comparison
    _plot_institute_type_comparison(df, fig_dir)

    # 6. Top IIT branch trends
    _plot_top_branch_trends(mainstream, fig_dir)

    # 7. Opening vs Closing rank scatter
    _plot_opening_vs_closing(mainstream, fig_dir)

    # 8. Correlation heatmap
    _plot_correlation_heatmap(df, fig_dir)

    # 9. Round progression heatmap
    _plot_round_heatmap(mainstream, fig_dir)

    # 10. Year-over-year change distribution
    _plot_yoy_distribution(df, fig_dir)

    log.info("Generated 10 charts")


def _plot_rank_distribution(df: pd.DataFrame, fig_dir: Path) -> None:
    """Closing and opening rank distributions (overlaid histograms)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.hist(df["closing_rank"].dropna().astype(float), bins=100,
            alpha=0.7, color="#2196F3", edgecolor="none")
    ax.set_title("Closing Rank Distribution")
    ax.set_xlabel("Closing Rank")
    ax.set_ylabel("Count")
    ax.axvline(df["closing_rank"].median(), color="red", linestyle="--",
               label=f"Median: {int(df['closing_rank'].median()):,}")
    ax.legend()

    ax = axes[1]
    ax.hist(df["opening_rank"].dropna().astype(float), bins=100,
            alpha=0.7, color="#4CAF50", edgecolor="none")
    ax.set_title("Opening Rank Distribution")
    ax.set_xlabel("Opening Rank")
    ax.set_ylabel("Count")
    ax.axvline(df["opening_rank"].median(), color="red", linestyle="--",
               label=f"Median: {int(df['opening_rank'].median()):,}")
    ax.legend()

    plt.suptitle("Rank Distributions (All Data)", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(fig_dir / "01_rank_distributions.png")
    plt.close()


def _plot_rank_by_year(df: pd.DataFrame, fig_dir: Path) -> None:
    """Closing rank by year — boxplot."""
    fig, ax = plt.subplots(figsize=(12, 6))
    year_order = sorted(df["year"].unique())
    sns.boxplot(data=df, x="year", y="closing_rank", order=year_order,
                ax=ax, fliersize=1, palette="viridis", linewidth=0.8)
    ax.set_title("Closing Rank by Year (AI · Gender-Neutral · Non-PwD)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Closing Rank")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.savefig(fig_dir / "02_closing_rank_by_year.png")
    plt.close()


def _plot_rank_by_round(df: pd.DataFrame, fig_dir: Path) -> None:
    """Closing rank by round — boxplot."""
    fig, ax = plt.subplots(figsize=(10, 6))
    round_order = sorted(df["round"].unique())
    sns.boxplot(data=df, x="round", y="closing_rank", order=round_order,
                ax=ax, fliersize=1, palette="coolwarm", linewidth=0.8)
    ax.set_title("Closing Rank by Round (AI · Gender-Neutral · Non-PwD)")
    ax.set_xlabel("Round")
    ax.set_ylabel("Closing Rank")
    plt.tight_layout()
    plt.savefig(fig_dir / "03_closing_rank_by_round.png")
    plt.close()


def _plot_category_comparison(df: pd.DataFrame, fig_dir: Path) -> None:
    """Category-wise closing rank comparison — violin plot."""
    # Filter to AI quota, non-PwD for fair comparison
    subset = df[(~df["is_pwd"]) & (df["quota"] == "AI")].copy()
    cat_order = ["OPEN", "EWS", "OBC-NCL", "SC", "ST"]
    subset = subset[subset["category"].isin(cat_order)]

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.violinplot(data=subset, x="category", y="closing_rank",
                   order=cat_order, ax=ax, palette="Set2",
                   inner="quartile", cut=0, linewidth=0.8)
    ax.set_title("Closing Rank by Category (AI Quota · Non-PwD)")
    ax.set_xlabel("Category")
    ax.set_ylabel("Closing Rank")
    plt.tight_layout()
    plt.savefig(fig_dir / "04_category_comparison.png")
    plt.close()


def _plot_institute_type_comparison(df: pd.DataFrame, fig_dir: Path) -> None:
    """Institute type closing rank comparison — boxplot."""
    subset = df[(~df["is_pwd"]) & (df["gender"] == "Gender-Neutral") & (df["quota"].isin(["AI", "OS"]))].copy()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    type_order = ["IIT", "NIT", "IIIT", "GFTI"]
    sns.boxplot(data=subset, x="institute_type", y="closing_rank",
                order=type_order, ax=ax, palette="Pastel1",
                fliersize=1, linewidth=0.8)
    ax.set_title("Closing Rank by Institute Type (AI/OS · Gender-Neutral · Non-PwD)")
    ax.set_xlabel("Institute Type")
    ax.set_ylabel("Closing Rank")
    plt.tight_layout()
    plt.savefig(fig_dir / "05_institute_type_comparison.png")
    plt.close()


def _plot_top_branch_trends(df: pd.DataFrame, fig_dir: Path) -> None:
    """Closing rank trends over years for top IIT branches (OPEN category, final round)."""
    iit_open = df[
        (df["institute_type"] == "IIT") & (df["category"] == "OPEN")
    ].copy()

    # Use final round of each year
    last_round = iit_open.groupby(["year", "institute", "program"])["round"].max().reset_index()
    last_round = last_round.rename(columns={"round": "max_round"})
    iit_final = iit_open.merge(last_round, on=["year", "institute", "program"])
    iit_final = iit_final[iit_final["round"] == iit_final["max_round"]]

    # Pick top 8 most common IIT branches
    top_branches = (
        iit_final.groupby(["institute", "program"])
        .size()
        .nlargest(8)
        .index
    )

    fig, ax = plt.subplots(figsize=(14, 7))
    for inst, prog in top_branches:
        subset = iit_final[(iit_final["institute"] == inst) & (iit_final["program"] == prog)]
        # Shorten name for legend
        inst_short = inst.replace("Indian Institute of Technology ", "IIT ")
        label = f"{inst_short} – {prog[:30]}"
        ax.plot(subset["year"], subset["closing_rank"].astype(float),
                marker="o", markersize=4, linewidth=1.5, label=label)

    ax.set_title("Closing Rank Trends — Top IIT Branches (OPEN · Final Round)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Closing Rank")
    ax.legend(fontsize=7, loc="upper left", framealpha=0.9)
    ax.invert_yaxis()  # lower rank = harder to get in
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / "06_top_iit_branch_trends.png")
    plt.close()


def _plot_opening_vs_closing(df: pd.DataFrame, fig_dir: Path) -> None:
    """Opening vs Closing rank scatter (sampled for performance)."""
    sample = df.dropna(subset=["opening_rank", "closing_rank"]).sample(
        n=min(20_000, len(df)), random_state=42
    )

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(sample["opening_rank"].astype(float),
               sample["closing_rank"].astype(float),
               alpha=0.15, s=3, c="#3F51B5")

    # Add y=x reference line
    lim = max(sample["closing_rank"].max(), sample["opening_rank"].max())
    ax.plot([0, lim], [0, lim], "r--", linewidth=1, alpha=0.6, label="y = x")

    ax.set_title("Opening Rank vs Closing Rank")
    ax.set_xlabel("Opening Rank")
    ax.set_ylabel("Closing Rank")
    ax.legend()
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(fig_dir / "07_opening_vs_closing.png")
    plt.close()


def _plot_correlation_heatmap(df: pd.DataFrame, fig_dir: Path) -> None:
    """Correlation heatmap for numeric columns."""
    numeric_cols = [
        "year", "round", "opening_rank", "closing_rank",
        "rank_spread", "duration_years",
    ]
    # Add percentile columns if available
    for col in ["opening_percentile", "closing_percentile",
                 "competitiveness_ratio", "applicants"]:
        if col in df.columns:
            numeric_cols.append(col)

    numeric_cols = [c for c in numeric_cols if c in df.columns]
    corr = df[numeric_cols].astype(float).corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, square=True, linewidths=0.5, ax=ax,
                cbar_kws={"shrink": 0.8})
    ax.set_title("Correlation Matrix — Numeric Features")
    plt.tight_layout()
    plt.savefig(fig_dir / "08_correlation_heatmap.png")
    plt.close()


def _plot_round_heatmap(df: pd.DataFrame, fig_dir: Path) -> None:
    """Heatmap: median closing rank by year × round."""
    pivot = df.pivot_table(
        index="year", columns="round",
        values="closing_rank", aggfunc="median",
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(pivot.astype(float), annot=True, fmt=",.0f", cmap="YlOrRd_r",
                linewidths=0.5, ax=ax, cbar_kws={"label": "Median Closing Rank"})
    ax.set_title("Median Closing Rank by Year × Round (AI · Gender-Neutral · Non-PwD)")
    ax.set_xlabel("Round")
    ax.set_ylabel("Year")
    plt.tight_layout()
    plt.savefig(fig_dir / "09_round_year_heatmap.png")
    plt.close()


def _plot_yoy_distribution(df: pd.DataFrame, fig_dir: Path) -> None:
    """Distribution of year-over-year closing rank changes."""
    if "closing_rank_yoy_change" not in df.columns:
        return

    yoy = df["closing_rank_yoy_change"].dropna().astype(float)
    yoy_clipped = yoy.clip(-10000, 10000)  # clip outliers for visibility

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(yoy_clipped, bins=100, alpha=0.7, color="#FF9800", edgecolor="none")
    ax.axvline(0, color="red", linestyle="--", linewidth=1)
    ax.set_title("Year-over-Year Closing Rank Change (Final Round)")
    ax.set_xlabel("Closing Rank Change (positive = cutoff relaxed)")
    ax.set_ylabel("Count")
    mean_change = yoy.mean()
    ax.axvline(mean_change, color="blue", linestyle="--",
               label=f"Mean: {mean_change:+,.0f}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "10_yoy_change_distribution.png")
    plt.close()


# =========================================================================
# 4. Feature Engineering
# =========================================================================

def engineer_features(
    df: pd.DataFrame,
    history_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Create derived features for ML modeling.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to featurize (may be a train or full set).
    history_df : pd.DataFrame, optional
        Historical data to compute lagged stats from.
        If None, uses *strictly prior years* within ``df`` itself.
        This prevents data leakage when used on the training set.

    Returns
    -------
    pd.DataFrame
        DataFrame with added feature columns.
    """
    df = df.copy()
    log.info("Engineering features on %d rows…", len(df))

    # --- 1. Log-transformed ranks ---
    for col in ["opening_rank", "closing_rank"]:
        if col in df.columns:
            df[f"log_{col}"] = np.log1p(df[col].astype(float))

    # --- 2. Frequency encodings ---
    for col in ["institute", "program", "category", "institute_type"]:
        if col in df.columns:
            freq = df[col].value_counts(normalize=True)
            df[f"{col}_freq"] = df[col].map(freq).astype(float)

    # --- 3. Historical statistics (leak-safe) ---
    df = _add_historical_stats(df, history_df)

    # --- 4. Previous-round closing rank (within same year) ---
    df = _add_prev_round_features(df)

    # --- 5. Is-final-round flag ---
    max_round = df.groupby(["year"] + BRANCH_KEY)["round"].transform("max")
    df["is_final_round"] = (df["round"] == max_round).astype(int)

    # --- 6. Years since EWS introduction (2019) ---
    df["years_since_ews"] = (df["year"] - 2019).clip(lower=0)

    log.info("Feature engineering complete — %d columns", len(df.columns))
    return df


def _add_historical_stats(
    df: pd.DataFrame,
    history_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Add historical mean, std, min, max of closing rank per branch group.

    To prevent leakage, only uses data from *strictly prior years*.
    If ``history_df`` is provided, computes from that instead.
    """
    group_key = ["institute", "program", "category", "quota", "gender"]

    if history_df is not None:
        # Use only final rounds from history for stable stats
        last_idx = history_df.groupby(["year"] + group_key)["round"].idxmax()
        hist_final = history_df.loc[last_idx]
        stats = (
            hist_final.groupby(group_key)["closing_rank"]
            .agg(["mean", "std", "min", "max", "count"])
            .rename(columns={
                "mean": "hist_mean_closing_rank",
                "std": "hist_std_closing_rank",
                "min": "hist_min_closing_rank",
                "max": "hist_max_closing_rank",
                "count": "hist_year_count",
            })
        )
        df = df.merge(stats, on=group_key, how="left")
    else:
        # Expanding window: for each row, use only data from prior years
        # This is slower but leak-safe for the full dataset
        records: list[dict[str, Any]] = []
        years = sorted(df["year"].unique())

        # Pre-compute per-year group stats to avoid repeated filtering
        last_idx = df.groupby(["year"] + group_key)["round"].idxmax()
        final_df = df.loc[last_idx].copy()

        cumulative: dict[tuple, list[float]] = {}
        year_stats: dict[int, pd.DataFrame] = {}

        for year in years:
            # Compute stats from all data up to (but not including) this year
            if cumulative:
                rows = []
                for key, ranks in cumulative.items():
                    if len(ranks) >= 1:
                        arr = np.array(ranks, dtype=float)
                        rows.append({
                            **dict(zip(group_key, key)),
                            "hist_mean_closing_rank": float(np.nanmean(arr)),
                            "hist_std_closing_rank": float(np.nanstd(arr, ddof=1)) if len(arr) > 1 else np.nan,
                            "hist_min_closing_rank": float(np.nanmin(arr)),
                            "hist_max_closing_rank": float(np.nanmax(arr)),
                            "hist_year_count": len(arr),
                        })
                if rows:
                    year_stats[year] = pd.DataFrame(rows)

            # Add this year's final-round data to cumulative
            this_year = final_df[final_df["year"] == year]
            for _, row in this_year.iterrows():
                key = tuple(row[c] for c in group_key)
                rank_val = row["closing_rank"]
                if pd.notna(rank_val):
                    cumulative.setdefault(key, []).append(float(rank_val))

        # Merge stats back
        if year_stats:
            all_stats = pd.concat(year_stats.values(), keys=year_stats.keys(),
                                   names=["year"]).reset_index(level=0)
            df = df.merge(all_stats, on=["year"] + group_key, how="left")

    return df


def _add_prev_round_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the previous round's closing rank as a feature."""
    group_key = ["year", "institute", "program", "category", "quota", "gender"]

    df = df.sort_values(group_key + ["round"])
    df["prev_round_closing_rank"] = (
        df.groupby(group_key)["closing_rank"].shift(1)
    )
    return df


# =========================================================================
# 5. Encoding & ML Dataset Creation
# =========================================================================

def encode_categoricals(
    df: pd.DataFrame,
    artifact_dir: Path,
    fit: bool = True,
    encoders: Optional[dict[str, LabelEncoder]] = None,
) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    """
    Label-encode categorical columns and save/load encoder mappings.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    artifact_dir : Path
        Where to save encoder mappings.
    fit : bool
        If True, fit new encoders. If False, use provided encoders.
    encoders : dict, optional
        Pre-fitted encoders (used when fit=False).

    Returns
    -------
    (encoded_df, encoders)
    """
    df = df.copy()
    if encoders is None:
        encoders = {}

    for col in CATEGORICAL_COLS:
        if col not in df.columns:
            continue

        enc_col = f"{col}_encoded"

        if fit:
            le = LabelEncoder()
            # Fit on all known values; handle unseen at transform time
            valid_mask = df[col].notna()
            le.fit(df.loc[valid_mask, col].astype(str))
            encoders[col] = le
        else:
            le = encoders[col]

        # Transform, mapping unseen to -1
        valid_mask = df[col].notna()
        df[enc_col] = -1
        known_classes = set(le.classes_)
        transform_mask = valid_mask & df[col].astype(str).isin(known_classes)
        if transform_mask.any():
            df.loc[transform_mask, enc_col] = le.transform(
                df.loc[transform_mask, col].astype(str)
            )

    # Save encoder mappings
    if fit:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        mappings: dict[str, dict[str, int]] = {}
        for col, le in encoders.items():
            mappings[col] = {
                str(cls): int(idx) for idx, cls in enumerate(le.classes_)
            }
        mapping_path = artifact_dir / "label_encoders.json"
        mapping_path.write_text(json.dumps(mappings, indent=2), encoding="utf-8")
        log.info("Saved label encoders → %s", mapping_path)

    return df, encoders


def create_temporal_split(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split into train/valid/test based on year boundaries.

    Train: ≤ 2023
    Valid: 2024
    Test:  2025–2026
    """
    train = df[df["year"] <= TRAIN_END_YEAR].copy()
    valid = df[df["year"].isin(VALID_YEARS)].copy()
    test = df[df["year"].isin(TEST_YEARS)].copy()

    log.info("Temporal split:")
    log.info("  Train: %d rows (years ≤ %d)", len(train), TRAIN_END_YEAR)
    log.info("  Valid: %d rows (years %s)", len(valid), VALID_YEARS)
    log.info("  Test:  %d rows (years %s)", len(test), TEST_YEARS)

    # Validate no leakage
    train_years = set(train["year"].unique())
    valid_years = set(valid["year"].unique())
    test_years = set(test["year"].unique())

    assert train_years.isdisjoint(valid_years), "Train/valid year leakage!"
    assert train_years.isdisjoint(test_years), "Train/test year leakage!"
    assert valid_years.isdisjoint(test_years), "Valid/test year leakage!"
    log.info("  ✓ No temporal leakage detected")

    return train, valid, test


def build_regression_datasets(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
    feature_dir: Path,
) -> None:
    """
    Build and save regression datasets targeting ``closing_rank``.

    Selects only the feature columns needed for modeling (drops provenance,
    raw strings, etc.)
    """
    target = "closing_rank"

    # Feature columns to keep
    feature_cols = [
        # Core identifiers (needed for grouping but not for model input)
        "year", "round",
        # Encoded categoricals
        "institute_type_encoded", "category_encoded", "quota_encoded",
        "gender_encoded", "degree_type_encoded",
        # Numeric features
        "opening_rank", "log_opening_rank",
        "duration_years",
        "institute_freq", "program_freq", "category_freq", "institute_type_freq",
        "hist_mean_closing_rank", "hist_std_closing_rank",
        "hist_min_closing_rank", "hist_max_closing_rank", "hist_year_count",
        "prev_round_closing_rank",
        "is_final_round", "years_since_ews",
        "is_pwd",
    ]

    # Add percentile features if available
    for col in ["opening_percentile", "applicants"]:
        if col in train.columns:
            feature_cols.append(col)

    # Only keep columns that exist
    feature_cols = [c for c in feature_cols if c in train.columns]

    # Add target
    all_cols = feature_cols + [target]
    all_cols = list(dict.fromkeys(all_cols))  # dedup

    for name, split_df in [("train", train), ("valid", valid), ("test", test)]:
        out = split_df[[c for c in all_cols if c in split_df.columns]].copy()
        path = feature_dir / f"regression_{name}.csv"
        out.to_csv(path, index=False)
        log.info("  regression_%s.csv: %d rows × %d cols", name, len(out), len(out.columns))

    log.info("Regression datasets saved")


def build_classification_datasets(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
    feature_dir: Path,
) -> None:
    """
    Build classification datasets with binary label ``cutoff_difficulty``.

    Since we don't have individual student ranks, we create a binary label:
      - 1 = "competitive" — closing rank is below (better than) the historical
        median for that branch group
      - 0 = "relaxed" — closing rank is above (easier than) the historical median

    This enables a classifier to predict whether a branch will be harder to
    get into than average — useful for counselling recommendations.

    A future improvement would use actual student ranks to create
    ``rank_beats_cutoff`` labels.
    """
    target = "cutoff_difficulty"

    for name, split_df in [("train", train), ("valid", valid), ("test", test)]:
        df = split_df.copy()

        # Binary label: is this cutoff tighter than the historical average?
        # NaN handling: if hist_mean is missing (new branch), fall back to global median
        closing = df["closing_rank"].astype(float)
        if "hist_mean_closing_rank" in df.columns:
            hist_mean = df["hist_mean_closing_rank"].copy()
            # Fall back to global median for rows without history
            global_median = float(train["closing_rank"].astype(float).median())
            hist_mean = hist_mean.fillna(global_median)
            df[target] = (closing <= hist_mean).fillna(False).astype(int)
        else:
            median_cutoff = float(train["closing_rank"].astype(float).median())
            df[target] = (closing <= median_cutoff).fillna(False).astype(int)

        # Feature columns (same as regression but swap target)
        feature_cols = [
            "year", "round",
            "institute_type_encoded", "category_encoded", "quota_encoded",
            "gender_encoded", "degree_type_encoded",
            "opening_rank", "log_opening_rank",
            "duration_years",
            "institute_freq", "program_freq", "category_freq", "institute_type_freq",
            "hist_mean_closing_rank", "hist_std_closing_rank",
            "hist_min_closing_rank", "hist_max_closing_rank", "hist_year_count",
            "prev_round_closing_rank",
            "is_final_round", "years_since_ews",
            "is_pwd",
        ]
        feature_cols = [c for c in feature_cols if c in df.columns]
        all_cols = feature_cols + [target]
        all_cols = list(dict.fromkeys(all_cols))

        out = df[[c for c in all_cols if c in df.columns]].copy()
        path = feature_dir / f"classification_{name}.csv"
        out.to_csv(path, index=False)
        log.info("  classification_%s.csv: %d rows × %d cols  (label=1: %.1f%%)",
                  name, len(out), len(out.columns),
                  out[target].mean() * 100)

    log.info("Classification datasets saved")
    log.info("  NOTE: Classification label 'cutoff_difficulty' = 1 means closing_rank ≤ historical mean")
    log.info("  NOTE: For actual admission prediction, student rank data is needed (Day 3+)")


# =========================================================================
# 6. Validation
# =========================================================================

def validate_datasets(feature_dir: Path) -> None:
    """Run validation checks on all generated datasets."""
    issues: list[str] = []

    for pattern in ["regression_*.csv", "classification_*.csv"]:
        for path in sorted(feature_dir.glob(pattern)):
            df = pd.read_csv(path)
            name = path.stem

            # NaN check
            nan_cols = df.columns[df.isnull().any()].tolist()
            nan_total = df.isnull().sum().sum()
            if nan_total > 0:
                nan_pcts = {c: f"{df[c].isnull().mean()*100:.1f}%" for c in nan_cols}
                log.info("  %s: %d NaN cells in %d columns: %s",
                          name, nan_total, len(nan_cols), nan_pcts)
            else:
                log.info("  %s: ✓ no NaN values", name)

            # Shape
            log.info("  %s: shape = %s", name, df.shape)

    # Check train/valid/test year integrity
    for task in ["regression", "classification"]:
        train_path = feature_dir / f"{task}_train.csv"
        valid_path = feature_dir / f"{task}_valid.csv"
        test_path = feature_dir / f"{task}_test.csv"

        if all(p.exists() for p in [train_path, valid_path, test_path]):
            tr = pd.read_csv(train_path)
            va = pd.read_csv(valid_path)
            te = pd.read_csv(test_path)

            tr_years = set(tr["year"].unique())
            va_years = set(va["year"].unique())
            te_years = set(te["year"].unique())

            if not tr_years.isdisjoint(va_years):
                issues.append(f"{task}: train/valid year overlap: {tr_years & va_years}")
            if not tr_years.isdisjoint(te_years):
                issues.append(f"{task}: train/test year overlap: {tr_years & te_years}")

            log.info("  %s: train years=%s, valid years=%s, test years=%s",
                      task, sorted(tr_years), sorted(va_years), sorted(te_years))

    if issues:
        for issue in issues:
            log.error("VALIDATION: %s", issue)
    else:
        log.info("  ✓ All validation checks passed")


# =========================================================================
# 7. Master Pipeline
# =========================================================================

def run_day2_pipeline(data_dir: Path) -> None:
    """
    Run the complete Day 2 pipeline:
    load → EDA → charts → feature engineering → splits → datasets → validate.
    """
    # --- Directory setup ---
    feature_dir = data_dir / "features"
    reports_dir = Path("reports")
    figures_dir = Path("figures")
    artifact_dir = Path("model_artifacts")

    for d in [feature_dir, reports_dir, figures_dir, artifact_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # --- Step 1: Load data ---
    log.info("=" * 60)
    log.info("STEP 1: Loading master dataset")
    log.info("=" * 60)
    df = load_master(data_dir)

    # --- Step 2: EDA ---
    log.info("=" * 60)
    log.info("STEP 2: Exploratory Data Analysis")
    log.info("=" * 60)
    summary = compute_eda_summary(df)
    write_eda_report(summary, reports_dir / "eda_summary.md")

    # Save raw summary as JSON too (machine-readable)
    (reports_dir / "eda_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    log.info("Saved eda_summary.json")

    # --- Step 3: Charts ---
    log.info("=" * 60)
    log.info("STEP 3: Generating charts")
    log.info("=" * 60)
    generate_charts(df, figures_dir)

    # --- Step 4: Temporal split (before feature engineering to prevent leakage) ---
    log.info("=" * 60)
    log.info("STEP 4: Temporal train/valid/test split")
    log.info("=" * 60)
    train_raw, valid_raw, test_raw = create_temporal_split(df)

    # --- Step 5: Feature engineering (leak-safe) ---
    log.info("=" * 60)
    log.info("STEP 5: Feature engineering")
    log.info("=" * 60)

    # Train: compute historical stats from expanding window within train
    train_fe = engineer_features(train_raw, history_df=None)

    # Valid & Test: compute historical stats from ALL training data
    valid_fe = engineer_features(valid_raw, history_df=train_raw)
    test_fe = engineer_features(test_raw, history_df=pd.concat([train_raw, valid_raw]))

    # --- Step 6: Encode categoricals ---
    log.info("=" * 60)
    log.info("STEP 6: Encoding categoricals")
    log.info("=" * 60)

    # Fit encoders on training data
    train_enc, encoders = encode_categoricals(train_fe, artifact_dir, fit=True)
    valid_enc, _ = encode_categoricals(valid_fe, artifact_dir, fit=False, encoders=encoders)
    test_enc, _ = encode_categoricals(test_fe, artifact_dir, fit=False, encoders=encoders)

    # --- Step 7: Save full feature-engineered dataset ---
    log.info("=" * 60)
    log.info("STEP 7: Saving feature-engineered dataset")
    log.info("=" * 60)
    full_fe = pd.concat([train_enc, valid_enc, test_enc], ignore_index=True)
    full_fe.to_csv(feature_dir / "feature_engineered.csv", index=False)
    log.info("Wrote feature_engineered.csv  (%d rows × %d cols)",
              len(full_fe), len(full_fe.columns))

    # --- Step 8: Build regression datasets ---
    log.info("=" * 60)
    log.info("STEP 8: Building regression datasets")
    log.info("=" * 60)
    build_regression_datasets(train_enc, valid_enc, test_enc, feature_dir)

    # --- Step 9: Build classification datasets ---
    log.info("=" * 60)
    log.info("STEP 9: Building classification datasets")
    log.info("=" * 60)
    build_classification_datasets(train_enc, valid_enc, test_enc, feature_dir)

    # --- Step 10: Validation ---
    log.info("=" * 60)
    log.info("STEP 10: Validation")
    log.info("=" * 60)
    validate_datasets(feature_dir)

    # --- Step 11: Print example rows ---
    log.info("=" * 60)
    log.info("STEP 11: Example rows")
    log.info("=" * 60)
    reg_train = pd.read_csv(feature_dir / "regression_train.csv")
    log.info("Regression train sample (first 3 rows):")
    print(reg_train.head(3).to_string())
    print()

    cls_train = pd.read_csv(feature_dir / "classification_train.csv")
    log.info("Classification train sample (first 3 rows):")
    print(cls_train.head(3).to_string())
    print()

    # --- Final summary ---
    log.info("=" * 60)
    log.info("DAY 2 PIPELINE COMPLETE")
    log.info("=" * 60)

    log.info("  Output directories:")
    for d in [feature_dir, reports_dir, figures_dir, artifact_dir]:
        file_count = sum(1 for _ in d.rglob("*") if _.is_file())
        total_size = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
        log.info("    %-20s  %2d files  %7.2f MB", d, file_count, total_size / 1e6)


# =========================================================================
# CLI
# =========================================================================

def main() -> None:
    """Command-line entry point for the Day 2 pipeline."""
    parser = argparse.ArgumentParser(
        description="JoSAA Day 2 — EDA & Feature Engineering Pipeline",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).parent / "data",
        help="Directory containing the processed data (default: ./data)",
    )
    args = parser.parse_args()

    run_day2_pipeline(args.data_dir)


if __name__ == "__main__":
    main()
