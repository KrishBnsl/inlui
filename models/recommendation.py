"""
JoSAA Day 4 — Monte Carlo Uncertainty & Recommendation Engine
=============================================================

This script converts the Day 3 deterministic closing rank predictor into a 
decision-support layer by:
1. Estimating empirical uncertainty (residuals) by institute type and round.
2. Running a Monte Carlo simulation to calculate admission probability.
3. Ranking choices for mock student profiles (Safe, Moderate, Ambitious).
4. Exporting readable case studies and metrics.

Usage:
    python recommendation.py
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def load_artifacts(models_dir: Path, data_dir: Path) -> Tuple[Any, Any, pd.DataFrame, pd.DataFrame, dict]:
    """Loads all necessary artifacts from Day 3 and datasets."""
    log.info("Loading model artifacts...")
    model = joblib.load(models_dir / "model_artifacts" / "final_regression_model.joblib")
    prep_pipeline = joblib.load(models_dir / "model_artifacts" / "preprocessing_pipeline.joblib")
    
    with open(models_dir / "model_artifacts" / "label_encoders.json", "r") as f:
        encoders = json.load(f)
        
    log.info("Loading datasets...")
    # The raw test features
    test_features = pd.read_csv(data_dir / "features" / "regression_test.csv")
    # The actual predictions and residuals for the test set
    test_preds = pd.read_csv(models_dir / "model_artifacts" / "residuals_analysis.csv")
    
    # Load the cleaned string data to map back human-readable names
    cleaned_data = pd.read_csv(data_dir / "processed" / "josaa_cleaned.csv")
    
    # In test_preds, the category, institute_type are encoded. 
    # We will use the test_features to align indices since they map 1:1 with predictions_test/residuals.
    test_preds["opening_rank"] = test_features["opening_rank"].values
    
    # Let's decode the categories to join with cleaned data
    cat_inv = {v: k for k, v in encoders["category"].items()}
    inst_type_inv = {v: k for k, v in encoders["institute_type"].items()}
    
    test_preds["category_str"] = test_preds["category_encoded"].map(cat_inv)
    test_preds["institute_type_str"] = test_preds["institute_type_encoded"].map(inst_type_inv)
    
    # Join with cleaned data to get institute and program names
    merged_test = pd.merge(
        test_preds,
        cleaned_data[["year", "round", "institute_type", "category", "opening_rank", "closing_rank", "institute", "program", "quota", "gender"]],
        left_on=["year", "round", "institute_type_str", "category_str", "opening_rank", "actual_closing_rank"],
        right_on=["year", "round", "institute_type", "category", "opening_rank", "closing_rank"],
        how="inner"
    )
    
    # Deduplicate if exact matches happened
    merged_test = merged_test.drop_duplicates(subset=["year", "round", "institute", "program", "category_str", "quota", "gender"])
    
    log.info(f"Successfully mapped {len(merged_test)} rows with human-readable names.")
    
    return model, prep_pipeline, merged_test, test_features, encoders


def estimate_residual_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes empirical standard deviation of residuals grouped by institute_type and round.
    Injects a minimum standard deviation to represent real-world volatility.
    """
    log.info("Estimating uncertainty bounds from empirical residuals...")
    grouped = df.groupby(["institute_type_str", "round"])["residual"].std().reset_index()
    grouped.rename(columns={"residual": "std_dev"}, inplace=True)
    
    # Because the Ridge model is extremely accurate (high R2), the empirical residuals
    # are very tight. We inject a minimum floor to simulate real macroeconomic 
    # volatility that single test-years may underrepresent.
    grouped["std_dev"] = grouped["std_dev"].clip(lower=100.0)
    
    return grouped


def run_monte_carlo(row: pd.Series, std_dev: float, student_rank: int, n_sims: int = 1000) -> dict:
    """
    Runs a Monte Carlo simulation for a single program choice.
    Simulates N potential cutoffs from N(pred, std_dev).
    Returns probability of admission and confidence intervals.
    """
    pred = row["predicted_closing_rank"]
    
    # Sample from Gaussian distribution
    simulated_cutoffs = np.random.normal(loc=pred, scale=std_dev, size=n_sims)
    
    # Probability of admission = % of simulations where student_rank <= simulated_cutoff
    # (Assuming lower rank means better performance)
    prob = np.mean(student_rank <= simulated_cutoffs)
    
    # Confidence intervals
    p05 = np.percentile(simulated_cutoffs, 5)
    p50 = np.percentile(simulated_cutoffs, 50)
    p95 = np.percentile(simulated_cutoffs, 95)
    
    return {
        "admission_probability": prob,
        "ci_lower_5": max(1, p05), # Rank can't be negative
        "expected_cutoff_50": p50,
        "ci_upper_95": p95
    }


def build_recommendation_scores(choices: pd.DataFrame, student_rank: int) -> pd.DataFrame:
    """
    Ranks choices using a heuristic combination of admission probability and margin.
    """
    # Margin score: How far is the expected cutoff from the student's rank?
    # Positive margin is safe.
    choices["margin"] = choices["predicted_closing_rank"] - student_rank
    choices["margin_score"] = choices["margin"] / choices["predicted_closing_rank"]
    
    # Composite score
    # We heavily weight probability, but break ties with margin
    choices["recommendation_score"] = (choices["admission_probability"] * 100) + (choices["margin_score"] * 20)
    
    # Categorize
    conditions = [
        choices["admission_probability"] >= 0.80,
        (choices["admission_probability"] >= 0.40) & (choices["admission_probability"] < 0.80),
        choices["admission_probability"] < 0.40
    ]
    labels = ["Safe", "Moderate", "Ambitious"]
    choices["category"] = np.select(conditions, labels, default="Ambitious")
    
    return choices.sort_values(by="recommendation_score", ascending=False)


def generate_case_studies(merged_test: pd.DataFrame, std_map: pd.DataFrame, reports_dir: Path):
    """
    Simulates the engine for 3 mock student profiles and generates a markdown report.
    """
    log.info("Generating Case Studies...")
    
    profiles = [
        {
            "name": "High Achiever",
            "rank": 2000,
            "category": "OPEN",
            "gender": "Gender-Neutral",
            "pref_inst": "IIT",
            "pref_branch": "Computer Science"
        },
        {
            "name": "Mid-Tier Aspirant",
            "rank": 15000,
            "category": "OBC-NCL",
            "gender": "Gender-Neutral",
            "pref_inst": "NIT",
            "pref_branch": "Electronics"
        },
        {
            "name": "Targeted State Student",
            "rank": 40000,
            "category": "OPEN",
            "gender": "Female-only",
            "pref_inst": "NIT",
            "pref_branch": "Civil"
        }
    ]
    
    md_lines = ["# JoSAA Day 4: Recommendation Engine Case Studies\n"]
    md_lines.append("This report demonstrates the uncertainty-aware recommendation engine output for three distinct student profiles. The engine utilizes Monte Carlo simulation (N=1000) over the Day 3 Ridge regression model to provide probabilistic admission estimates rather than fragile point predictions.\n")
    
    all_recommendations = []
    
    for p in profiles:
        log.info(f"Processing profile: {p['name']}")
        # Filter the universe of choices matching demographic
        mask = (
            (merged_test["category_str"] == p["category"]) & 
            (merged_test["gender"] == p["gender"]) &
            (merged_test["institute_type_str"] == p["pref_inst"]) & 
            (merged_test["program"].str.contains(p["pref_branch"], case=False, na=False)) &
            (merged_test["round"] == 6) # Look at final round typically
        )
        subset = merged_test[mask].copy()
        
        if len(subset) == 0:
            log.warning(f"No choices found for {p['name']}.")
            continue
            
        # Add standard deviation mapped by inst_type and round
        subset = subset.merge(std_map, on=["institute_type_str", "round"], how="left")
        subset["std_dev"] = subset["std_dev"].fillna(150.0) # Fallback
        
        # Run Monte Carlo
        mc_results = []
        for _, row in subset.iterrows():
            res = run_monte_carlo(row, row["std_dev"], p["rank"])
            mc_results.append(res)
            
        mc_df = pd.DataFrame(mc_results)
        for col in mc_df.columns:
            subset[col] = mc_df[col].values
            
        # Rank
        ranked = build_recommendation_scores(subset, p["rank"])
        all_recommendations.append(ranked)
        
        # Write to Markdown
        md_lines.append(f"## Profile: {p['name']}")
        md_lines.append(f"- **Student Rank:** {p['rank']}")
        md_lines.append(f"- **Category:** {p['category']}")
        md_lines.append(f"- **Preferences:** {p['pref_inst']} - {p['pref_branch']}")
        md_lines.append("\n### Top 10 Recommended Options")
        
        top10 = ranked.head(10)[["institute", "program", "predicted_closing_rank", "admission_probability", "ci_lower_5", "ci_upper_95", "category"]]
        
        md_table = "| Institute | Program | Predicted Cutoff | Admission Prob | 90% CI Bounds | Classification |\n"
        md_table += "|---|---|---|---|---|---|\n"
        
        for _, row in top10.iterrows():
            prob_pct = f"{row['admission_probability']*100:.1f}%"
            bounds = f"[{int(row['ci_lower_5'])} - {int(row['ci_upper_95'])}]"
            md_table += f"| {row['institute']} | {row['program']} | {int(row['predicted_closing_rank'])} | {prob_pct} | {bounds} | {row['category']} |\n"
            
        md_lines.append(md_table)
        md_lines.append("\n")
        
    report_path = reports_dir / "case_study_report.md"
    with open(report_path, "w") as f:
        f.write("\n".join(md_lines))
        
    log.info(f"Saved {report_path}")
    
    # Save the full results CSV
    if all_recommendations:
        full_df = pd.concat(all_recommendations)
        csv_path = reports_dir / "recommendation_results.csv"
        full_df.to_csv(csv_path, index=False)
        log.info(f"Saved {csv_path}")
        
    return all_recommendations


def plot_uncertainty_intervals(recommendations: List[pd.DataFrame], figures_dir: Path):
    """
    Plots the top choices with their 90% uncertainty intervals against the student's rank.
    """
    log.info("Generating uncertainty plots...")
    if not recommendations:
        return
        
    # We will just plot the first profile's top 10 as an example
    df = recommendations[0].head(10).copy()
    
    df["display_name"] = df["institute"].str.replace("Indian Institute of Technology", "IIT")
    
    plt.figure(figsize=(10, 6))
    
    # Plot expected cutoff
    plt.errorbar(
        x=df["expected_cutoff_50"],
        y=df["display_name"],
        xerr=[df["expected_cutoff_50"] - df["ci_lower_5"], df["ci_upper_95"] - df["expected_cutoff_50"]],
        fmt='o', color='blue', ecolor='lightblue', elinewidth=5, capsize=0, label="Expected Cutoff & 90% CI"
    )
    
    # Plot example student ranks to show Safe/Moderate/Ambitious contexts
    plt.axvline(x=2000, color='green', linestyle='--', alpha=0.7, label="Rank 2000 (Safe)")
    plt.axvline(x=4000, color='orange', linestyle='--', alpha=0.7, label="Rank 4000 (Moderate)")
    plt.axvline(x=6000, color='red', linestyle='--', alpha=0.7, label="Rank 6000 (Ambitious)")
    
    plt.xlabel("Closing Rank (Lower is Better/Harder)")
    plt.title("Monte Carlo Prediction Intervals vs. Student Rank")
    plt.gca().invert_yaxis()  # Best choice at top
    plt.legend()
    plt.tight_layout()
    
    plot_path = figures_dir / "uncertainty_interval_plot.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    log.info(f"Saved {plot_path}")


def main():
    models_dir = Path(".")
    data_dir = models_dir / "data"
    reports_dir = models_dir / "reports"
    figures_dir = models_dir / "figures"
    
    reports_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)
    
    # 1. Load artifacts
    model, prep_pipeline, merged_test, test_features, encoders = load_artifacts(models_dir, data_dir)
    
    # 2. Estimate residual distribution
    std_map = estimate_residual_distribution(merged_test)
    
    # 3 & 4 & 5. Generate Case Studies & Recommendations (which runs Monte Carlo)
    recs = generate_case_studies(merged_test, std_map, reports_dir)
    
    # 6. Plot
    plot_uncertainty_intervals(recs, figures_dir)
    
    log.info("Day 4 Pipeline Complete.")


if __name__ == "__main__":
    main()
