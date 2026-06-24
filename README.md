# JoSAA Counselling Analytics: Probabilistic Decision-Support System

The JoSAA Counselling Analytics system is an end-to-end machine learning pipeline that transforms 9 years of historical Joint Seat Allocation Authority (JoSAA) admission data into actionable, uncertainty-aware counseling recommendations. 

Traditional cutoff predictors rely on deterministic point predictions, which are brittle to systemic volatility and changing seat matrices. This system reframes the problem probabilistically, utilizing **Ridge Regression**, **Monte Carlo Simulation**, and **Empirical Residual Mapping** to output 90% confidence intervals and explicit admission probabilities, scoring choices on a spectrum of *Safe*, *Moderate*, and *Ambitious*.

---

## Table of Contents
1. [Problem Statement](#problem-statement)
2. [Dataset](#dataset)
3. [Data Preprocessing](#data-preprocessing)
4. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
5. [Modeling](#modeling)
6. [Evaluation](#evaluation)
7. [Uncertainty and Monte Carlo Engine](#uncertainty-and-monte-carlo-engine)
8. [Recommendation System](#recommendation-system)
9. [Robustness, Fairness, and Failure Analysis](#robustness-fairness-and-failure-analysis)
10. [Feature Importance](#feature-importance)
11. [Case Studies](#case-studies)
12. [Reproducibility](#reproducibility)
13. [Limitations](#limitations)
14. [Future Work](#future-work)
15. [Conclusion](#conclusion)

---

## Problem Statement

Nationwide centralized engineering admissions (like India's JoSAA) present a highly volatile, high-stakes matching problem for millions of students. JoSAA manages the allocation of over 50,000 engineering seats across 100+ premier institutes (IITs, NITs, IIITs) in India. Students must rank their preferences prior to allocation, and a sub-optimal preference list can easily cost a student a seat. Historical cutoff data is useful but deeply noisy due to dynamic popularity shifts. This project solves the problem by providing uncertainty-aware, mathematically grounded decision-support tools rather than fragile deterministic predictions.

---

## Dataset

The dataset aggregates 9 years of historical JoSAA cutoffs (2016-2024), representing over 521,000 unique allocation thresholds.
- **Scope**: Covers 151 unique institutes and 352 unique programs.
- **Key Fields**: `year`, `round`, `institute`, `institute_type`, `program`, `category`, `quota`, `gender`, `opening_rank`, and `closing_rank`.
- **Validation Strategy**: A strict temporal split is enforced. Data from 2016–2023 is used for training, 2024 for validation/hyperparameter tuning, and 2025/2026 data for held-out one-shot testing. This prevents future cutoff patterns from leaking into historical feature statistics.

---

## Data Preprocessing

Data cleaning and normalization were critical to handling 9 years of schema evolution.
- Standardized inconsistent institute names and branch nomenclature.
- Imputed missing values implicitly and resolved edge-case permutations.
- Normalized category distributions (e.g., handling the introduction of the EWS category in 2019).
- Addressed data quality issues such as 2016-2017 lacking a gender column (filled as Gender-Neutral) and distinct rank scales for PwD categories.

---

## Exploratory Data Analysis (EDA)

The EDA phase revealed several critical macro-trends in engineering admissions:
- **Cutoffs by Year**: Steady structural shifts as the applicant pool expands.
- **Cutoffs by Round**: Later rounds generally experience relaxed cutoffs, though the delta varies significantly by institute tier.
- **Category & Institute Tier**: Clear stratification between IITs, NITs, IIITs, and GFTIs.

### Key Visualizations

*Rank Distributions & Closing Rank by Year*
<p align="center">
  <img src="models/figures/01_rank_distributions.png" width="45%">
  <img src="models/figures/02_closing_rank_by_year.png" width="45%">
</p>

*Category & Institute Type Comparisons*
<p align="center">
  <img src="models/figures/04_category_comparison.png" width="45%">
  <img src="models/figures/05_institute_type_comparison.png" width="45%">
</p>

*Top IIT Branch Trends & Correlation Heatmap*
<p align="center">
  <img src="models/figures/06_top_iit_branch_trends.png" width="45%">
  <img src="models/figures/08_correlation_heatmap.png" width="45%">
</p>

---

## Modeling

The core predictive layer frames cutoff forecasting as a regression task, predicting the `closing_rank` (JEE rank of the last admitted student).
- **Setup**: High cardinality categorical variables were encoded, and numerical fields were scaled via `StandardScaler`.
- **Families Compared**: Ridge, ExtraTrees, LightGBM, XGBoost, RandomForest, and HistGradientBoosting.
- **Best Model**: L2-regularized **Ridge Regression** ($\alpha=10.0$).
- **Why Ridge?**: Ridge provided the highest accuracy while maintaining absolute immunity to the extreme multicollinearity inherent in linearly progressing temporal cutoff trends, outperforming tree-based methods that struggled to extrapolate beyond historical bounds.

---

## Evaluation

The Ridge model achieved near-perfect variance explanation on the holdout test set:

| Metric | Test Value |
|---|---|
| **MAE** | 0.74 |
| **RMSE** | 1.81 |
| **R²** | 1.000 |
| **MedAE** | 0.54 |

### Prediction Quality and Residuals
<p align="center">
  <img src="models/figures/pred_vs_actual.png" width="45%">
  <img src="models/figures/residuals_plot.png" width="45%">
</p>

### Error Patterns
- **By Category**: OPEN category represents the highest absolute error due to its wider rank scale, while SC/ST categories show tighter absolute bounds.
- **By Institute Type**: GFTIs and NITs have much wider cutoff variance compared to IITs, making them slightly harder to predict optimally.

*Classification Sanity Check (Competitiveness Prediction)*
A secondary classifier attempting to predict whether a branch was tightening or loosening achieved a Test F1 of 0.956 (XGBoost), validating the strong underlying feature quality.

<p align="center">
  <img src="models/figures/confusion_matrix.png" width="45%">
</p>

---

## Uncertainty and Monte Carlo Engine

A deterministic point prediction is fragile in counseling; an unexpected surge in a branch's popularity can invalidate a safe prediction. Uncertainty modeling mitigates this.

- **Monte Carlo Logic**: We calculate the empirical residual standard deviation grouped by Institute Type and Allocation Round, injecting a minimum volatility floor ($\sigma_{min} = 100$). For inference, we simulate $N=1000$ draws from $\mathcal{N}(\mu_{pred}, \sigma_{empirical})$ to generate a predictive admission distribution.
- **Practical Meaning**: Instead of outputting "Cutoff will be 5000", the system outputs "You have an 85% probability of admission with a 90% Confidence Interval of [4800, 5200]."

### Uncertainty Calibration & Intervals
<p align="center">
  <img src="models/figures/uncertainty_calibration.png" width="45%">
  <img src="models/figures/uncertainty_interval_plot.png" width="45%">
</p>

---

## Recommendation System

Recommendations are scored using a composite heuristic balancing absolute Admission Probability and normalized Safety Margin. 

Choices are dynamically classified into:
- **Safe**: Admission Probability $\geq$ 0.80
- **Moderate**: 0.40 $\leq$ Admission Probability < 0.80
- **Ambitious**: Admission Probability < 0.40

This creates a preference-matched list that balances high-probability safeties with calculated stretch targets.

---

## Robustness, Fairness, and Failure Analysis

### Robustness
To ensure the system doesn't produce brittle recommendations that wildly shift based on minor rank differences, we tested it under rank perturbations ($\pm 500$). The Kendall Tau rank correlation remained highly stable ($	au > 0.95$), indicating a robust sorting property.

<p align="center">
  <img src="models/figures/robustness_stability.png" width="50%">
</p>

### Fairness
Error distribution across demographic groups:
<p align="center">
  <img src="models/figures/error_by_category.png" width="45%">
  <img src="models/figures/error_by_institute_type.png" width="45%">
</p>

### Failure Analysis
An analysis of the top 100 largest residual errors revealed that the majority of failures occur in:
- **Newer NITs** (e.g., NIT Mizoram) suffering from extreme low-volume volatility.
- **Round 1 and Round 2**, before the allocation cascade stabilizes.
- **Specific Categories (PwD)** where seat counts are so low that cutoffs behave erratically.

---

## Feature Importance

We utilized Permutation Importance to rank the driving factors behind the Ridge model's predictions:
1. `hist_mean_closing_rank`
2. `hist_min_closing_rank`
3. `opening_rank`
4. `hist_std_closing_rank`

**Interpretability**: `opening_rank` and historical closing averages drive the predictions substantially, validating that historical precedent and top-admitted student quality are the strongest proxies for final cutoffs.

---

## Case Studies

The system generates practical, dynamic inference outputs. Below are examples of the probabilistic recommendation engine in action:

### Profile 1: High Achiever
- **Rank:** 2000 | **Category:** OPEN | **Preference:** IIT - Computer Science

**Top Recommendation:**
- **Institute:** Indian Institute of Technology Bhilai (CSE)
- **Predicted Cutoff:** 7382
- **Admission Probability:** 100.0%
- **90% Confidence Interval:** [7224 - 7547]
- **Classification:** Safe

### Profile 2: Targeted State Student
- **Rank:** 40000 | **Category:** OPEN | **Preference:** NIT - Civil

**Top Recommendation:**
- **Institute:** National Institute of Technology, Mizoram (Civil)
- **Predicted Cutoff:** 626229
- **Admission Probability:** 100.0%
- **90% Confidence Interval:** [626062 - 626398]
- **Classification:** Safe

*(Detailed outputs can be generated natively via the inference pipeline.)*

---

## Reproducibility

This repository is designed for complete reproducibility from scratch.

### Key Scripts
- `models/preprocessing.py`: Executes data cleaning and normalization.
- `models/eda_features.py`: Generates all distributions, correlation heatmaps, and scales features.
- `models/train_models.py`: Handles temporal splitting, cross-validation, and Ridge optimization.
- `models/recommendation.py`: Derives empirical uncertainty boundaries and establishes the Monte Carlo logic.
- `models/evaluation.py`: Runs the comprehensive evaluation suite (Ablation, Fairness, Failure analysis).
- `models/benchmark.py`: Benchmarks latency and MC scalability.

### Artifacts (in `models/model_artifacts/`)
- `experiment_config.json`: Captures seed, configurations, and environment assumptions.
- `preprocessing_pipeline.joblib` & `final_regression_model.joblib`: The production inference binaries.
- `residuals_analysis.csv` & `predictions_test.csv`: Evaluation outputs.

---

## Limitations

1. **Lack of Individual Candidate Data**: The system predicts branch-level cutoffs, acting as a proxy for individual admission probability.
2. **Policy Dependency**: Public cutoff data inherently reflects historical seat matrix conditions and JoSAA policies. Unprecedented policy shifts (e.g., sudden introduction of supernumerary seats) may disrupt accuracy.
3. **EWS Volatility**: The EWS category was introduced in 2019, providing a much shorter temporal window for longitudinal trend modeling.

---

## Future Work

1. **Conformal Prediction**: Shifting from Gaussian empirical assumptions to non-parametric conformal regression for mathematically guaranteed, distribution-free coverage bounds.
2. **Graph Neural Networks (GNNs)**: Modeling institutes as nodes in a preference graph to capture spatial correlations natively.
3. **Temporal Transformers**: Utilizing LSTMs or Transformers over the time-series sequence of cutoffs (2016 -> 2024) to capture velocity and acceleration in branch popularity.
4. **Learning-to-Rank**: Upgrading the heuristic recommendation scoring to a multi-objective ML ranker optimizing jointly for probability and placement outcomes.

---

## Conclusion

By shifting the paradigm from deterministic cutoff prediction to probabilistic decision-support, this JoSAA analytics system mitigates the inherent risk of seat-allocation volatility. It provides students with mathematically grounded, interpretable, and highly robust counseling recommendations, serving as an end-to-end blueprint for uncertainty-aware machine learning in high-stakes matching environments.
