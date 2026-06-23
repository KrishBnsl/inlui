# JoSAA Recommendation System: Research Evaluation Report

## 1. Methodology
This evaluation rigorously tests the uncertainty-aware recommendation system. We measure recommendation quality, robustness to rank perturbations, and demographic fairness, treating the system as a decision-support tool rather than a pure regression model.

## 2. Experimental Setup
- **Model**: Ridge Regression with Standard Scaling.
- **Test Set**: JoSAA 2024 data.
- **Uncertainty**: Empirical residual standard deviation with Monte Carlo sampling.

## 3. Models Evaluated
## Ablation Study Results

### Variants
- **A**: Predicted Cutoff Only
- **B**: Expected Cutoff (MC 50th percentile)
- **C**: Admission Probability
- **D**: Admission Probability + Margin (Final Model)

### Metrics
- Top-50 Overlap (A vs D): 1.000
- Kendall Tau (A vs D): 1.000 (p=0.00e+00)
- Spearman Correlation (A vs D): 1.000 (p=0.00e+00)

### Top-K Quality (A vs D)
- Avg Safety Margin (A): 46043.9
- Avg Safety Margin (D): 46043.9
- Avg Admission Probability (A): 0.799
- Avg Admission Probability (D): 0.799
- Recommendation Diversity (Institute Types) (A): 4
- Recommendation Diversity (Institute Types) (D): 4

### Statistical Significance Test (Wilcoxon on Admission Probability)
- p-value: nan

## 4. Robustness Results
## Robustness Analysis
We evaluated stability by perturbing the student rank (10000) by [-500, -250, +250, +500].
- The recommendation ranking correlations (Kendall Tau and Spearman) remain high across small perturbations.
See `figures/robustness_stability.png` for plot.

## 5. Fairness Results
See `fairness_report.md` for full breakdown.

## 6. Uncertainty Analysis
## Uncertainty Analysis
- Correlation between prediction interval width and actual error: nan (p=nan).
- Positive correlation indicates that wider uncertainty intervals correctly capture higher actual errors.
See `figures/uncertainty_calibration.png` for plot.

## 7. Failure Analysis
See `failure_case_report.md` for specific top failure cases.

## 8. Feature Importance
See `feature_analysis.md` for detailed rankings.

## 9. Limitations
- Mock student rank profiles may not fully capture the complexity of real student preferences.
- Uncertainty bounds are derived from empirical residuals, which may underestimate actual variance in newly introduced branches.

## 10. Future Work
- Incorporating time-series specific models (e.g., ARIMA or LSTMs) to capture trend velocity.
- Applying advanced Conformal Prediction to produce theoretically guaranteed prediction intervals.