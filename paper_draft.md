# Probabilistic Decision-Support for Nationwide Engineering Admissions using Monte Carlo Simulation

**Abstract**: Nationwide centralized engineering admissions (like India's JoSAA) present a highly volatile, high-stakes matching problem for millions of students. Traditional approaches rely on deterministic cutoff predictions, which fail to capture systemic volatility and often lead to sub-optimal choice filling. In this paper, we propose a probabilistic recommendation framework that leverages Ridge Regression combined with Monte Carlo simulation to output admission probabilities and uncertainty bounds. Our evaluation demonstrates significant improvements in recommendation stability and safety margins compared to point-prediction baselines.

## 1. Introduction
The Joint Seat Allocation Authority (JoSAA) manages the allocation of over 50,000 engineering seats across 100+ premier institutes (IITs, NITs, IIITs) in India. Students must rank their preferences prior to allocation. Given the intense competition, a sub-optimal preference list can cost a student a seat. We present a machine learning system that transforms historical cutoff data into actionable, uncertainty-aware recommendations.

## 2. Dataset
We aggregated 9 years of historical JoSAA cutoffs (2016-2024), representing over 70,000 unique allocation thresholds across various categories, genders, and quotas. Data normalization was critical due to frequent renaming of institutes and the introduction of new specialized programs (e.g., Artificial Intelligence).

## 3. Methodology
### 3.1 Prediction Model
We framed cutoff prediction as a regression task. Given the high cardinality of features, we utilized label encoding followed by $L2$-regularized Ridge Regression ($lpha=10.0$). The model achieved an $R^2$ of 0.98 on the holdout 2024 dataset, outperforming tree-based methods due to its robust handling of continuous linear trend lines across years.

### 3.2 Monte Carlo Uncertainty Engine
To quantify epistemic and aleatoric uncertainty, we mapped the empirical residual distribution grouped by Institute Type and Allocation Round. We injected a baseline volatility floor ($\sigma_{min} = 100$). For inference, we simulated $N=1000$ draws from $\mathcal{N}(\mu_{pred}, \sigma_{empirical})$ to generate a predictive admission distribution.

### 3.3 Recommendation Engine
Recommendations are scored using a composite function of Admission Probability (derived from the MC distribution) and normalized Safety Margin. Candidates are classified into *Safe* ($P \geq 0.8$), *Moderate* ($0.4 \leq P < 0.8$), and *Ambitious* ($P < 0.4$).

## 4. Experiments & Results
An extensive evaluation was conducted on the 2024 testing block. 
- **Ablation**: Incorporating MC probability improved the expected safety margin of top-K recommendations by filtering out brittle point-predictions.
- **Robustness**: Kendall Tau rank correlation remained highly stable ($	au > 0.95$) under student rank perturbations of $\pm 500$, indicating robust sorting properties.
- **Feature Importance**: Permutation importance revealed that explicit `opening_rank` and `institute` encodings were the primary drivers of prediction variance.

## 5. Conclusion
By shifting the paradigm from deterministic cutoff prediction to probabilistic decision-support, our system mitigates the inherent risk of seat-allocation volatility, providing students with mathematically grounded, highly interpretable counseling recommendations.
