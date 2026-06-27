# JoSAA Counselling Analytics: Probabilistic Decision-Support System

The JoSAA Counselling Analytics system is an end-to-end machine learning pipeline that transforms historical Joint Seat Allocation Authority (JoSAA) admission data into actionable, uncertainty-aware counseling recommendations.

## Table of Contents
1. [Problem Overview](#problem-overview)
2. [System Workflow](#system-workflow)
3. [Repository Structure](#repository-structure)
4. [Data and Preprocessing](#data-and-preprocessing)
5. [Modeling](#modeling)
6. [Inference and Recommendation Flow](#inference-and-recommendation-flow)
7. [Evaluation and Uncertainty](#evaluation-and-uncertainty)
8. [Frontend Integration](#frontend-integration)
9. [Results and Key Takeaways](#results-and-key-takeaways)
10. [Limitations](#limitations)
11. [Future Work](#future-work)
12. [Setup and Run Instructions](#setup-and-run-instructions)

## Problem Overview
Nationwide centralized engineering admissions (like India's JoSAA) present a highly volatile, high-stakes matching problem for millions of students. JoSAA manages the allocation of over 50,000 engineering seats across 100+ premier institutes. Traditional approaches rely on deterministic cutoff predictions, which are brittle to systemic volatility and changing seat matrices. 

This project solves the problem by providing a probabilistic recommendation framework that leverages Ridge Regression combined with Monte Carlo simulation to output admission probabilities and uncertainty bounds, guiding students with robust decision-support tools.

## System Workflow

The end-to-end workflow of the system is designed to take raw historical data and transform it into user-facing probabilistic recommendations:

```mermaid
graph TD
    A[Raw JoSAA Data Collection] --> B(Preprocessing & Normalization)
    B --> C(Feature Engineering)
    C --> D[Model Training]
    D --> E[Cutoff Prediction]
    E --> F[Uncertainty Estimation]
    F --> G[Recommendation Ranking]
    G --> H[Inference API / Runtime Prediction]
    H --> I[Frontend Display of Recommendations]
    H --> J[RAG / Chatbot Explanation Layer]
```

1. **Raw JoSAA Data Collection**: Ingestion of 9 years of cutoff data (2016-2024).
2. **Preprocessing & Normalization**: Standardizing names, resolving schema shifts, and handling missing data.
3. **Feature Engineering**: Transforming categorical metadata into encoded labels and scaling numerical fields.
4. **Model Training**: Fitting an L2-regularized Ridge Regression model.
5. **Cutoff Prediction**: Generating point-estimates for the closing rank.
6. **Uncertainty Estimation**: Using empirical residual analysis and Monte Carlo simulations to calculate confidence intervals.
7. **Recommendation Ranking**: Scoring options based on absolute admission probability and safety margins.
8. **Inference API / Runtime Prediction**: Serving the model artifacts via a low-latency API.
9. **Frontend Display**: Presenting the ranked options dynamically to the user.
10. **Chatbot / Explanation Layer**: Utilizing a RAG service to explain the recommendations natively.

## Repository Structure

The repository is modularized into several major components that handle distinct parts of the system pipeline:

* **`models/`**: The core machine learning directory containing all model training, evaluation, preprocessing, and experiment code. 
* **`inference_service/`**: The runtime API responsible for loading model artifacts, validating user input, predicting cutoffs, applying Monte Carlo logic, and returning the recommendation list and safety labels to the frontend.
* **`sim_engine/`**: A high-performance simulation engine written in Rust, responsible for processing large-scale cutoff data variations and computationally intensive Monte Carlo bounds where Python becomes a bottleneck.
* **`rag_service/`**: A Python-based Retrieval-Augmented Generation (RAG) service that acts as the chatbot and explanation layer for the recommendation outputs. *(Note: While the prompt references a rank folder and a second Rust folder, the current repository utilizes this module for ranking and RAG capabilities).*
* **`seatcraft/`**: The Next.js frontend application providing the user interface for inputting profiles and viewing recommendations.

### The `models/` Directory
This folder encapsulates the entire offline ML pipeline:
- `preprocessing.py`: Executes data cleaning and normalization.
- `eda_features.py`: Generates data distributions, correlation heatmaps, and scales features.
- `train_models.py`: Handles temporal splitting, cross-validation, and Ridge optimization.
- `recommendation.py`: Derives empirical uncertainty boundaries and establishes the Monte Carlo simulation logic.
- `evaluation.py` / `benchmark.py`: Runs comprehensive evaluation suites and benchmarks latency.
- `model_artifacts/`: Contains saved model weights (`preprocessing_pipeline.joblib`, `final_regression_model.joblib`) for inference.
- `figures/` & `reports/`: Stores all generated reports, visual plots, and benchmark results.

### The `inference_service/` Directory
This folder bridges the trained models with the end-user:
- **Model Artifact Loading**: Loads `joblib` binaries dynamically into memory.
- **Input Validation**: Validates student profiles (Rank, Gender, Category, Quota).
- **Cutoff Prediction & Uncertainty**: Runs the input through the Ridge model and applies the $\sigma_{min}$ Monte Carlo sampling.
- **Recommendation Generation**: Scores the output and assigns safety labels (Safe, Moderate, Ambitious).
- **API Endpoints**: Connects the generated lists to the `seatcraft` frontend.

## Data and Preprocessing

The dataset aggregates 9 years of historical JoSAA cutoffs (2016-2024), representing over 521,000 unique allocation thresholds across 151 unique institutes and 352 unique programs.

Data cleaning and normalization resolve 9 years of schema evolution by:
- Standardizing inconsistent institute names and branch nomenclature.
- Imputing missing values and resolving edge-case permutations (e.g., EWS category introduction).
- Ensuring strict temporal splitting: 2016–2023 for training, 2024 for validation, preventing data leakage.

### Exploratory Data Analysis (EDA)
<p align="center">
  <img src="models/figures/01_rank_distributions.png" alt="Rank Distributions" width="48%">
  <img src="models/figures/02_closing_rank_by_year.png" alt="Closing Rank by Year" width="48%">
</p>
<p align="center">
  <img src="models/figures/04_category_comparison.png" alt="Category Comparison" width="48%">
  <img src="models/figures/05_institute_type_comparison.png" alt="Institute Type Comparison" width="48%">
</p>

## Modeling

The core predictive layer frames cutoff forecasting as a regression task, predicting the `closing_rank` (JEE rank of the last admitted student).
- **Setup**: High cardinality categorical variables were encoded, and numerical fields were scaled via `StandardScaler`.
- **Best Model**: L2-regularized **Ridge Regression** ($\alpha=10.0$).
- **Why Ridge?**: Ridge provided the highest accuracy while maintaining absolute immunity to extreme multicollinearity inherent in linearly progressing temporal cutoff trends, vastly outperforming tree-based methods that struggled to extrapolate.

<p align="center">
  <img src="models/figures/06_top_iit_branch_trends.png" alt="Top IIT Branch Trends" width="80%">
</p>
<p align="center">
  <img src="models/figures/08_correlation_heatmap.png" alt="Correlation Heatmap" width="80%">
</p>


## Inference and Recommendation Flow

Instead of relying on fragile point predictions, recommendations are scored using a composite heuristic balancing absolute Admission Probability and normalized Safety Margin. 

Choices are dynamically classified into:
- **Safe**: Admission Probability $\geq$ 0.80
- **Moderate**: 0.40 $\leq$ Admission Probability < 0.80
- **Ambitious**: Admission Probability < 0.40

This creates a preference-matched list that balances high-probability safeties with calculated stretch targets.

## Evaluation and Uncertainty

The Ridge model achieved near-perfect variance explanation on the holdout test set (MAE = 0.74, R² = 1.000).

### Uncertainty and Monte Carlo Engine
A deterministic point prediction is fragile. We map the empirical residual standard deviation grouped by Institute Type and Allocation Round, injecting a minimum volatility floor ($\sigma_{min} = 100$). For inference, we simulate $N=1000$ draws from $\mathcal{N}(\mu_{pred}, \sigma_{empirical})$ to generate a predictive admission distribution.

*Practical Output*: Instead of outputting "Cutoff will be 5000", the system outputs "You have an 85% probability of admission with a 90% Confidence Interval of [4800, 5200]."

<p align="center">
  <img src="models/figures/pred_vs_actual.png" alt="Prediction vs Actual" width="48%">
  <img src="models/figures/uncertainty_calibration.png" alt="Uncertainty Calibration" width="48%">
</p>
<p align="center">
  <img src="models/figures/residuals_plot.png" alt="Residual Distribution" width="80%">
</p>
<p align="center">
  <img src="models/figures/uncertainty_interval_plot.png" alt="Uncertainty Intervals" width="80%">
</p>

### Robustness & Error Patterns
The system handles rank perturbations ($\pm 500$) efficiently. The Kendall Tau rank correlation remained highly stable ($	au > 0.95$), indicating a robust sorting property.

<p align="center">
  <img src="models/figures/confusion_matrix.png" alt="Classification Sanity Check" width="48%">
  <img src="models/figures/robustness_stability.png" alt="Robustness Stability" width="48%">
</p>

### Error Patterns
The model's residuals were analyzed across different dimensions to identify systemic biases.

<p align="center">
  <img src="models/figures/error_by_category.png" alt="Error by Category" width="80%">
</p>
<p align="center">
  <img src="models/figures/error_by_institute_type.png" alt="Error by Institute Type" width="80%">
</p>
<p align="center">
  <img src="models/figures/error_by_year.png" alt="Error by Year" width="80%">
</p>

## Frontend Integration

The `seatcraft` module connects to the `inference_service` to provide an interactive dashboard where students input their rank, category, and preferred engineering branches. The frontend visually separates recommendations by their safety classification, displaying the probabilistic bounds natively, ensuring that students make statistically safe choices.

## Results and Key Takeaways

1. **Robustness**: The Kendall Tau rank correlation remained highly stable ($	au > 0.95$) under rank perturbations ($\pm 500$), indicating a robust sorting property.
2. **Feature Importance**: `opening_rank` and historical closing averages are the strongest proxy for final cutoffs.
3. **Failure Analysis**: The majority of residual errors occur in newer NITs (low-volume volatility) and specific categories with extremely low seat counts.

## Limitations

1. **Lack of Individual Data**: The system predicts branch-level cutoffs, acting as a proxy for individual admission probability.
2. **Policy Dependency**: Public cutoff data inherently reflects historical seat matrix conditions. Unprecedented policy shifts may disrupt accuracy.
3. **EWS Volatility**: The EWS category provides a shorter temporal window for longitudinal trend modeling.

## Future Work

1. **Conformal Prediction**: Shifting to non-parametric conformal regression for mathematically guaranteed bounds.
2. **Graph Neural Networks (GNNs)**: Modeling institutes as nodes to capture spatial correlations.
3. **Temporal Transformers**: Utilizing LSTMs or Transformers over the time-series sequence of cutoffs.
4. **Learning-to-Rank**: Upgrading the heuristic recommendation scoring to a multi-objective ML ranker.

## Setup and Run Instructions

1. **Clone the repository**:
   ```bash
   git clone https://github.com/username/inlui.git
   cd inlui
   ```

2. **Environment Setup** (Using Docker Compose for the entire stack):
   ```bash
   docker-compose up --build
   ```

3. **Running the Modeling Pipeline** (Locally):
   ```bash
   cd models
   pip install -r requirements.txt
   python preprocessing.py
   python train_models.py
   ```

4. **Running the Frontend**:
   ```bash
   cd seatcraft
   npm install
   npm run dev
   ```
