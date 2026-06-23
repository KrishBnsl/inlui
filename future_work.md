# Research Extensions & Future Work

While the current system achieves high empirical accuracy, there are several promising avenues for future research to push the system towards state-of-the-art predictive bounds.

### 1. Conformal Prediction
Current uncertainty intervals rely on Gaussian assumptions over empirical residuals. Implementing non-parametric **Conformal Prediction** (e.g., using split-conformal regression) would provide mathematically guaranteed, distribution-free coverage bounds (e.g., exact 90% confidence intervals) regardless of the underlying data distribution.

### 2. Graph Neural Networks (GNNs)
Institutes and branches form implicit preference graphs (e.g., CSE at IIT Delhi is correlated with CSE at IIT Bombay). Training a **Graph Neural Network** over the seat allocation flow could natively capture spatial dependencies between institutes that tabular features currently miss.

### 3. Temporal Transformers
The current model uses `year` as a continuous linear feature. Utilizing a **Temporal Transformer** or LSTM over the time-series sequence of cutoffs (2016 -> 2024) could better capture velocity and acceleration in branch popularity (e.g., the sudden rise in AI/Data Science programs).

### 4. Bayesian Neural Networks
Replacing the Ridge Regressor with a **Bayesian Neural Network (BNN)** would allow the model to learn epistemic uncertainty directly in its weights, automatically outputting probability distributions without requiring a secondary empirical residual mapping.

### 5. Multi-objective Recommendation
Presently, recommendations rank primarily by Admission Probability and Safety Margin. A multi-objective **Learning-to-Rank** approach could jointly optimize for probability, historical placement metrics, and geographical proximity to the student.
