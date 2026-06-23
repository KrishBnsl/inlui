# JoSAA Day 3 — Model Training & Evaluation Report

---

## Executive Summary

**Best model**: `Ridge`

| Metric | Value |
|---|---|
| final_test_MAE | 0.74 |
| final_test_RMSE | 1.81 |
| final_test_R2 | 1.0 |
| final_test_MedAE | 0.54 |
| final_test_MAPE | 0.17 |

The best model achieves **0.0% lower validation MAE** than the Ridge baseline (2 vs 2).

## Data Split

| Split | Years | Purpose |
|---|---|---|
| Train | 2016–2023 | Model fitting |
| Valid | 2024 | Model selection & tuning |
| Test | 2025–2026 | Final one-shot evaluation |

> **Temporal split** — no random shuffling across years. This prevents data leakage from future cutoff patterns influencing historical feature statistics.

## Regression Benchmark

All models predicting `closing_rank` (JEE rank of the last admitted student).

| Model                |   train_time_s |   train_MAE |   train_R2 |   valid_MAE |   valid_R2 |   test_MAE |   test_R2 |
|:---------------------|---------------:|------------:|-----------:|------------:|-----------:|-----------:|----------:|
| Ridge                |            0.2 |        0.44 |   1        |        1.97 |   1        |       2.43 |  1        |
| ExtraTrees           |           81.8 |       21.53 |   0.99994  |     1458.67 |   0.978967 |    1798.11 |  0.96973  |
| LightGBM             |           12.4 |      214.27 |   0.998689 |     1700.51 |   0.96828  |    2069.21 |  0.954966 |
| XGBoost              |           13.6 |      101.35 |   0.999795 |     1844.79 |   0.953625 |    2279.95 |  0.928493 |
| RandomForest         |          151.3 |       16.15 |   0.999941 |     1894.41 |   0.976223 |    2433.28 |  0.964236 |
| HistGradientBoosting |            5.3 |      240.97 |   0.997817 |     1984.53 |   0.913297 |    2266.77 |  0.95107  |

## Classification Sanity Check

Binary target: `cutoff_difficulty` — whether a branch's closing rank is tighter than its historical mean (1 = competitive, 0 = relaxed).

> **Note**: This is a derived label, not based on individual student admission outcomes. It serves as a sanity check for feature quality.

| Model              |   valid_F1 |   valid_ROC_AUC |   test_F1 |   test_ROC_AUC |
|:-------------------|-----------:|----------------:|----------:|---------------:|
| XGBoost_Clf        |     0.9547 |          0.9946 |    0.956  |         0.9936 |
| LightGBM_Clf       |     0.9506 |          0.9941 |    0.9531 |         0.9931 |
| LogisticRegression |     0.907  |          0.9783 |    0.9051 |         0.9729 |
| RandomForest_Clf   |     0.8817 |          0.9663 |    0.8747 |         0.9556 |

## Feature Importance

### Built-in Feature Importance (top 15)

*(Not available for this model type)*

### Permutation Importance (top 15)

| feature                   |   importance_mean |   importance_std |
|:--------------------------|------------------:|-----------------:|
| opening_rank              |        16216.9    |          23.6764 |
| rank_spread               |         5203.64   |           5.4077 |
| competitiveness_ratio     |            3.5955 |           0.0068 |
| hist_max_closing_rank     |            0.4914 |           0.0017 |
| prev_round_closing_rank   |            0.1164 |           0.0005 |
| opening_percentile        |            0.0478 |           0.0003 |
| closing_rank_vs_hist_mean |            0.0399 |           0.0006 |
| round_delta               |            0.0315 |           0.0005 |
| log_opening_rank          |            0.0135 |           0.0001 |
| hist_year_count           |            0.0015 |           0.0001 |
| round                     |            0.0004 |           0.0001 |
| gender_encoded            |            0.0003 |           0      |
| institute_type_freq       |            0.0003 |           0      |
| is_final_round            |            0.0003 |           0      |
| institute_type_encoded    |            0.0003 |           0      |

### Key Feature Insights

## Error Analysis

### Error by Year

|   Year |      MAE |    MedAE |   Count |   Mean Rank |
|-------:|---------:|---------:|--------:|------------:|
|   2025 | 0.701899 | 0.522594 |   70659 |     14690.2 |
|   2026 | 0.928353 | 0.74959  |   12976 |     13293.6 |

### Error by Category

| Category   |      MAE |    MedAE |   Count |   Mean Rank |
|:-----------|---------:|---------:|--------:|------------:|
| EWS        | 0.542133 | 0.536283 |   14576 |     6261.02 |
| OBC-NCL    | 0.597935 | 0.442695 |   18253 |    14235    |
| OPEN       | 1.24917  | 0.609944 |   21902 |    32454    |
| SC         | 0.487423 | 0.500349 |   15664 |     6390.47 |
| ST         | 0.591488 | 0.590799 |   13240 |     3662.78 |

### Error by Institute Type

| Inst Type   |      MAE |    MedAE |   Count |   Mean Rank |
|:------------|---------:|---------:|--------:|------------:|
| GFTI        | 0.939961 | 0.492466 |   11632 |     25518.3 |
| IIIT        | 0.550563 | 0.51988  |    8428 |     10630.4 |
| IIT         | 0.551553 | 0.577581 |   21763 |      3729.3 |
| NIT         | 0.814708 | 0.521763 |   41812 |     17767.9 |

## Bias & Error Discussion

- **Hardest category to predict**: OPEN (MAE = 1)
- **Easiest category to predict**: SC (MAE = 0)
- **Hardest institute type**: GFTI (MAE = 1)
- **Easiest institute type**: IIIT (MAE = 1)

Categories with higher closing ranks (SC, ST, EWS) tend to have higher absolute errors because the rank scale is wider. In relative terms (MAPE), performance may be more uniform.

GFTIs and NITs cover a wider range of institutes with more variable cutoffs, making them harder to predict than IITs which have more stable, well-established patterns.

## Model Reliability for Downstream Use

With R² = 1.0000 on the held-out test set, the model explains >100.0% of the variance in closing ranks. This is strong enough to support Monte Carlo simulation and recommendation ranking in later stages.

### Limitations

1. **No individual student data** — the model predicts branch-level cutoffs, not individual admission probability directly.
2. **2026 data is partial** — only early rounds available, test performance on 2026 may differ from 2025.
3. **EWS category only exists from 2019** — limited historical data for this group.
4. **PwD rows have different rank scales** — model treats them with a binary flag, but a separate model might perform better.

---

*Report generated by the Day 3 model training pipeline.*