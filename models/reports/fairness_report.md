# Fairness Analysis Report

This report evaluates the model's performance across different student demographic categories.

### Grouped Metrics
| Category   |   Count |      MAE |     RMSE |   Avg Uncertainty Width |
|:-----------|--------:|---------:|---------:|------------------------:|
| SC         |   14785 | 0.480081 | 0.713937 |                     329 |
| OBC-NCL    |   15016 | 0.484483 | 1.53998  |                     329 |
| EWS        |   13590 | 0.575865 | 0.823407 |                     329 |
| ST         |   12974 | 0.809619 | 0.899874 |                     329 |
| OPEN       |   16465 | 1.41543  | 3.65278  |                     329 |

### Findings
- The model shows varying degrees of absolute error across categories.
- Categories with typically lower closing ranks (e.g. SC, ST) may exhibit different MAE than OPEN simply due to the magnitude of ranks.