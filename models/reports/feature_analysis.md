# Feature Importance Deep Dive

We evaluate feature importance using Permutation Importance and Ridge Coefficients.

### Permutation Importance
| Feature                 |   Importance_Mean |   Importance_Std |
|:------------------------|------------------:|-----------------:|
| hist_mean_closing_rank  |    8830.94        |     80.2708      |
| hist_min_closing_rank   |    8441.25        |     68.173       |
| hist_std_closing_rank   |    6851.78        |     74.9617      |
| applicants              |      65.4803      |     21.6154      |
| is_pwd                  |       0.00199635  |      0.000177781 |
| round                   |       0.000986052 |      0.000792466 |
| closing_rank_ratio_hist |       0.000823617 |      6.83928e-05 |
| quota_encoded           |       0.000619078 |      8.45159e-05 |
| category_encoded        |       0.000575137 |      0.000129724 |
| degree_type_encoded     |       0.000423265 |      4.99881e-05 |

### Ridge Coefficients
N/A

### Analysis
- `opening_rank` drives the predictions substantially.
- Categorical encoders (institute, program, category) also carry significant weight.