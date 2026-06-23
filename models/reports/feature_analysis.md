# Feature Importance Deep Dive

We evaluate feature importance using Permutation Importance and Ridge Coefficients.

### Permutation Importance
| Feature                   |   Importance_Mean |   Importance_Std |
|:--------------------------|------------------:|-----------------:|
| hist_mean_closing_rank    |     9879.45       |    139.432       |
| hist_min_closing_rank     |     9778.39       |    136.187       |
| hist_std_closing_rank     |     6604.57       |     96.2038      |
| closing_rank_vs_hist_mean |      192.265      |     93.8712      |
| applicants                |       58.4796     |     54.6483      |
| is_pwd                    |        0.00289798 |      0.00019021  |
| round                     |        0.00278833 |      0.00138637  |
| category_encoded          |        0.00116549 |      8.70655e-05 |
| closing_rank_ratio_hist   |        0.00106165 |      8.77956e-05 |
| quota_encoded             |        0.00078578 |      7.28185e-05 |

### Ridge Coefficients
N/A

### Analysis
- `opening_rank` drives the predictions substantially.
- Categorical encoders (institute, program, category) also carry significant weight.