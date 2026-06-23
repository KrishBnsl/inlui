# JoSAA Day 2 — Exploratory Data Analysis Report

---

## Dataset Overview

| Metric | Value |
|---|---|
| Total records | 521,160 |
| Columns | 24 |
| Years | 11 (2016–2026) |
| Rounds | 7 (1–7) |
| Unique institutes | 151 |
| Unique programs | 352 |
| Unique disciplines | 352 |
| Unique categorys | 5 |
| Unique institute_types | 4 |
| Unique quotas | 4 |
| Unique genders | 2 |

## Missing Values

| Column | Missing Count |
|---|---|
| `closing_rank_yoy_change` | 439,206 |
| `round_progression` | 98,823 |

## `closing_rank` Statistics

| Stat | Value |
|---|---|
| Mean | 13,376.3 |
| Median | 4,298.0 |
| Std Dev | 42,265.4 |
| Min | 1 |
| Max | 1,497,038 |
| Q25 | 1,332.0 |
| Q75 | 11,678.0 |

## `opening_rank` Statistics

| Stat | Value |
|---|---|
| Mean | 10,309.1 |
| Median | 3,499.0 |
| Std Dev | 29,750.2 |
| Min | 1 |
| Max | 1,497,038 |
| Q25 | 1,064.0 |
| Q75 | 9,550.0 |

## `rank_spread` Statistics

| Stat | Value |
|---|---|
| Mean | 3,067.3 |
| Median | 304.0 |
| Std Dev | 19,354.5 |
| Min | -3,557 |
| Max | 982,920 |
| Q25 | 0.0 |
| Q75 | 1,600.0 |

## Top 15 Institutes by Record Count

| Rank | Institute | Records |
|---|---|---|
| 1 | National Institute of Technology, Rourkela | 19,768 |
| 2 | Indian Institute of Technology Kharagpur | 16,340 |
| 3 | National Institute of Technology Raipur | 13,705 |
| 4 | National Institute of Technology, Warangal | 13,656 |
| 5 | National Institute of Technology Calicut | 13,103 |
| 6 | National Institute of Technology, Tiruchirappalli | 12,760 |
| 7 | Dr. B R Ambedkar National Institute of Technology, Jalandhar | 12,638 |
| 8 | National Institute of Technology Karnataka, Surathkal | 12,282 |
| 9 | Maulana Azad National Institute of Technology Bhopal | 12,082 |
| 10 | National Institute of Technology Patna | 11,801 |
| 11 | Motilal Nehru National Institute of Technology Allahabad | 11,459 |
| 12 | Visvesvaraya National Institute of Technology, Nagpur | 11,342 |
| 13 | National Institute of Technology Durgapur | 11,274 |
| 14 | National Institute of Technology Agartala | 10,911 |
| 15 | Indian Institute of Engineering Science and Technology, Shibpur | 10,883 |

## Top 15 Programs by Record Count

| Rank | Program | Records |
|---|---|---|
| 1 | Computer Science and Engineering | 73,358 |
| 2 | Electronics and Communication Engineering | 53,558 |
| 3 | Mechanical Engineering | 53,176 |
| 4 | Civil Engineering | 48,297 |
| 5 | Electrical Engineering | 37,211 |
| 6 | Chemical Engineering | 30,102 |
| 7 | Electrical and Electronics Engineering | 16,194 |
| 8 | Metallurgical and Materials Engineering | 15,431 |
| 9 | Architecture | 13,026 |
| 10 | Information Technology | 12,875 |
| 11 | Bio Technology | 9,628 |
| 12 | Mathematics and Computing | 8,185 |
| 13 | Engineering Physics | 8,144 |
| 14 | Mining Engineering | 6,226 |
| 15 | Chemistry | 5,094 |

## Rows per Year

| Year | Rows |
|---|---|
| 2016 | 24,670 |
| 2017 | 29,011 |
| 2018 | 46,221 |
| 2019 | 56,938 |
| 2020 | 53,011 |
| 2021 | 54,003 |
| 2022 | 57,180 |
| 2023 | 61,050 |
| 2024 | 55,441 |
| 2025 | 70,659 |
| 2026 | 12,976 |

## Category Distribution

| Category | Count | % |
|---|---|---|
| OPEN | 141,924 | 27.2% |
| OBC-NCL | 115,789 | 22.2% |
| SC | 102,696 | 19.7% |
| ST | 88,872 | 17.1% |
| EWS | 71,879 | 13.8% |

## Most Volatile Branches (highest CV of closing rank)

| Institute | Program | Category | CV | Mean Rank | Std | Years |
|---|---|---|---|---|---|---|
| National Institute of Technology Calicut | Architecture | OPEN | 1.8500 | 83 | 153 | 9 |
| Indian Institute of Technology Kharagpur | Agricultural and Food Engineer... | OPEN | 1.5603 | 110 | 172 | 3 |
| Maulana Azad National Institute of Techn... | Planning | SC | 1.5101 | 1,422 | 2,147 | 9 |
| National Institute of Technology Raipur | Architecture | EWS | 1.4428 | 2,996 | 4,323 | 7 |
| Visvesvaraya National Institute of Techn... | Architecture | ST | 1.3918 | 432 | 601 | 11 |
| National Institute of Technology, Tiruch... | Architecture | SC | 1.3192 | 163 | 215 | 9 |
| National Institute of Technology Calicut | Architecture | OPEN | 1.2983 | 308 | 399 | 9 |
| National Institute of Technology, Kuruks... | Computer Engineering | ST | 1.2400 | 4,707 | 5,837 | 6 |
| National Institute of Technology, Tiruch... | Architecture | EWS | 1.2320 | 1,092 | 1,345 | 7 |
| National Institute of Technology Karnata... | Electronics and Communication ... | OBC-NCL | 1.2212 | 222 | 271 | 9 |

## Most Stable Branches (lowest CV of closing rank)

| Institute | Program | Category | CV | Mean Rank | Std | Years |
|---|---|---|---|---|---|---|
| Indian Institute of Technology Patna | Economics | OBC-NCL | 0.0010 | 5,533 | 5 | 3 |
| Indian Institute of Technology Jodhpur | Chemistry with Specialization | OBC-NCL | 0.0047 | 9,077 | 42 | 3 |
| Indian Institute of Technology Mandi | BS in Chemical Sciences | ST | 0.0051 | 1,530 | 8 | 4 |
| Indian Institute of Technology Roorkee | Energy Engineering | OPEN | 0.0063 | 4,795 | 30 | 3 |
| Indian Institute of Technology Delhi | Biochemical Engineering and Bi... | SC | 0.0064 | 1,567 | 10 | 3 |
| Indian Institute of Technology Patna | Economics | ST | 0.0077 | 1,502 | 12 | 3 |
| National Institute of Technology, Rourke... | Artificial Intelligence | ST | 0.0080 | 315 | 3 | 3 |
| Sardar Vallabhbhai National Institute of... | Electronics and VLSI Engineeri... | OPEN | 0.0081 | 22,976 | 185 | 3 |
| Birla Institute of Technology, Deoghar O... | Computer Science and Engineeri... | SC | 0.0091 | 11,403 | 104 | 4 |
| National Institute of Technology, Kuruks... | Artificial Intelligence and Ma... | EWS | 0.0093 | 1,334 | 12 | 4 |

---

## Key Patterns & Modeling Recommendations

### Promising Features for Cutoff Prediction

1. **Historical closing rank** (group mean/std) — strongest signal for branch-level prediction
2. **Year trend** — cutoffs shift systematically as JEE applicant pool grows
3. **Round number** — later rounds generally have relaxed cutoffs
4. **Institute type** — IIT/NIT/IIIT/GFTI have very different rank ranges
5. **Category** — OPEN vs. reserved categories have distinct distributions
6. **Competitiveness ratio** — normalizes across years with different applicant counts
7. **YoY change & round progression** — capture momentum/trends

### Recommended Model Families

1. **Gradient Boosted Trees (XGBoost / LightGBM)** — best for tabular data with mixed types
2. **Random Forest** — solid baseline, handles non-linearity well
3. **Linear models (Ridge/Lasso)** — useful for interpretability and as a baseline
4. **Quantile Regression** — for prediction intervals (important for counselling)

### Data Quality Notes for Modeling

- 785 rows have `opening_rank > closing_rank` (legitimate JoSAA data, not errors)
- 2016–2017 data has no gender column (filled as Gender-Neutral)
- 2026 has only 1 round so far — limited for testing
- EWS category only appears from 2019 onwards
- PwD rows have very different rank scales — consider modeling separately