# SeatCraft Technical Report

## 1. Executive research summary

SeatCraft is an exploratory engineering study of pre-counselling JoSAA cutoff
forecasting and uncertainty-aware choice-list construction. The primary
research question is deliberately narrower than predicting an individual
admission: can a future closing rank be forecast using only information that
would exist before the target counselling season?

The central result is a methodology correction. An earlier headline of roughly
MAE 2.43 and R² 1.0 used same-season opening-rank features, the observed
target-year programme universe, and final-test residuals for uncertainty. Those
inputs do not answer a pre-counselling question, so that result is invalidated.
The strict experiment builds each target frame from an earlier snapshot,
selects the model on 2021–2024 only, and opens the 2025 holdout afterward.

Mean validation MAE selects the simple last-year cutoff baseline, not Ridge or
histogram gradient boosting. On 52,831 matched 2025 rows, it obtains MAE
2,504.75, RMSE 18,241.87, median absolute error 422, and R² 0.79269. Its nominal
90% empirical prediction interval covers 91.70% of that matched cohort, with a
mean width of 8,646.75 ranks. Only 74.77% of the observed 2025 universe is
represented by the prior-year frame. These values come from the canonical
machine-readable evidence in
[`research/reports/metrics.json`](../research/reports/metrics.json).

This remains exploratory evidence. The local dataset is hash-bound, but its raw
acquisition record and redistribution permission are absent. A clean clone must
therefore fail closed until provenance-verified source data is supplied.
SeatCraft is decision support, not an admission guarantee.

## 2. Research question

How accurately can JoSAA closing ranks be forecast using only information
available before a target counselling season, and how can uncertainty-aware
ranking support safer choice-list construction?

Historical cutoffs are often treated as deterministic forecasts even though
programmes, seat pools, category ranks, round schedules, policy, and student
preferences change. The study separates four testable hypotheses:

1. A strictly past-only estimator can improve MAE over a historical-mean
   baseline.
2. A nonlinear tabular model can improve large-error behaviour relative to an
   interpretable Ridge comparator.
3. Earlier rolling-origin residuals can produce a useful empirical 90%
   prediction interval without consuming the final test outcome.
4. Same-season opening-rank features make performance appear materially better
   but are unavailable for a genuine pre-counselling forecast.

Current evidence supports the first hypothesis for the selected last-year
rule. Histogram gradient boosting does not improve MAE or RMSE over last year
on the selection origins; it does obtain better RMSE and R² on the later final
cohort, so the second hypothesis depends on the declared objective and period.
The interval reaches 91.70% coverage on 2025 without a formal
coverage guarantee. The legacy opening-rank experiment is useful only as a
leakage warning: it also uses the observed target-year universe, so it is not a
clean causal ablation.

The product adds eligibility filters, exam-rank routing, empirical uncertainty,
and transparent heuristic ordering. Those mechanisms are not evidence of a
calibrated individual-admission model.

## 3. Dataset and provenance

The executable run uses the local file:

```text
research/data/processed/josaa_cleaned.csv
SHA-256 6fa3ff7ec8f6823a4ff158ec2b1c49fc79d7f81d23255420803c8ec604045039
size    109,657,862 bytes
```

The full local table has 521,160 rows spanning 2016–2026. The strict experiment
selects 508,184 rows from 2016–2025 using an explicit 2025 data cutoff. The
partial 2026 observations in the local file are never treated as complete and
are not loaded into the generated 2026 serving frame.

The generated quality payload records 12,976 post-cutoff rows, zero missing
required-field cells, zero duplicate outcome-key rows, and 785 rows whose
recorded opening rank exceeds the closing rank. The latter rows are retained as
source-quality flags rather than silently reversed without authoritative data.

One observation is a year/round outcome for:

```text
institute × programme × category × quota × gender pool × PwD status × round
```

Core fields cover year, round, institute, institute type, programme, duration,
degree type, quota, category, PwD flag, gender pool, opening rank, and closing
rank. The table contains no applicant names or direct personal identifiers.
Category, quota, gender-pool, and PwD aggregates are nevertheless socially
sensitive and require careful interpretation.

The repository lacks the immutable raw inputs, source URLs, retrieval dates,
original checksums, acquisition log, and permission/license record needed to
prove the processed table's origin. A `source_file` value such as
`2024_round_1.csv` is a filename, not provenance. The evidence is consequently
labelled `exploratory_unverified_data_provenance`, with provenance `unverified`,
license `unresolved`, and redistribution `not_authorized`.

JoSAA publishes official opening/closing-rank material on its
[archive page](https://josaa.nic.in/archive/). Its
[copyright policy](https://josaa.nic.in/copyright-policy/) requires appropriate
permission and source acknowledgement for reproduction. These pages identify a
possible future acquisition route; they do not prove that the current local
CSV came from that route or authorize its redistribution.

The former CSV under `optional/rust-sim/` is a historical collaborator fixture
that remains in repository history but is intentionally untracked in the
current tree. Its acquisition source, license, and redistribution basis are
unresolved; history also records mock-data modifications. It is excluded from
all primary research evidence, must not be substituted for the local research
table, and may be supplied locally only when the user has an authorized source.

The repository's MIT license covers the software source; it does not establish
reuse or redistribution rights for either provenance-unverified cutoff table.

### 3.1 Data dictionary summary

| Field family | Canonical examples | Role in the strict study |
|---|---|---|
| Temporal identity | `year`, `round` | Defines rolling origins and planned target rounds |
| Programme identity | `institute`, `institute_type`, `program`, `duration_years`, `degree_type` | Identifies the forecasted offering; encoded only inside fitted pipelines |
| Seat profile | `category`, `quota`, `gender`, `is_pwd` | Defines a distinct aggregate cutoff stratum and eligibility filter |
| Observed ranks | `opening_rank`, `closing_rank` | Closing rank is the target; same-round opening rank is excluded from strict features |
| Trace fields | `source_file`, `source_row_id` | Local row traceability only; not a verified acquisition record |
| Derived historical fields | mean, standard deviation, count, last cutoff, years since observed | Computed from years strictly earlier than each target |
| In-round fields | previous completed round number and closing rank | Allowed only for an explicitly declared in-round update |

The preprocessing path coerces common null tokens, parses numeric and
PwD-suffixed ranks, decomposes seat type into category and PwD status,
normalizes quota and gender-pool labels, parses programme duration/degree,
derives institute type, and removes duplicate natural keys while retaining the
first source trace. Normalization mappings are exported as machine-readable
JSON. The strict evaluator validates required fields and unique
year/outcome identities again before feature construction. Missing historical
statistics are retained and imputed inside each fitted estimator; missing
targets are never admitted to evaluation.

Known data risks include varying round counts between seasons, renamed or new
programmes, the shorter EWS history beginning in 2019, category-specific rank
scales, and rows whose recorded opening rank exceeds the closing rank. These
conditions are quality signals to preserve and investigate, not values to
silently repair without an authoritative source.

## 4. Data preparation

`research/pipeline/preprocessing.py` implements schema harmonization, rank
parsing, value normalization, de-duplication, institute-type extraction,
programme parsing, and auxiliary joins. The strict temporal feature builder in
`packages/josaa-core/src/josaa_core/research_features.py` then:

1. validates uniqueness on the year/outcome identity;
2. shifts every historical statistic by at least one year;
3. constructs each target frame from an explicitly declared prior-year
   snapshot;
4. removes opening rank, closing rank, source fields, target-year applicant
   counts, and other observed-only fields from that frame;
5. computes historical summaries only from years strictly before the target;
6. predicts before joining target outcomes; and
7. reports both unmatched observed rows and forecast rows without an observed
   target counterpart.

For a target origin `t`, the evaluator loads observations with `year < t`,
copies the programme/profile/round universe from snapshot `t-1`, replaces the
year with `t`, merges past-only features, and makes predictions. Observed
year-`t` closing ranks are joined afterward solely to score the matched cohort.
This prevents the legacy mistake of relabelling observed target-year feature
rows as a future forecast.

The code specifies a rebuild procedure, but the absent raw inputs and
transformation log mean it cannot prove that it produced the exact hashed local
CSV. A future publication-grade pipeline needs immutable raw files, an
acquisition manifest, source checksums, and a documented permission decision.

## 5. Feature-availability policy

The executable policy lives in
`packages/josaa-core/src/josaa_core/feature_schema.py` and
`packages/josaa-core/src/josaa_core/research_features.py`.

| Feature family | Raw inputs | `pre_counselling` | `in_round_update` | Availability rule |
|---|---|---:|---:|---|
| Target year and planned round | year, round | yes | yes | Explicit planning identifiers; round universe copied from a prior snapshot |
| Programme/profile metadata | institute, programme, category, quota, gender, PwD, degree | yes | yes | Copied from the declared prior-year snapshot |
| Historical mean/std/count | earlier closing ranks and years | yes | yes | Every contributing row has `year < target_year` |
| Last closing rank | earlier closing rank | yes | yes | Most recent earlier year for the same identity and round |
| Previous-round closing rank | completed target-year round | no | yes | Must be strictly earlier than the requested round |
| Same-round opening rank/log/percentile | target-round opening rank | no | no | Published with or after that round; unavailable for its forecast |
| `is_final_round` | observed target-year maximum round | no | no | Observes the target season |
| Target-year applicant count | applicant count | no | no | No provenance-verified pre-season source is versioned |
| Full-table frequency encodings | all rows | no | no | Can encode validation/test universe composition |
| Past-only institute/programme counts | rows with `year < target_year` | yes | yes | Recomputed per origin; current-year rows never contribute |
| Closing rank and derivatives | target outcome | no | no | Supervised answer |

`pre_counselling` is the primary professor-facing mode. `in_round_update` may
consume a completed earlier round, never the requested or a later round. That
guard is implemented and unit tested. A secondary 2025 Round-5 diagnostic is
executed on its own target subset; it does not replace or directly compare with
the all-round pre-counselling benchmark.

## 6. Temporal evaluation design

The design is an expanding-window rolling-origin evaluation:

- warm-up origins 2019–2020 contribute earlier out-of-fold residual evidence;
- origins 2021–2024 are the only model-selection years;
- the pre-declared selection criterion is mean MAE;
- 2025 is the pipeline-held-out final test origin;
- the dataset cutoff is explicitly 2025;
- the generated serving frame predicts 2026 from the 2025 snapshot; and
- all stochastic procedures use seed 42.

The canonical experiment identifier is
`seatcraft-strict-pre-counselling-2021-2025-seed42-1c009aa9f98bbf23`; the
dataset version is `local-unverified-cutoff-2025-6fa3ff7ec8f6`, feature mode is
`pre_counselling`, and the recorded Git state is commit
`ee751d0f236961f612d867864d68abbb1d39d9ab` with a dirty worktree. Because the
experiment code is uncommitted, `metrics.json` additionally records SHA-256
hashes for the evaluator, feature policy, feature builder, baseline, and
registry source files.

The model is selected before the 2025 comparison is inspected. The target-year
programme universe is not observed in advance. In 2025, the forecast frame has
55,441 rows and the observed table has 70,659 rows. Of those, 52,831 match;
17,828 observed rows lie outside the prior-year universe, and 2,610 forecast
rows have no 2025 observation. Target-universe coverage is therefore
52,831 / 70,659 = 74.76896%. All final error metrics describe the matched
prior-universe cohort rather than every observed 2025 row.

Selected-model results by reported origin are:

| Year | Matched rows | MAE | RMSE | Median AE | R² | Prior-universe coverage |
|---:|---:|---:|---:|---:|---:|---:|
| 2021 | 48,988 | 3,092.99 | 16,130.29 | 508 | 0.81924 | 90.71% |
| 2022 | 50,615 | 2,384.31 | 13,416.75 | 468 | 0.84350 | 88.52% |
| 2023 | 53,738 | 2,274.71 | 13,260.74 | 516 | 0.87953 | 88.02% |
| 2024 | 47,826 | 2,389.82 | 14,651.54 | 456 | 0.88954 | 86.26% |
| **2025 final** | **52,831** | **2,504.75** | **18,241.87** | **422** | **0.79269** | **74.77%** |

## 7. Model card

The selected estimator is `LastObservationRegressor`. It predicts each target
cutoff as the most recent closing rank for the same
programme/profile/round identity from a strictly earlier year. Its sole model
feature is `last_closing_rank`.

| Contract field | Value |
|---|---|
| Model family | `LastObservationRegressor` |
| Model version | `pre-counselling-last-observation-cutoff2025-seed42-853816b3481e` |
| Feature mode | `pre_counselling` |
| Data cutoff / snapshot | 2025 |
| Prediction frame | 2026 |
| Model SHA-256 | `853816b3481e3d618e1530499101d8b995e742c428e15c5f2c48594a0856da24` |
| Dataset SHA-256 | `6fa3ff7ec8f6823a4ff158ec2b1c49fc79d7f81d23255420803c8ec604045039` |
| Seed | 42 |
| Selection metric | Mean MAE across pre-final rolling origins |
| Final test used in selection | No |

The small serving contracts in `research/artifacts/` are
`model_metadata.json`, `pre_counselling_feature_schema.json`, and
`uncertainty_calibration.json`. The generated model binary
`pre_counselling_model.joblib` and 2026 `inference_universe.csv` remain local
assets because the underlying source data cannot yet be redistributed. The
universe has 70,659 rows and contains no observed 2026 opening or closing ranks.

| Generated serving file | Bytes | SHA-256 | Expected in Git? |
|---|---:|---|---:|
| `pre_counselling_model.joblib` | 336 | `853816b3481e3d618e1530499101d8b995e742c428e15c5f2c48594a0856da24` | no |
| `inference_universe.csv` | 14,540,609 | `d9645f42c17e6bc715df12920b7f4aa4bc3c1f0072bb2d169c23c1182b93577b` | no |
| `pre_counselling_feature_schema.json` | 297 | `d0bafbb15483cdebc228404ee2a41178f64879208f4a9e734756172877fad798` | yes |
| `uncertainty_calibration.json` | 4,506 | `ba4cd6fa9898465e24b08d447f7e03a162a8a07c4c05c85ff59aba3eae7dc8ed` | yes |
| `model_metadata.json` | 2,047 | `3ce97385e74b9b83d9a01a922e03736f78a0d47fea58a8a07226dda17da02a6f` | yes |

Intended uses are a temporal research baseline, a conservative comparator for
richer forecasting methods, and cutoff context inside a choice-list
decision-support interface. Out-of-scope uses include individual admission
guarantees, empirically calibrated admission-probability claims, automated
exclusion or other high-stakes decisions, and unseen-program prediction without
a separately evaluated cold-start method.

## 8. Baselines and model selection

The experiment compares a naive last-year cutoff, historical mean, Ridge with
availability-correct features, and histogram gradient boosting. Candidate
ranking uses mean MAE over 2021–2024 only:

| Candidate | Mean MAE | Mean RMSE | Mean median AE |
|---|---:|---:|---:|
| **Naive last year** | **2,535.46** | 14,364.83 | **487.00** |
| Histogram gradient boosting | 2,878.39 | 15,740.18 | 823.42 |
| Historical mean | 3,279.50 | 15,378.47 | 712.50 |
| Ridge | 6,800.97 | 15,942.75 | 4,763.58 |

The last-year estimator wins the declared MAE objective and also has the lowest
validation RMSE and median error. The baseline victory is retained as a negative
ML result rather than hidden or overridden after inspecting 2025. Ridge remains
an interpretable experimental comparator, not the selected model.

The pipeline-held-out 2025 comparison is:

| Model | MAE | RMSE | Median AE | R² |
|---|---:|---:|---:|---:|
| **Selected last year** | **2,504.75** | 18,241.87 | **422.00** | 0.79269 |
| Histogram gradient boosting | 2,661.17 | **15,922.81** | 715.56 | **0.84205** |
| Historical mean | 3,041.86 | 16,143.33 | 578.57 | 0.83765 |
| Ridge | 5,609.05 | 16,162.10 | 3,612.28 | 0.83727 |

Last year retains the best MAE and median error; histogram gradient boosting has
the best 2025 RMSE and R². No single metric supports a universal-best claim.

### 8.1 Estimator specifications

| Estimator | Inputs/preprocessing | Fixed hyperparameters |
|---|---|---|
| Last year | Most recent strictly earlier cutoff for the same outcome identity and round | No learned coefficients |
| Historical mean | Expanding mean of strictly earlier cutoffs for that identity and round | No learned coefficients |
| Ridge | Median imputation with indicators and sparse scaling for numeric fields; most-frequent imputation and one-hot encoding (`min_frequency=2`) for categorical fields | `alpha=10`, `solver=lsqr`, `tol=1e-4` |
| Histogram gradient boosting | Median-imputed numeric fields and ordinal-encoded categorical fields with unknown value `-1` | squared-error loss, learning rate `0.06`, `max_iter=120`, `max_leaf_nodes=31`, `min_samples_leaf=30`, L2 `1.0`, seed `42` |

No post-holdout hyperparameter search is performed. These candidates and the
mean-validation-MAE selection rule are fixed in the evaluator. The canonical
JSON records the feature lists, parameters, code hashes, dataset identifier,
and experiment identifier needed to distinguish a rerun from a changed study.

## 9. Ablation studies

Because the selected model is itself a one-feature historical baseline, feature
ablations are applied to the Ridge comparator:

| Variant | Eligibility | 2025 MAE | 2025 RMSE | Interpretation |
|---|---|---:|---:|---|
| Ridge, strict pre-counselling | comparator | 5,609.05 | 16,162.10 | Full availability-correct Ridge feature set |
| Ridge without opening-rank features | policy-equivalent | 5,609.05 | 16,162.10 | Opening features are already forbidden |
| Ridge without previous-round features | policy-equivalent | 5,609.05 | 16,162.10 | Previous-round outcomes are already forbidden pre-season |
| Ridge without historical statistics | comparator refit | 12,598.15 | 32,660.80 | Large degradation when past cutoff summaries are removed |
| Ridge without past-only frequencies | comparator refit | 5,524.69 | 16,175.33 | Removing counts improves MAE by 84.36 ranks in this run |
| Legacy same-round opening diagnostic | **ineligible** | 3,347.09 | 9,978.72 | Uses unavailable opening rank and the observed target-year universe |

The legacy diagnostic is a leakage stress test, not an eligible candidate or a
clean causal feature ablation. The former approximate MAE 2.43 / R² 1.0 value is
not retained as current research evidence.

The separate in-round diagnostic forecasts 2025 Round 5 after Rounds 1–4 are
complete. It uses a matched cohort of 10,467 rows (89.37% of the observed
Round-5 universe):

| In-round Ridge variant | MAE | RMSE | Median AE | R² |
|---|---:|---:|---:|---:|
| With completed previous-round fields | 3,154.68 | 7,129.34 | 2,169.35 | 0.96991 |
| Without previous-round fields | 5,637.73 | 16,171.51 | 3,541.19 | 0.84519 |

These rows are comparable with each other, but not with the all-round primary
benchmark: they use a later information set and a different target subset.

## 10. Uncertainty and prediction intervals

For each origin, the selected estimator makes an out-of-fold prediction and the
interval radius is fixed before that origin's residual enters the calibration
pool. The method uses an empirical quantile of earlier absolute residuals from
the same selected model.

For the 2025 final test:

- nominal coverage is 90%;
- calibration uses 288,592 residuals from 2019–2024 only;
- the global absolute-residual radius is 4,173 ranks;
- institute-type/round radii are used when at least 100 calibration examples
  exist, otherwise the global radius is used;
- empirical coverage is 91.70184%;
- mean interval width is 8,646.75 ranks; and
- no 2025 test residual is used to construct the interval that covers 2025.

For the later generated 2026 serving frame, the now-historical 2025 OOF
residuals may legitimately enter calibration. Its artifact contains 341,423
residuals from 2019–2025 and a global radius of 4,073 ranks, with the same
minimum group count of 100.

These ranges are **empirical prediction intervals** for a future cutoff. They
are not confidence intervals, do not have a finite-sample conformal guarantee,
and can lose nominal coverage under temporal distribution shift. They do not
make the product's admission-likelihood estimate empirically calibrated to
individual admission outcomes.

## 11. Recommendation methodology

The inference API first applies exact normalized profile filters and then routes
the appropriate rank for every candidate row:

- IIT rows use JEE Advanced rank and are excluded when Advanced rank is absent;
- NIT, IIIT, and GFTI rows use JEE Main rank;
- category and PwD status are normalized and filtered independently;
- gender-neutral and applicable female-only pools are handled explicitly;
- the requested round is treated as schedule metadata, never as an observed
  target-round result in `pre_counselling` mode;
- institute types and case-insensitive branch keywords constrain the prior-year
  universe; and
- AI/OS rows are allowed, while HS rows are conservatively excluded unless
  verified institute-state metadata proves home-state eligibility.

For every row, `rank_used` and `rank_type_used` make the routing visible. The
cutoff model predicts `predicted_closing_rank`. A seeded Monte Carlo normal approximation,
scaled from rolling-origin residual evidence, estimates the
fraction of simulated cutoffs for which the routed student rank is admissible.
This raw `admission_probability` has no distance-based floor and is not calibrated
to individual admissions. The current
`calibrated_probability_percent` field remains null.

Let (r) be the routed candidate rank and \(\hat{c}\) the predicted closing
rank. The displayed rank margin and ratio are

\[
m = \hat{c} - r, \qquad \rho = \frac{r}{\max(\hat{c}, 1)}.
\]

For (S=1{,}000) deterministic-seed draws, the serving layer sets the normal
scale to the earlier rolling-origin 90% absolute-residual radius divided by
1.64485, samples \(C_s = \max(1, \hat{c} + \sigma z_s)\) with
\(z_s \sim \mathcal{N}(0,1)\), and reports

\[
\hat{p}_{\mathrm{cutoff}} = \frac{1}{S}\sum_{s=1}^{S}\mathbf{1}[r \le C_s].
\]

This is an uncalibrated **admission-likelihood estimate under the stated cutoff
model**, not an observed individual-admission probability. Monte Carlo sampling
approximates the chosen model distribution; it does not calibrate that
distribution or supply outcome labels that the dataset does not contain. The
separately displayed empirical prediction interval uses the OOF residual radius
directly rather than the Monte Carlo sample percentiles.

`recommendation_score` is a separate heuristic for ordering and fit. It combines
admissibility, closeness to the predicted cutoff, cutoff competitiveness,
institute type, and branch desirability. It must never be described as a
modeled cutoff-exceedance probability.

The implemented fit term is

\[
F = \exp\!\left(-\frac{(\log(\hat{c}/r)-\log(1.18))^2}{2(0.75)^2}\right)
    \min\!\left(\frac{\hat{p}_{\mathrm{cutoff}}}{0.35},1\right).
\]

With normalized cutoff-competitiveness \(C\), institute heuristic \(I\), and
branch heuristic \(B\), desirability is \(D=0.45C+0.25I+0.30B\). The
admissibility heuristic \(A\) rewards usable likelihood without allowing only
very safe rows to dominate. The final ordering score is

\[
100\left(0.45D\max(A,F)+0.30F+0.25A\right).
\]

The constants are an explicit design policy, not learned preference weights.
Learning-to-rank with preference or utility labels remains proposed work.

Choice-list buckets use the transparent ratio
`rank_used / predicted_closing_rank`:

| Bucket | Ratio rule | Meaning |
|---|---:|---|
| `safe_backup` | ≤ 0.80 | Student rank is comfortably better than the forecast cutoff |
| `best_realistic` | > 0.80 and ≤ 1.05 | Near-cutoff realistic target |
| `ambitious_reach` | > 1.05 and ≤ 1.40 | Moderately harder than the forecast cutoff |
| `unlikely_reach` | > 1.40 | Too far from the normal result window |

For the default `top_n=100`, best-fit mode aims for a 30/50/20 safe/realistic/
reach mix and scales those targets for other limits. It never invents rows to
meet a quota: unlikely reaches are excluded, only eligible rows are returned,
and the API exposes universe, post-filter, scored, returned, institute-type,
and bucket counts. Other explicit sort modes are highest likelihood estimate, most
competitive cutoff, and safest backup.

## 12. Results and subgroup error analysis

The primary final-test result is the selected-model row in Section 8. Its
median error of 422 ranks and RMSE of 18,241.87 expose a heavy-tailed error
distribution that a headline MAE alone would hide. The matched-cohort boundary
and 74.77% universe coverage must accompany the error metrics.

Pooled 2021–2025 descriptive slices include:

- IIT MAE 368.59 versus NIT MAE 3,788.49;
- OPEN-category MAE 5,439.56 versus ST-category MAE 891.98; and
- MAE 364.27 for actual ranks 1–5,000 versus 73,551.90 above 100,000.

The full counts and metrics are in
[`research/reports/research_subgroup_metrics.csv`](../research/reports/research_subgroup_metrics.csv).
These strata use different rank lists, ranges, sample sizes, and programme
mixes. Negative R² in narrow pooled slices can coexist with useful absolute
errors. The slices are diagnostics, not evidence of fairness or unfairness.
A defensible fairness study requires comparable outcomes, uncertainty, minimum
sample thresholds, explicit questions, and domain review.

The canonical supporting tables are:

- [model selection](../research/reports/research_model_selection.csv);
- [rolling-origin folds](../research/reports/research_fold_metrics.csv);
- [ablations](../research/reports/research_ablation_metrics.csv);
- [in-round diagnostic](../research/reports/research_in_round_diagnostic.csv); and
- [artifact/data manifest](../research/reports/artifact_manifest.json).

The six strict figures are generated by
`research/pipeline/research_evaluation.py` and hash-listed in both
`metrics.json` and the manifest:

- [predicted versus actual](../research/reports/figures/strict_pred_vs_actual.png);
- [residual distribution](../research/reports/figures/strict_residual_distribution.png);
- [error by year](../research/reports/figures/strict_error_by_year.png);
- [error by institute type](../research/reports/figures/strict_error_by_institute_type.png);
- [empirical interval coverage](../research/reports/figures/strict_interval_coverage.png); and
- [Ridge ablations](../research/reports/figures/strict_ablation.png).

## 13. RAG/Gemini methodology

The advisor is an independently testable FastAPI service under `apps/rag-api/`.
It uses dependency-injected Gemini embedding/chat clients and an in-memory,
thread-safe FAISS index. The frontend passes a bounded summary of the current ML
recommendations, retrieves up to the configured top-k document chunks, and
receives both an answer and source metadata.

The ingestion path:

1. strips client-supplied path components from filenames;
2. enforces a byte limit and cross-checks extension, declared MIME type, and
   file signature;
3. accepts PDF, PNG, JPEG, and WebP inputs;
4. rejects encrypted, malformed, oversized, empty, or unsupported documents;
5. limits PDFs to 500 pages and 2,000,000 extracted characters;
6. extracts page-aware PDF text or invokes the injected multimodal provider for
   image transcription;
7. chunks text with source/page/chunk metadata; and
8. embeds and inserts those chunks into the in-memory FAISS store.

Image validation is not evidence of OCR accuracy. The image path remains
provider-dependent and no local OCR-quality claim is made.

At answer time, the service performs similarity search, builds short source
excerpts, and invokes Gemini once with retrieved text and optional ML context.
System instructions treat the user question, documents, OCR text, excerpts, and
ML context as untrusted data. Inputs are escaped and placed in separate
boundaries; document instructions cannot override the grounding policy. The
assistant must not invent JoSAA rules, cutoffs, sources, or current policy, and
must direct users to official information when evidence is missing or
conflicting.

`GET /health` distinguishes healthy and degraded provider states, names the
configured embedding/chat models, and reports vector availability. A missing
`GOOGLE_API_KEY` produces a visible degraded state. Requests have bounded
service/provider timeouts, upstream details are sanitized, and tests inject
mock gateways rather than silently replacing real failures in production.
Mocked invocation proves application wiring, not live Gemini connectivity.

The current implementation returns retrieved source metadata and excerpts
beside an answer. It does not verify that the generated prose cites each excerpt
correctly, and it does not claim that retrieval eliminates hallucination. A
future evaluation set should pre-register document/question pairs and score:

| Dimension | Proposed measurement | Current evidence |
|---|---|---|
| Retrieval relevance | Recall@k, MRR, and blinded relevance labels for retrieved chunks | Not yet benchmarked |
| Citation correctness | Whether each answer attribution is entailed by the referenced excerpt | Not yet benchmarked |
| Faithfulness | Unsupported-claim rate against retrieved context and current model output | Not yet benchmarked |
| Answer completeness | Coverage of a reference answer's required facts | Not yet benchmarked |
| Insufficient context | Abstention/redirect behaviour when no supporting chunk is available | Defensive behaviour unit-tested; no dataset result |
| Prompt-injection robustness | Attack success rate for document-, OCR-, question-, and ML-context instructions | Boundary handling unit-tested; systematic evaluation proposed |

Human annotation guidance, inter-rater agreement, document licensing, and a
held-out evaluation split must be defined before any grounding score is
reported.

## 14. Reproducibility

The recorded strict run used:

| Component | Version |
|---|---|
| Python | 3.12.3 |
| NumPy | 1.26.4 |
| pandas | 2.2.3 |
| scikit-learn | 1.6.1 |
| joblib | 1.5.0 |
| Platform | WSL2 Linux 5.15.167.4, glibc 2.39, x86-64 |
| Seed | 42 |

Install the minimal research environment with:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-dev.txt
```

The strict evaluator requires the exact input hash given in Section 3. If the
asset is missing or different, evaluation must stop with a clear data-asset
blocker. There is deliberately no invented downloader. Once permission and
provenance-verified JoSAA round files exist under `research/data/`, rebuild in
this order:

```bash
python research/pipeline/preprocessing.py --data-dir research/data
python -m research.pipeline.research_evaluation \
  --data research/data/processed/josaa_cleaned.csv \
  --output-dir research/reports \
  --artifact-dir research/artifacts \
  --evaluation-years 2021 2022 2023 2024 2025 \
  --final-test-year 2025 \
  --data-cutoff-year 2025 \
  --prediction-year 2026
```

Never infer completeness from the maximum year in a file. The cutoff and
prediction year remain explicit because the local source includes partial 2026
records.

The checked evidence uses the baseline commit timestamp as a reproducible build
timestamp:

```bash
SOURCE_DATE_EPOCH=1783093190 \
python -m research.pipeline.research_evaluation \
  --data research/data/processed/josaa_cleaned.csv \
  --output-dir research/reports \
  --artifact-dir research/artifacts \
  --evaluation-years 2021 2022 2023 2024 2025 \
  --final-test-year 2025 \
  --data-cutoff-year 2025 \
  --prediction-year 2026
```

`SOURCE_DATE_EPOCH` is explicitly labelled as a reproducible build timestamp,
not the wall-clock execution time. Omitting it records execution time instead;
seeded metric values remain reproducible, but timestamp-bearing JSON hashes can
naturally differ.

Versioned research evidence belongs in `research/reports/`: `metrics.json`,
the selection/fold/ablation/subgroup CSVs, `artifact_manifest.json`, and
`model_registry.csv`. Small serving contracts belong in `research/artifacts/`.
The generated model binary and inference universe are local-only. The manifest
records hashes, expected availability, provenance, licensing, and whether each
asset is expected in Git.

`research/pipeline/build_registry.py` may rebuild the selected-model registry
only from the current metrics, selection evidence, and model metadata; it must
not resurrect the invalidated same-season benchmark under
`research/reports/legacy/`.

Interpretation rule: do not copy a metric into documentation or the interface
unless it exists in `research/reports/metrics.json` or a referenced canonical
CSV and the dataset hash matches the manifest. A new input hash defines a new
dataset and requires a new evidence bundle and review.

## 15. Limitations and threats to validity

- **Provenance and licensing:** the strongest blocker is the missing raw-source
  and permission chain, not model accuracy.
- **Prior-universe selection:** new, renamed, merged, discontinued, or
  seat-pool-changed programmes are absent unless represented in the earlier
  snapshot. Matched-cohort errors may be optimistic or pessimistic relative to
  the missing rows.
- **Simple selected rule:** last year won the declared validation MAE, but it
  cannot learn trends or structural changes. Histogram gradient boosting had
  better final RMSE/R², illustrating objective and period sensitivity.
- **Heavy-tailed ranks:** rank scales differ by category and large ranks
  dominate squared error. MAE, RMSE, median error, R², sample size, and universe
  coverage should be reported together.
- **Empirical intervals:** calibration uses historical continuity, group radii
  fall back when sparse, and distribution shift can break nominal coverage.
- **Availability assumptions:** target year and planned round are assumed known;
  target-year applicant counts remain excluded without an official versioned
  pre-season source.
- **Narrow in-round evidence:** the executed Round-5 comparison is a secondary
  diagnostic on one target subset, not a full multi-origin in-round benchmark.
- **Admission mechanics:** the model omits student preference dynamics, the
  allocation algorithm, withdrawals, float/slide behaviour, and policy or seat
  matrix changes.
- **Eligibility metadata:** home-state routing remains conservative until a
  verified institute-to-state mapping is available.
- **External validity:** policy, exam, seat-matrix, or counselling changes can
  invalidate historical continuity.
- **RAG evaluation:** retrieval, citation grounding, answer quality, and OCR
  accuracy lack a persistent benchmark; vectors disappear on restart.
- **Fresh-clone serving:** the large derived assets cannot be regenerated
  legitimately until source provenance and redistribution authority are
  resolved.

## 16. Ethical use and admission disclaimer

SeatCraft forecasts aggregate cutoff rows; it does not model a student's true
admission outcome. A cutoff prediction, Monte Carlo estimate, interval, bucket,
or heuristic score must not be presented as a guarantee. The tool must not be
used for automated exclusion or other high-stakes decisions. Students should
review official JoSAA rules, seat matrices, and counselling notices and retain
control over their submitted choice order.

Category, quota, gender-pool, and PwD fields describe aggregate seat strata, not
individual merit. Differences across their error slices do not establish
discrimination or fairness because rank scales and sample compositions differ.

Uploaded documents may contain private information. The current vector store is
in memory rather than durable storage, but users should still upload only
non-sensitive material. Retrieved text and generated answers can be incomplete
or wrong. Gemini output is non-authoritative even when it supplies a source
excerpt, and provider failure must remain visible rather than being replaced by
synthetic advice.

## 17. Research status and proposed work

### Completed and evidenced

- Strictly past-only feature construction and prior-snapshot target universes.
- Expanding-window model selection followed by an isolated 2025 final test.
- Last-year, historical-mean, Ridge, and nonlinear comparator execution.
- Earlier-origin empirical prediction intervals and coverage measurement.
- Descriptive error slices, feature-policy guards, serving artifacts, and an
  uncertainty-aware FastAPI recommendation path.
- Rank-source routing, recommendation grouping, Next.js integration, mocked
  RAG-provider tests, browser contract tests, and machine-readable verification.

### In progress

- Provenance reconstruction and permission review for the local processed
  dataset. No resolution is claimed until immutable source records exist.
- Separation of legacy exploratory artifacts from the strict evidence bundle
  and expansion of drift checks around public documentation.
- A reproducible RAG evaluation design; metrics have not yet been collected.

### Proposed research

1. Acquire provenance-verified raw files, document permission, and build an
   immutable acquisition/source manifest.
2. Compare split, weighted, and adaptive conformal intervals under temporal
   distribution shift, measuring both coverage and width.
3. Evaluate hierarchical, Bayesian, or time-series models that share strength
   across sparse institute/programme/profile groups.
4. Integrate an official pre-season seat matrix and benchmark cold-start
   coverage for unseen or renamed choices.
5. Learn choice-list ranking from explicit preference objectives and evaluate
   utility, safety coverage, and regret rather than only a hand-built heuristic.
6. Detect distribution shifts after seat-matrix, examination, or policy changes.
7. Design comparable-scale fairness/error analyses with uncertainty, minimum
   counts, intersectional slices, and domain review.
8. Persist the retrieval index and build a licensed RAG evaluation dataset for
   retrieval relevance, citation correctness, faithfulness, completeness,
   insufficient-context behaviour, OCR quality, and prompt-injection attacks.
9. Conduct human-centred evaluation with students or counsellors only after an
   appropriate ethical protocol, consent process, and non-coercive study design.

## 18. References and methodological influences

These sources motivate methods or define external interfaces; none endorses
SeatCraft, and their inclusion does not make this repository a publication.

1. Joint Seat Allocation Authority, [Archive](https://josaa.nic.in/archive/)
   and [Copyright Policy](https://josaa.nic.in/copyright-policy/). These are
   the authoritative pages identified for a future provenance-verified data
   acquisition path; they do not establish the origin or redistribution rights
   of the current local CSV.
2. A. E. Hoerl and R. W. Kennard, “Ridge Regression: Biased Estimation for
   Nonorthogonal Problems,” *Technometrics*, 12(1), 55–67, 1970,
   [doi:10.1080/00401706.1970.10488634](https://doi.org/10.1080/00401706.1970.10488634).
3. scikit-learn, [TimeSeriesSplit documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html).
   SeatCraft implements its own year-level expanding-window protocol, but
   follows the same principle that future observations must not train earlier
   evaluations.
4. V. Vovk, A. Gammerman, and G. Shafer, *Algorithmic Learning in a Random
   World*, Springer, 2005,
   [doi:10.1007/b106715](https://doi.org/10.1007/b106715). Conformal prediction
   is a proposed extension; the current empirical intervals are not presented
   as conformal.
5. C. J. C. Burges et al., “Learning to Rank using Gradient Descent,” ICML
   2005, [doi:10.1145/1102351.1102363](https://doi.org/10.1145/1102351.1102363).
   Preference-supervised ranking remains proposed work.
6. P. Lewis et al., “Retrieval-Augmented Generation for Knowledge-Intensive
   NLP Tasks,” *NeurIPS 2020*,
   [proceedings page](https://proceedings.neurips.cc/paper/2020/hash/6b493230-Abstract.html).
7. M. Douze et al., “The Faiss library,” arXiv:2401.08281, 2024,
   [project citation](https://github.com/facebookresearch/faiss#reference).
8. Google AI for Developers, [Gemini embeddings documentation](https://ai.google.dev/gemini-api/docs/embeddings).

Titles, author lists, venues, DOI targets, and documentation URLs above were
checked against the publisher, proceedings, project, or official documentation
pages on 2026-07-11.

## 19. Short professor-facing project summary

Students often treat last year's JoSAA cutoff as a guarantee. SeatCraft turns
that habit into a testable temporal-forecasting question with changing
programme universes, sparse groups, heavy-tailed errors, strict feature-time
availability, and uncertainty under distribution shift. It also provides a
product setting where JEE Main versus Advanced routing and recommendation
semantics must remain auditable.

The most important result is research integrity rather than an inflated score:
an apparently excellent MAE 2.43 result collapsed after target-season leakage
and test-residual uncertainty were removed. Pre-final selection instead chooses
the last-year baseline at mean 2021–2024 MAE 2,535.46. On the pipeline-held-out 2025
matched cohort it obtains MAE 2,504.75, median error 422, and 91.70% empirical
coverage for a nominal 90% interval, while exposing only 74.77% prior-universe
coverage.

The implemented study includes explicit `pre_counselling` and
`in_round_update` policies, expanding-window evaluation, simple and nonlinear
comparators, OOF uncertainty, subgroup and ablation evidence, a hash-bound
artifact contract, a FastAPI inference layer, a Next.js choice-list interface,
and a dependency-injected Gemini/FAISS advisor. The default architecture is
Python-only; `optional/rust-sim/` is collaborator-contributed and outside the
primary implementation claim.

The immediate mentorship questions are how to formalize the estimand, acquire
and document legitimate source data, pre-declare objectives, construct
uncertainty under distribution shift, and define meaningful choice-list and
fairness evaluations. The strongest next research directions are
temporal-shift conformal prediction, hierarchical cold-start forecasting with
an official seat matrix, and preference-supervised learning-to-rank.
