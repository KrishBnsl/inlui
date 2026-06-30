//! Production-facing orchestrator: route → normalize → (cold start) → Monte Carlo.

use std::collections::HashMap;

use super::cold_start::{ColdStartProxy, distribution_from_proxy};
use super::cutoff_mc::{
    CutoffDistribution, DistributionError, SeatCapacityAdjustment, adjusted_distribution,
    run_monte_carlo_on_values,
};
use super::normalization::{self, CohortRegistry};
use super::quota_matrix::{QuotaAsset, QuotaAssetKey, QuotaMatrix, UserProfile};

#[derive(Debug, Clone)]
pub struct PredictorEngineConfig {
    pub cohorts: CohortRegistry,
    /// Year whose rank scale is used for the student's input and projected outputs.
    pub target_year: u16,
    pub target_year_total: u32,
    pub iterations: usize,
    pub seat_adjustment: SeatCapacityAdjustment,
}

#[derive(Debug, Clone)]
pub struct PredictorEngine {
    pub matrix: QuotaMatrix,
    pub config: PredictorEngineConfig,
    /// Cold-start proxies keyed by the *new* asset they apply to.
    pub cold_start: HashMap<QuotaAssetKey, ColdStartProxy>,
}

#[derive(Debug, Clone)]
pub struct QuotaAssetPrediction {
    pub key: QuotaAssetKey,
    pub distribution: CutoffDistribution,
    /// Distribution was simulated in percentile space (normalized).
    pub uses_percentile_model: bool,
    /// True when a cold-start proxy supplied the distribution.
    pub uses_cold_start_proxy: bool,
    pub probability_percent: f64,
    /// Median projected closing rank on the target year's scale.
    pub projected_cutoff_rank: u32,
}

impl PredictorEngine {
    pub async fn from_db(
        pool: &sqlx::PgPool,
        target_year: u16,
        target_year_total: u32,
        cohorts: CohortRegistry,
        cold_start: HashMap<QuotaAssetKey, ColdStartProxy>,
    ) -> Result<Self, sqlx::Error> {
        let rows = crate::predictor::db::fetch_cutoffs_for_simulation(pool).await?;
        Ok(Self::from_rows(
            rows,
            target_year,
            target_year_total,
            cohorts,
            cold_start,
        ))
    }

    /// Build an engine from pre-fetched DB rows (avoids a second round-trip when two engines share the same data).
    pub fn from_rows(
        rows: Vec<crate::predictor::db::HistoricalCutoff>,
        target_year: u16,
        target_year_total: u32,
        cohorts: CohortRegistry,
        cold_start: HashMap<QuotaAssetKey, ColdStartProxy>,
    ) -> Self {
        let matrix = QuotaMatrix::from_db_rows(rows);
        Self {
            matrix,
            config: PredictorEngineConfig {
                cohorts,
                target_year,
                target_year_total,
                iterations: 10_000,
                seat_adjustment: SeatCapacityAdjustment::default(),
            },
            cold_start,
        }
    }

    pub fn predict_for_user(
        &self,
        user: &UserProfile,
        user_rank: u32,
    ) -> Result<Vec<QuotaAssetPrediction>, DistributionError> {
        let routed = self.matrix.route(user);
        if routed.is_empty() {
            return Err(DistributionError::NoEligibleAssets);
        }

        let user_percentile =
            normalization::rank_to_percentile(user_rank, self.config.target_year_total);

        let predictions: Vec<QuotaAssetPrediction> = routed
            .iter()
            .filter_map(|asset| match self.predict_asset(asset, user_percentile) {
                Ok(p) => Some(p),
                Err(e) => {
                    tracing::debug!("Failed to predict asset {:?}: {:?}", asset.key, e);
                    None
                }
            })
            .collect();

        if predictions.is_empty() {
            Err(DistributionError::NoEligibleAssets)
        } else {
            Ok(predictions)
        }
    }

    fn predict_asset(
        &self,
        asset: &QuotaAsset,
        user_percentile: f64,
    ) -> Result<QuotaAssetPrediction, DistributionError> {
        let (distribution, uses_percentile_model, uses_cold_start_proxy) =
            self.resolve_distribution(asset)?;

        let dist = adjusted_distribution(distribution, self.config.seat_adjustment);
        let probability_percent =
            run_monte_carlo_on_values(user_percentile, dist, self.config.iterations)?;

        let projected_cutoff_rank =
            normalization::percentile_to_rank(dist.mean, self.config.target_year_total);

        Ok(QuotaAssetPrediction {
            key: asset.key.clone(),
            distribution: dist,
            uses_percentile_model,
            uses_cold_start_proxy,
            probability_percent,
            projected_cutoff_rank,
        })
    }

    fn resolve_distribution(
        &self,
        asset: &QuotaAsset,
    ) -> Result<(CutoffDistribution, bool, bool), DistributionError> {
        if !asset.yearly_cutoffs.is_empty() {
            let dist = normalization::fit_percentile_distribution(
                &asset.yearly_cutoffs,
                &self.config.cohorts,
            )?;
            return Ok((dist, true, false));
        }

        let proxy = self
            .cold_start
            .get(&asset.key)
            .ok_or(DistributionError::NoHistoryOrProxy)?;

        let proxy_asset = self
            .matrix
            .find_by_key(&proxy.proxy_key)
            .ok_or(DistributionError::ProxyAssetNotFound)?;

        if proxy_asset.yearly_cutoffs.is_empty() {
            return Err(DistributionError::ProxyHasNoHistory);
        }

        let dist =
            distribution_from_proxy(&proxy_asset.yearly_cutoffs, &self.config.cohorts, proxy)?;
        Ok((dist, true, true))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::enumerations::enums::{Category, Counselling, Quota};
    use crate::predictor::normalization::ExamCohort;

    fn cohorts() -> CohortRegistry {
        CohortRegistry::new([
            ExamCohort {
                year: 2022,
                total_candidates: 900_000,
            },
            ExamCohort {
                year: 2023,
                total_candidates: 950_000,
            },
            ExamCohort {
                year: 2024,
                total_candidates: 1_000_000,
            },
            ExamCohort {
                year: 2025,
                total_candidates: 1_050_000,
            },
            ExamCohort {
                year: 2026,
                total_candidates: 1_100_000,
            },
        ])
    }

    fn dtu_cse_gen_delhi() -> QuotaAsset {
        QuotaAsset {
            key: QuotaAssetKey {
                institute: "DTU".into(),
                branch: "CSE".into(),
                quota: Quota::HomeState,
                category: Category::General,
                counselling: Counselling::JacDelhi,
            },
            institute_home_state: Some(crate::enumerations::enums::IndianState::Delhi),
            required_horizontal: Default::default(),
            yearly_cutoffs: vec![(2022, 4_800), (2023, 5_100), (2024, 4_950), (2025, 5_300)],
        }
    }

    fn engine_with(asset: QuotaAsset) -> PredictorEngine {
        PredictorEngine {
            matrix: QuotaMatrix {
                assets: vec![asset],
            },
            config: PredictorEngineConfig {
                cohorts: cohorts(),
                target_year: 2026,
                target_year_total: 1_100_000,
                iterations: 20_000,
                seat_adjustment: SeatCapacityAdjustment::default(),
            },
            cold_start: HashMap::new(),
        }
    }

    #[test]
    fn engine_routes_and_returns_prediction() {
        let engine = engine_with(dtu_cse_gen_delhi());
        let user = UserProfile {
            category: Category::General,
            quota: Quota::HomeState,
            homestate: crate::enumerations::enums::IndianState::Delhi,
            horizontal: Default::default(),
        };
        let preds = engine.predict_for_user(&user, 5_100).unwrap();
        assert_eq!(preds.len(), 1);
        assert!(preds[0].uses_percentile_model);
        assert!(!preds[0].uses_cold_start_proxy);
        assert!(preds[0].probability_percent > 10.0);
    }

    #[test]
    fn cold_start_uses_proxy_asset() {
        let proxy = dtu_cse_gen_delhi();
        let proxy_key = proxy.key.clone();
        let mut new_ai = QuotaAsset {
            key: QuotaAssetKey {
                branch: "AI & DS".into(),
                ..proxy.key.clone()
            },
            yearly_cutoffs: vec![],
            ..proxy
        };
        new_ai.key.branch = "AI & DS".into();

        let mut engine = engine_with(proxy);
        engine.matrix.assets.push(new_ai.clone());
        engine.cold_start.insert(
            new_ai.key.clone(),
            ColdStartProxy {
                proxy_key,
                percentile_mean_offset: 0.2,
                percentile_std_scale: 1.1,
            },
        );

        let user = UserProfile {
            category: Category::General,
            quota: Quota::HomeState,
            homestate: crate::enumerations::enums::IndianState::Delhi,
            horizontal: Default::default(),
        };
        let preds = engine.predict_for_user(&user, 5_100).unwrap();
        assert_eq!(preds.len(), 2);
        assert!(preds.iter().any(|p| p.uses_cold_start_proxy));
    }
}
