//! Nearest-neighbor proxy when a branch has no historical cutoffs.

use super::cutoff_mc::CutoffDistribution;
use super::normalization;
use super::normalization::CohortRegistry;
use super::quota_matrix::QuotaAssetKey;

/// Link a new branch to a comparable existing asset's volatility model.
#[derive(Debug, Clone)]
pub struct ColdStartProxy {
    pub proxy_key: QuotaAssetKey,
    /// Added to proxy mean (percentile space): positive = easier cutoff / worse competition.
    pub percentile_mean_offset: f64,
    /// Scales proxy σ (e.g. 1.2 for more uncertainty on a new programme).
    pub percentile_std_scale: f64,
}

impl Default for ColdStartProxy {
    fn default() -> Self {
        Self {
            proxy_key: QuotaAssetKey {
                institute: String::new(),
                branch: String::new(),
                quota: crate::enumerations::enums::Quota::OtherState,
                category: crate::enumerations::enums::Category::General,
                counselling: crate::enumerations::enums::Counselling::JoSAA,
            },
            percentile_mean_offset: 0.0,
            percentile_std_scale: 1.0,
        }
    }
}

/// Derive a percentile distribution from a proxy asset's history.
pub fn distribution_from_proxy(
    proxy_yearly_cutoffs: &[(u16, u32)],
    cohorts: &CohortRegistry,
    proxy: &ColdStartProxy,
) -> Result<CutoffDistribution, super::cutoff_mc::DistributionError> {
    let base = normalization::fit_percentile_distribution(proxy_yearly_cutoffs, cohorts)?;
    Ok(CutoffDistribution {
        mean: base.mean + proxy.percentile_mean_offset,
        std_dev: base.std_dev * proxy.percentile_std_scale,
    })
}

/// Convenience builder: new branch leans toward a more competitive proxy (e.g. CSE).
pub fn toward_competitive_proxy(
    proxy_key: QuotaAssetKey,
    weight_toward_upper: f64,
) -> ColdStartProxy {
    let w = weight_toward_upper.clamp(0.0, 1.0);
    ColdStartProxy {
        proxy_key,
        percentile_mean_offset: (1.0 - w) * 0.15,
        percentile_std_scale: 1.15,
    }
}
