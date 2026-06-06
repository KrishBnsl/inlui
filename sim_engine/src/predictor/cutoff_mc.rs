//! Market-volatility Monte Carlo: model each branch's closing rank as a distribution
//! learned from historical rounds, then estimate P(user gets seat) across simulated years.

use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};
use rayon::prelude::*;

/// Historical closing ranks for one programme (e.g. DTU CSE, Delhi GEN, round 6).
#[derive(Debug, Clone)]
pub struct BranchCutoffHistory {
    pub institute: String,
    pub branch: String,
    /// `(year, closing_rank)` — lower rank number means more competitive seat.
    pub yearly_cutoffs: Vec<(u16, u32)>,
}

/// Normal model of cutoff volatility: closing rank ~ N(μ, σ²).
#[derive(Debug, Clone, Copy)]
pub struct CutoffDistribution {
    pub mean: f64,
    pub std_dev: f64,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct SeatCapacityAdjustment {
    /// Shift applied to μ before simulation (positive = seats added → cutoff rank worsens / increases).
    pub mean_shift: f64,
}

#[derive(Debug, Clone)]
pub struct BranchPrediction {
    pub institute: String,
    pub branch: String,
    pub distribution: CutoffDistribution,
    /// Estimated P(user rank clears simulated cutoff), in [0, 100].
    pub probability_percent: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistributionError {
    NotEnoughHistory,
    InvalidStdDev,
    MissingCohort(u16),
    NoEligibleAssets,
    NoHistoryOrProxy,
    ProxyAssetNotFound,
    ProxyHasNoHistory,
}

impl std::fmt::Display for DistributionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotEnoughHistory => write!(f, "need at least one historical cutoff"),
            Self::InvalidStdDev => write!(f, "standard deviation must be positive"),
            Self::MissingCohort(year) => {
                write!(f, "missing exam cohort size for year {year}")
            }
            Self::NoEligibleAssets => write!(f, "no quota assets match the user profile"),
            Self::NoHistoryOrProxy => write!(f, "asset has no history and no cold-start proxy"),
            Self::ProxyAssetNotFound => write!(f, "cold-start proxy points to unknown asset"),
            Self::ProxyHasNoHistory => write!(f, "proxy asset has no historical cutoffs"),
        }
    }
}

impl std::error::Error for DistributionError {}

/// Fit μ and σ from historical closing ranks (sample std-dev when n ≥ 2).
pub fn fit_cutoff_distribution(cutoffs: &[u32]) -> Result<CutoffDistribution, DistributionError> {
    if cutoffs.is_empty() {
        return Err(DistributionError::NotEnoughHistory);
    }
    let values: Vec<f64> = cutoffs.iter().map(|&r| r as f64).collect();
    super::normalization::fit_cutoff_distribution_f64(&values)
}

impl BranchCutoffHistory {
    pub fn distribution(&self) -> Result<CutoffDistribution, DistributionError> {
        let cutoffs: Vec<u32> = self.yearly_cutoffs.iter().map(|(_, r)| *r).collect();
        fit_cutoff_distribution(&cutoffs)
    }
}

/// Apply a seat-matrix change (e.g. +20 seats) as a shift to the mean cutoff before simulating.
pub fn adjusted_distribution(
    base: CutoffDistribution,
    adjustment: SeatCapacityAdjustment,
) -> CutoffDistribution {
    CutoffDistribution {
        mean: base.mean + adjustment.mean_shift,
        std_dev: base.std_dev,
    }
}

/// Monte Carlo estimate of allotment probability for one branch.
///
/// Each iteration draws a simulated closing rank from `Normal(mean, std_dev)`.
/// Success when `user_rank <= simulated_cutoff` (lower rank is better).
pub fn simulate_branch_probability(
    user_rank: u32,
    distribution: CutoffDistribution,
    iterations: usize,
) -> Result<f64, DistributionError> {
    simulate_branch_probability_with_adjustment(user_rank, distribution, SeatCapacityAdjustment::default(), iterations)
}

pub fn simulate_branch_probability_with_adjustment(
    user_rank: u32,
    distribution: CutoffDistribution,
    adjustment: SeatCapacityAdjustment,
    iterations: usize,
) -> Result<f64, DistributionError> {
    let dist = adjusted_distribution(distribution, adjustment);
    run_monte_carlo_on_values(user_rank as f64, dist, iterations)
}

/// Monte Carlo on any comparable scale (raw rank or normalized percentile).
///
/// Success when `user_value <= simulated_cutoff` (lower rank / lower percentile is better).
pub fn run_monte_carlo_on_values(
    user_value: f64,
    distribution: CutoffDistribution,
    iterations: usize,
) -> Result<f64, DistributionError> {
    if iterations == 0 {
        return Ok(0.0);
    }

    if distribution.std_dev <= 0.0 {
        let success = user_value <= distribution.mean;
        return Ok(if success { 100.0 } else { 0.0 });
    }

    let normal = Normal::new(distribution.mean, distribution.std_dev)
        .map_err(|_| DistributionError::InvalidStdDev)?;

    let successes: usize = (0..iterations)
        .into_par_iter()
        .map_init(StdRng::from_os_rng, |rng, _| {
            let simulated_cutoff = normal.sample(rng);
            usize::from(user_value <= simulated_cutoff)
        })
        .sum();

    Ok(successes as f64 / iterations as f64 * 100.0)
}

/// Predict several branches for one student rank in parallel.
pub fn predict_branches(
    user_rank: u32,
    histories: &[BranchCutoffHistory],
    iterations: usize,
) -> Result<Vec<BranchPrediction>, DistributionError> {
    histories
        .iter()
        .map(|history| {
            let distribution = history.distribution()?;
            let probability_percent =
                simulate_branch_probability(user_rank, distribution, iterations)?;
            Ok(BranchPrediction {
                institute: history.institute.clone(),
                branch: history.branch.clone(),
                distribution,
                probability_percent,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dtu_cse_history() -> BranchCutoffHistory {
        BranchCutoffHistory {
            institute: "DTU".into(),
            branch: "CSE".into(),
            yearly_cutoffs: vec![
                (2022, 4_800),
                (2023, 5_100),
                (2024, 4_950),
                (2025, 5_300),
            ],
        }
    }

    #[test]
    fn fits_mean_close_to_example() {
        let dist = dtu_cse_history().distribution().unwrap();
        assert!((dist.mean - 5_037.5).abs() < 0.1);
        assert!((dist.std_dev - 213.0).abs() < 5.0);
    }

    #[test]
    fn much_better_rank_almost_always_succeeds() {
        let dist = dtu_cse_history().distribution().unwrap();
        let p = simulate_branch_probability(3_000, dist, 20_000).unwrap();
        assert!(p > 99.0);
    }

    #[test]
    fn rank_near_mean_has_fractional_probability() {
        let dist = dtu_cse_history().distribution().unwrap();
        let p = simulate_branch_probability(5_100, dist, 50_000).unwrap();
        assert!(p > 20.0 && p < 60.0, "expected middling probability, got {p}");
    }

    #[test]
    fn seat_increase_shifts_probability_down_when_cutoff_worsens() {
        let dist = dtu_cse_history().distribution().unwrap();
        let user_rank = 5_100;
        let base = simulate_branch_probability(user_rank, dist, 30_000).unwrap();
        let harder = simulate_branch_probability_with_adjustment(
            user_rank,
            dist,
            SeatCapacityAdjustment { mean_shift: -500.0 },
            30_000,
        )
        .unwrap();
        assert!(harder < base);
    }
}
