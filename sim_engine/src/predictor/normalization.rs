//! Rank inflation guard: fit and simulate on exam percentiles, then map back to ranks.

use std::collections::HashMap;

use super::cutoff_mc::{CutoffDistribution, DistributionError};

/// Total applicants for a given exam year (e.g. JEE Mains 2024 registered candidates).
#[derive(Debug, Clone, Copy)]
pub struct ExamCohort {
    pub year: u16,
    pub total_candidates: u32,
}

#[derive(Debug, Clone, Default)]
pub struct CohortRegistry {
    by_year: HashMap<u16, u32>,
}

impl CohortRegistry {
    pub fn new(cohorts: impl IntoIterator<Item = ExamCohort>) -> Self {
        let mut by_year = HashMap::new();
        for c in cohorts {
            by_year.insert(c.year, c.total_candidates);
        }
        Self { by_year }
    }

    pub fn insert(&mut self, cohort: ExamCohort) {
        self.by_year.insert(cohort.year, cohort.total_candidates);
    }

    pub fn total_for_year(&self, year: u16) -> Option<u32> {
        self.by_year.get(&year).copied()
    }

    /// Closing rank → percentile (0–100). Lower percentile = more competitive rank.
    pub fn rank_to_percentile(&self, year: u16, rank: u32) -> Result<f64, DistributionError> {
        let total = self
            .total_for_year(year)
            .ok_or(DistributionError::MissingCohort(year))?;
        Ok(rank_to_percentile(rank, total))
    }

    /// Percentile → projected rank for a target year.
    pub fn percentile_to_rank(&self, year: u16, percentile: f64) -> Result<u32, DistributionError> {
        let total = self
            .total_for_year(year)
            .ok_or(DistributionError::MissingCohort(year))?;
        Ok(percentile_to_rank(percentile, total))
    }
}

/// `percentile = 100 * rank / total` — comparable across exam years.
pub fn rank_to_percentile(rank: u32, total_candidates: u32) -> f64 {
    if total_candidates == 0 {
        return 0.0;
    }
    (rank as f64 / total_candidates as f64) * 100.0
}

pub fn percentile_to_rank(percentile: f64, total_candidates: u32) -> u32 {
    if total_candidates == 0 {
        return 1;
    }
    let rank = (percentile / 100.0) * total_candidates as f64;
    rank.round().clamp(1.0, total_candidates as f64) as u32
}

/// Convert yearly closing ranks into percentiles using each year's cohort size.
pub fn yearly_ranks_to_percentiles(
    yearly_cutoffs: &[(u16, u32)],
    cohorts: &CohortRegistry,
) -> Result<Vec<f64>, DistributionError> {
    yearly_cutoffs
        .iter()
        .map(|(year, rank)| cohorts.rank_to_percentile(*year, *rank))
        .collect()
}

/// Fit volatility model in percentile space (reduces cross-year inflation noise),
/// applying an exponentially weighted moving average to prioritize recent years.
pub fn fit_percentile_distribution(
    yearly_cutoffs: &[(u16, u32)],
    cohorts: &CohortRegistry,
) -> Result<CutoffDistribution, DistributionError> {
    let mut year_percentiles = Vec::with_capacity(yearly_cutoffs.len());
    for &(year, rank) in yearly_cutoffs {
        let pct = cohorts.rank_to_percentile(year, rank)?;
        year_percentiles.push((year, pct));
    }
    fit_cutoff_distribution_weighted(&year_percentiles)
}

/// Same success rule as rank space: user wins when `user <= cutoff`.
/// Uses an exponentially weighted moving average (decay factor = 0.6) to give priority to recent years.
pub fn fit_cutoff_distribution_weighted(
    values: &[(u16, f64)],
) -> Result<CutoffDistribution, DistributionError> {
    if values.is_empty() {
        return Err(DistributionError::NotEnoughHistory);
    }

    let max_year = values.iter().map(|(y, _)| *y).max().unwrap_or(2025);
    let decay_factor = 0.6_f64; // Gives 60% weight to year N, 36% to N-1, etc.

    let mut weighted_sum = 0.0;
    let mut v1 = 0.0;
    let mut v2 = 0.0;

    let mut weights = Vec::with_capacity(values.len());

    for (year, val) in values {
        // diff is how many years old the data is
        let diff = max_year.saturating_sub(*year);
        let weight = f64::powf(decay_factor, diff as f64);
        weights.push(weight);

        weighted_sum += val * weight;
        v1 += weight;
        v2 += weight * weight;
    }

    let mean = weighted_sum / v1;

    let std_dev = if values.len() < 2 {
        0.0
    } else {
        let mut var_sum = 0.0;
        for (i, w) in weights.iter().enumerate() {
            let diff = values[i].1 - mean;
            var_sum += w * diff * diff;
        }

        // Reliability-weighted unbiased variance (Bessel's correction equivalent)
        let variance = if (v1 * v1 - v2) > 0.0 {
            (v1 / (v1 * v1 - v2)) * var_sum
        } else {
            0.0
        };
        variance.sqrt()
    };

    if values.len() >= 2 && std_dev <= 0.0 {
        return Err(DistributionError::InvalidStdDev);
    }

    Ok(CutoffDistribution { mean, std_dev })
}

/// Unweighted version for simulated percentile outcomes.
pub fn fit_cutoff_distribution_f64(
    values: &[f64],
) -> Result<CutoffDistribution, DistributionError> {
    if values.is_empty() {
        return Err(DistributionError::NotEnoughHistory);
    }

    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;

    let std_dev = if values.len() < 2 {
        0.0
    } else {
        let variance = values
            .iter()
            .map(|v| {
                let d = v - mean;
                d * d
            })
            .sum::<f64>()
            / (n - 1.0);
        variance.sqrt()
    };

    if values.len() >= 2 && std_dev <= 0.0 {
        return Err(DistributionError::InvalidStdDev);
    }

    Ok(CutoffDistribution { mean, std_dev })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentile_is_monotonic_with_rank() {
        let p1 = rank_to_percentile(1_000, 1_000_000);
        let p2 = rank_to_percentile(10_000, 1_000_000);
        assert!(p1 < p2);
    }

    #[test]
    fn round_trip_rank_through_percentile() {
        let total = 500_000;
        let rank = 15_000;
        let pct = rank_to_percentile(rank, total);
        let back = percentile_to_rank(pct, total);
        assert!((back as i32 - rank as i32).abs() <= 1);
    }
}
