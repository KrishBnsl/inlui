//! Seat allotment predictors that do not require a full national candidate pool.
//!
//! - [`cutoff_mc`] — Gaussian Monte Carlo core
//! - [`normalization`] — percentile transform across exam years
//! - [`quota_matrix`] — multi-dimensional quota assets + routing
//! - [`cold_start`] — proxy distributions for new branches
//! - [`engine`] — wires everything for production calls

pub mod cold_start;
pub mod cutoff_mc;
pub mod db;
pub mod engine;
pub mod normalization;
pub mod quota_matrix;
