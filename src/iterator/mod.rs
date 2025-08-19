//! Iterator module
//!
//! Provides a streaming API to generate all valid `Expression` values that
//! use all digits from the input in increasing structural complexity. This is
//! useful for exhaustive enumeration (e.g., exploring the search space),
//! benchmarking, or building custom filters beyond a single target value.
//!
//! For finding a single expression equal to a target, prefer `solver::ExpressionSolver`.

pub mod constants;
pub mod core;
pub mod generator;
pub mod state;
pub mod types;

pub use core::ExpressionIterator;

#[cfg(test)]
mod tests;

/// Convenience constructor for creating an `ExpressionIterator` from `&str`.
pub fn iter_expressions(digits: &str) -> core::ExpressionIterator {
    core::ExpressionIterator::from_digits(digits)
}
