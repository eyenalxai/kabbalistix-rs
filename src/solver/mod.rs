//! Solver module split into submodules

mod constants;
mod core;
mod errors;

pub use core::ExpressionSolver;
pub use errors::SolverError;

#[cfg(test)]
mod tests;
