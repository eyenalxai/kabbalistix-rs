pub mod constants;
mod core;
mod errors;

pub use core::ExpressionSolver;
pub use errors::SolverError;

#[cfg(test)]
mod tests;
