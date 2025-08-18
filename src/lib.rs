#![deny(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::unwrap_or_default,
    clippy::get_unwrap,
    clippy::map_unwrap_or,
    clippy::unnecessary_unwrap,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::todo,
    clippy::unimplemented,
    clippy::unreachable,
    clippy::exit,
    clippy::mem_forget,
    clippy::clone_on_ref_ptr,
    clippy::mutex_atomic,
    clippy::rc_mutex
)]

//! Kabbalistix - A library for finding mathematical expressions from digit strings
//!
//! This library provides functionality to generate and evaluate mathematical expressions
//! that can be constructed from a given string of digits to match a target value.

pub mod expression;
pub mod solver;
pub mod utils;

// Re-export the main public API
pub use expression::{Expression, ExpressionError};
pub use solver::{ExpressionSolver, SolverError};
pub use utils::{UtilsError, validate_digit_string};

/// Find an expression from the given digits that evaluates to the target value
///
/// This is a convenience function that creates a default solver and attempts to find
/// a matching expression.
///
/// # Arguments
///
/// * `digits` - A string containing only ASCII digits
/// * `target` - The target value to match
///
/// # Returns
///
/// * `Ok(Some(Expression))` - If a matching expression is found
/// * `Ok(None)` - If no matching expression is found
/// * `Err(SolverError)` - If there's an error in the input or solving process
///
/// # Errors
///
/// This function will return an error if:
/// * The input digit string is empty
/// * The input digit string contains non-digit characters
/// * There's an internal error during expression generation or evaluation
///
/// # Examples
///
/// ```
/// use kabbalistix::find_expression;
///
/// // Find an expression using digits "123" that equals 6
/// match find_expression("123", 6.0) {
///     Ok(Some(expr)) => println!("Found: {}", expr),
///     Ok(None) => println!("No solution found"),
///     Err(e) => println!("Error: {}", e),
/// }
/// ```
pub fn find_expression(digits: &str, target: f64) -> Result<Option<Expression>, SolverError> {
    // Validate the input
    validate_digit_string(digits)?;

    // Create a solver and find the expression
    let solver = ExpressionSolver::new();
    Ok(solver.find_expression(digits, target))
}
