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
    validate_digit_string(digits)?;

    let solver = ExpressionSolver::new();
    Ok(solver.find_expression(digits, target))
}
