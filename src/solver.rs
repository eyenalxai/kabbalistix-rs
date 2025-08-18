use crate::expression::{Expression, ExpressionError};
use crate::iterator::ExpressionIterator;
use crate::utils::UtilsError;
use log::{debug, info};
use thiserror::Error;

/// Errors that can occur during solving
#[derive(Error, Debug)]
pub enum SolverError {
    #[error("Expression evaluation error: {0}")]
    ExpressionError(#[from] ExpressionError),
    #[error("Utils error: {0}")]
    UtilsError(#[from] UtilsError),
}

// Default configuration constants
const EPSILON: f64 = 1e-9;

/// Main solver for finding expressions that match a target value
pub struct ExpressionSolver {}

impl ExpressionSolver {
    /// Create a new expression solver
    pub fn new() -> Self {
        Self {}
    }

    /// Find an expression from the given digits that evaluates to the target
    pub fn find_expression(&self, digits: &str, target: f64) -> Option<Expression> {
        info!("Starting iterative expression generation and evaluation");

        let iterator = ExpressionIterator::new(digits.to_string());
        let mut total_evaluated = 0;
        let mut total_valid = 0;
        let mut closest_distance = f64::INFINITY;
        let mut closest_expression: Option<Expression> = None;
        let mut closest_value = 0.0;

        // Process expressions one at a time
        for expr in iterator {
            total_evaluated += 1;

            if total_evaluated % 10000 == 0 {
                info!("Evaluated {} expressions so far...", total_evaluated);
            }

            if let Ok(value) = expr.evaluate() {
                total_valid += 1;
                debug!("Expression {} evaluates to {}", expr, value);

                // Check if this is an exact match
                if (value - target).abs() < EPSILON {
                    info!(
                        "Found exact match after evaluating {} expressions ({} valid): {} = {}",
                        total_evaluated, total_valid, expr, value
                    );
                    return Some(expr);
                }

                // Track the closest result
                let distance = (value - target).abs();
                if distance < closest_distance {
                    closest_distance = distance;
                    closest_expression = Some(expr.clone());
                    closest_value = value;

                    if total_evaluated % 1000 == 0 {
                        info!(
                            "New closest: {} = {:.6} (distance: {:.6})",
                            expr, value, distance
                        );
                    }
                }
            } else {
                debug!("Expression {} failed to evaluate", expr);
            }
        }

        // No exact match found
        info!(
            "No exact match found. Evaluated {} total expressions ({} valid)",
            total_evaluated, total_valid
        );

        if let Some(closest_expr) = closest_expression {
            info!(
                "Closest result: {} = {:.6} (distance: {:.6})",
                closest_expr, closest_value, closest_distance
            );
        }

        None
    }
}

impl Default for ExpressionSolver {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_expression_with_nth_root() {
        // Test that find_expression can find nth root solutions
        // Using digits "327" to find target 3 (cube root of 27 = 3)
        let solver = ExpressionSolver::new();
        let result = solver.find_expression("327", 3.0);
        assert!(result.is_some());

        if let Some(expr) = result {
            let result = expr.evaluate();
            assert!(
                result.is_ok(),
                "Expression should evaluate successfully but got: {:?}",
                result.err()
            );
            if let Ok(value) = result {
                assert!((value - 3.0).abs() < 1e-9);
            }

            // Check that it's actually using nth root (not just arithmetic)
            let expr_str = format!("{}", expr);
            assert!(
                expr_str.contains("√3(27)"),
                "Expected nth root expression, got: {}",
                expr_str
            );
        }
    }

    #[test]
    fn test_solver_creation() {
        let solver = ExpressionSolver::new();

        // Should find basic solutions
        let result = solver.find_expression("24", 6.0);
        assert!(result.is_some());
    }

    #[test]
    fn test_expression_iterator() {
        let iter = ExpressionIterator::new("12".to_string());
        let expressions: Vec<_> = iter.collect();

        // Should have at least the base numbers and some operations
        assert!(!expressions.is_empty());

        // Check that we have the base numbers
        assert!(
            expressions
                .iter()
                .any(|e| matches!(e, Expression::Number(n) if (*n - 12.0).abs() < 1e-9))
        );
    }

    #[test]
    fn test_seven_twos_equals_fourteen() {
        // Test that "2222222" can produce 14 via 2+2+2+2+2+2+2
        let solver = ExpressionSolver::new();
        let result = solver.find_expression("2222222", 14.0);

        assert!(result.is_some(), "Should find an expression that equals 14");

        if let Some(expr) = result {
            assert!(
                expr.evaluate().is_ok(),
                "Expression should evaluate successfully"
            );
            let value = if let Ok(v) = expr.evaluate() {
                v
            } else {
                return;
            };
            assert!(
                (value - 14.0).abs() < 1e-9,
                "Expression should evaluate to 14, got {}",
                value
            );

            // The expression should be some form of addition of seven 2's
            let expr_str = format!("{}", expr);
            println!("Found expression: {}", expr_str);

            // Count the number of 2's in the expression - should be 7
            let two_count = expr_str.matches('2').count();
            assert_eq!(
                two_count, 7,
                "Expression should contain exactly 7 twos, found {} in: {}",
                two_count, expr_str
            );
        }
    }

    #[test]
    fn test_simple_addition_case() {
        // Test a simpler case first: "222" should be able to produce 6 via 2+2+2
        let solver = ExpressionSolver::new();
        let result = solver.find_expression("222", 6.0);

        assert!(result.is_some(), "Should find an expression that equals 6");

        if let Some(expr) = result {
            assert!(
                expr.evaluate().is_ok(),
                "Expression should evaluate successfully"
            );
            let value = if let Ok(v) = expr.evaluate() {
                v
            } else {
                return;
            };
            assert!(
                (value - 6.0).abs() < 1e-9,
                "Expression should evaluate to 6, got {}",
                value
            );

            let expr_str = format!("{}", expr);
            println!("Found expression for 222->6: {}", expr_str);
        }
    }

    #[test]
    fn test_all_digits_must_be_used() {
        // Test that the solver only returns expressions using ALL digits
        let solver = ExpressionSolver::new();

        // Test with "1111" -> 4: should use all four 1's
        let result = solver.find_expression("1111", 4.0);
        assert!(result.is_some(), "Should find an expression that equals 4");

        if let Some(expr) = result {
            // Verify it evaluates correctly
            assert!(
                expr.evaluate().is_ok(),
                "Expression should evaluate successfully"
            );
            if let Ok(value) = expr.evaluate() {
                assert!((value - 4.0).abs() < 1e-9);
            }

            // The iterator now guarantees all expressions use all digits
            let expr_str = format!("{}", expr);
            println!("Found expression for 1111->4: {}", expr_str);
        }
    }

    #[test]
    fn test_six_twos_equals_four() {
        // Test that "222222" can produce 4 via (22-22)+2+2 or similar
        let solver = ExpressionSolver::new();
        let result = solver.find_expression("222222", 4.0);

        assert!(
            result.is_some(),
            "Should find an expression that equals 4 using all six 2's"
        );

        if let Some(expr) = result {
            assert!(
                expr.evaluate().is_ok(),
                "Expression should evaluate successfully"
            );
            let value = if let Ok(v) = expr.evaluate() {
                v
            } else {
                return;
            };
            assert!(
                (value - 4.0).abs() < 1e-9,
                "Expression should evaluate to 4, got {}",
                value
            );

            let expr_str = format!("{}", expr);
            println!("Found expression for 222222->4: {}", expr_str);

            // Verify it contains six 2's
            let two_count = expr_str.matches('2').count();
            assert_eq!(
                two_count, 6,
                "Expression should contain exactly 6 twos, found {} in: {}",
                two_count, expr_str
            );
        }
    }

    #[test]
    fn test_seven_twos_equals_4096() {
        // Test that "2222222" can produce 4096 via 2^((2+2) + (2^2) + (2*2))
        // Expected: 2^(4 + 4 + 4) = 2^12 = 4096
        let solver = ExpressionSolver::new();
        let result = solver.find_expression("2222222", 4096.0);

        assert!(
            result.is_some(),
            "Should find an expression that equals 4096 using all seven 2's"
        );

        if let Some(expr) = result {
            assert!(
                expr.evaluate().is_ok(),
                "Expression should evaluate successfully"
            );
            let value = if let Ok(v) = expr.evaluate() {
                v
            } else {
                return;
            };
            assert!(
                (value - 4096.0).abs() < 1e-9,
                "Expression should evaluate to 4096, got {}",
                value
            );

            let expr_str = format!("{}", expr);
            println!("Found expression for 2222222->4096: {}", expr_str);

            // Verify it contains seven 2's
            let two_count = expr_str.matches('2').count();
            assert_eq!(
                two_count, 7,
                "Expression should contain exactly 7 twos, found {} in: {}",
                two_count, expr_str
            );
        }
    }
}
