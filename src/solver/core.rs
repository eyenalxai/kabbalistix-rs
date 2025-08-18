use log::{debug, info};

use crate::expression::Expression;
use crate::iterator::ExpressionIterator;
use crate::solver::constants::EPSILON;

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

        for expr in iterator {
            total_evaluated += 1;

            if total_evaluated % 10000 == 0 {
                info!("Evaluated {} expressions so far...", total_evaluated);
            }

            if let Ok(value) = expr.evaluate() {
                total_valid += 1;
                debug!("Expression {} evaluates to {}", expr, value);

                if (value - target).abs() < EPSILON {
                    info!(
                        "Found exact match after evaluating {} expressions ({} valid): {} = {}",
                        total_evaluated, total_valid, expr, value
                    );
                    return Some(expr);
                }

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
