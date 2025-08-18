use crate::expression::{Expression, ExpressionError};
use crate::utils::{UtilsError, digits_to_number, generate_partitions};
use log::{debug, info};
use rayon::prelude::*;
use std::sync::{Arc, Mutex};
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
const MAX_ROOT_DEGREE: f64 = 10.0;
const EPSILON: f64 = 1e-9;

/// Main solver for finding expressions that match a target value
pub struct ExpressionSolver;

impl ExpressionSolver {
    /// Create a new expression solver
    pub fn new() -> Self {
        Self
    }

    /// Find an expression from the given digits that evaluates to the target
    pub fn find_expression(&self, digits: &str, target: f64) -> Option<Expression> {
        let all_expressions = self.generate_expressions(digits, 0, digits.len());
        let total_expressions = all_expressions.len();

        debug!("Generated {} expressions to evaluate", total_expressions);

        // Use Arc<Mutex<_>> to track counts across threads
        let evaluated_count = Arc::new(Mutex::new(0usize));
        let valid_count = Arc::new(Mutex::new(0usize));

        // Use parallel iterator to find matching expression
        let result = all_expressions.into_par_iter().find_map_any(|expr| {
            // Update evaluated count
            {
                let mut count = evaluated_count.lock().ok()?;
                *count += 1;
            }

            if let Ok(value) = expr.evaluate() {
                // Update valid count
                {
                    let mut count = valid_count.lock().ok()?;
                    *count += 1;
                }

                debug!("Expression {} evaluates to {}", expr, value);
                if (value - target).abs() < EPSILON {
                    return Some(expr);
                }
            } else {
                debug!("Expression {} failed to evaluate", expr);
            }
            None
        });

        // Get final counts for logging
        let final_evaluated = evaluated_count.lock().map(|c| *c).unwrap_or(0);
        let final_valid = valid_count.lock().map(|c| *c).unwrap_or(0);

        match result {
            Some(expr) => {
                info!(
                    "Found match after evaluating {} expressions ({} valid)",
                    final_evaluated, final_valid
                );
                Some(expr)
            }
            None => {
                info!(
                    "No match found. Evaluated {} expressions ({} valid)",
                    final_evaluated, final_valid
                );
                None
            }
        }
    }

    /// Generate all possible expressions from a digit string
    fn generate_expressions_recursive(
        &self,
        digits: &str,
        start: usize,
        end: usize,
    ) -> Vec<Expression> {
        if start >= end || start >= digits.len() || end > digits.len() {
            return Vec::new();
        }

        let length = end - start;
        let mut expressions = Vec::new();

        // Base case: single number (always include this)
        if let Ok(num) = digits_to_number(digits, start, end) {
            expressions.push(Expression::Number(num));
        }

        if length >= 2 {
            self.add_binary_operations(digits, start, end, &mut expressions);
            self.add_nth_root_operations(digits, start, end, &mut expressions);
            self.add_negation_operations(&mut expressions);
        }

        expressions
    }

    /// Add binary operations (add, sub, mul, div, pow) to expressions
    fn add_binary_operations(
        &self,
        digits: &str,
        start: usize,
        end: usize,
        expressions: &mut Vec<Expression>,
    ) {
        let partitions = generate_partitions(start, end, 2);

        // Process all partitions in parallel
        let all_new_expressions: Vec<Expression> = partitions
            .par_iter()
            .filter_map(|partition| {
                if let (Some(&(start1, end1)), Some(&(start2, end2))) =
                    (partition.first(), partition.get(1))
                {
                    let left_exprs = self.generate_expressions_recursive(digits, start1, end1);
                    let right_exprs = self.generate_expressions_recursive(digits, start2, end2);

                    // Use parallel processing for the cartesian product of expressions
                    let partition_expressions: Vec<Expression> = left_exprs
                        .par_iter()
                        .flat_map(|left| {
                            right_exprs.par_iter().flat_map(move |right| {
                                [
                                    Expression::Add(
                                        Box::new(left.clone()),
                                        Box::new(right.clone()),
                                    ),
                                    Expression::Sub(
                                        Box::new(left.clone()),
                                        Box::new(right.clone()),
                                    ),
                                    Expression::Mul(
                                        Box::new(left.clone()),
                                        Box::new(right.clone()),
                                    ),
                                    Expression::Div(
                                        Box::new(left.clone()),
                                        Box::new(right.clone()),
                                    ),
                                    Expression::Pow(
                                        Box::new(left.clone()),
                                        Box::new(right.clone()),
                                    ),
                                ]
                            })
                        })
                        .collect();

                    Some(partition_expressions)
                } else {
                    None
                }
            })
            .flatten()
            .collect();

        expressions.extend(all_new_expressions);
    }

    /// Add nth root operations to expressions
    fn add_nth_root_operations(
        &self,
        digits: &str,
        start: usize,
        end: usize,
        expressions: &mut Vec<Expression>,
    ) {
        let partitions = generate_partitions(start, end, 2);

        // Process all partitions in parallel
        let all_root_expressions: Vec<Expression> = partitions
            .par_iter()
            .filter_map(|partition| {
                if let (Some(&(start1, end1)), Some(&(start2, end2))) =
                    (partition.first(), partition.get(1))
                {
                    // First block must form an integer >= 2 for the root index
                    if let Ok(n_num) = digits_to_number(digits, start1, end1) {
                        if n_num >= 2.0 && n_num.fract() == 0.0 && n_num <= MAX_ROOT_DEGREE {
                            let n_expr = Expression::Number(n_num);
                            let a_exprs = self.generate_expressions_recursive(digits, start2, end2);

                            // Create nth root expressions in parallel
                            let partition_roots: Vec<Expression> = a_exprs
                                .par_iter()
                                .map(|a_expr| {
                                    Expression::NthRoot(
                                        Box::new(n_expr.clone()),
                                        Box::new(a_expr.clone()),
                                    )
                                })
                                .collect();

                            Some(partition_roots)
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                } else {
                    None
                }
            })
            .flatten()
            .collect();

        expressions.extend(all_root_expressions);
    }

    /// Add unary negation operations to expressions
    fn add_negation_operations(&self, expressions: &mut Vec<Expression>) {
        // Unary negation (only for composite expressions to avoid redundancy)
        let composite_expressions: Vec<_> = expressions.iter().skip(1).cloned().collect();
        for expr in composite_expressions {
            expressions.push(Expression::Neg(Box::new(expr)));
        }
    }

    /// Generate all possible expressions from a digit string
    fn generate_expressions(&self, digits: &str, start: usize, end: usize) -> Vec<Expression> {
        self.generate_expressions_recursive(digits, start, end)
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
}
