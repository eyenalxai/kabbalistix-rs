use crate::expression::{Expression, ExpressionError};
use crate::utils::{UtilsError, digits_to_number, generate_partitions};
use std::collections::HashMap;
use thiserror::Error;

/// Errors that can occur during solving
#[derive(Error, Debug)]
pub enum SolverError {
    #[error("Expression evaluation error: {0}")]
    ExpressionError(#[from] ExpressionError),
    #[error("Utils error: {0}")]
    UtilsError(#[from] UtilsError),
}

/// Memoization cache for expressions
type ExprCache = HashMap<(usize, usize), Vec<Expression>>;

/// Configuration for expression generation
pub struct SolverConfig {
    pub max_root_degree: f64,
    pub epsilon: f64,
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            max_root_degree: 10.0,
            epsilon: 1e-9,
        }
    }
}

/// Main solver for finding expressions that match a target value
pub struct ExpressionSolver {
    config: SolverConfig,
}

impl ExpressionSolver {
    pub fn new(config: SolverConfig) -> Self {
        Self { config }
    }

    /// Get a reference to the solver configuration
    pub fn config(&self) -> &SolverConfig {
        &self.config
    }

    /// Find an expression from the given digits that evaluates to the target
    pub fn find_expression(&self, digits: &str, target: f64) -> Option<Expression> {
        self.find_expression_with_verbose(digits, target, false)
    }

    /// Find an expression from the given digits that evaluates to the target with optional verbose output
    pub fn find_expression_with_verbose(
        &self,
        digits: &str,
        target: f64,
        verbose: bool,
    ) -> Option<Expression> {
        let all_expressions = self.generate_expressions(digits, 0, digits.len());
        let total_expressions = all_expressions.len();

        if verbose {
            eprintln!("Generated {} expressions to evaluate", total_expressions);
        }

        let mut evaluated_count = 0;
        let mut valid_count = 0;

        for expr in all_expressions {
            evaluated_count += 1;
            if let Ok(value) = expr.evaluate() {
                valid_count += 1;
                if (value - target).abs() < self.config.epsilon {
                    if verbose {
                        eprintln!(
                            "Found match after evaluating {} expressions ({} valid)",
                            evaluated_count, valid_count
                        );
                    }
                    return Some(expr);
                }
            }
        }

        if verbose {
            eprintln!(
                "No match found. Evaluated {} expressions ({} valid)",
                evaluated_count, valid_count
            );
        }
        None
    }

    /// Generate all possible expressions from a digit string with memoization
    fn generate_expressions_memo(
        &self,
        digits: &str,
        start: usize,
        end: usize,
        cache: &mut ExprCache,
    ) -> Vec<Expression> {
        if start >= end || start >= digits.len() || end > digits.len() {
            return Vec::new();
        }

        // Check cache first
        if let Some(cached) = cache.get(&(start, end)) {
            return cached.clone();
        }

        let length = end - start;
        let mut expressions = Vec::new();

        // Base case: single number (always include this)
        if let Ok(num) = digits_to_number(digits, start, end) {
            expressions.push(Expression::Number(num));
        }

        if length >= 2 {
            self.add_binary_operations(digits, start, end, cache, &mut expressions);
            self.add_nth_root_operations(digits, start, end, cache, &mut expressions);
            self.add_negation_operations(&mut expressions);
        }

        // Cache the result
        cache.insert((start, end), expressions.clone());
        expressions
    }

    /// Add binary operations (add, sub, mul, div, pow) to expressions
    fn add_binary_operations(
        &self,
        digits: &str,
        start: usize,
        end: usize,
        cache: &mut ExprCache,
        expressions: &mut Vec<Expression>,
    ) {
        for partition in generate_partitions(start, end, 2) {
            if let (Some(&(start1, end1)), Some(&(start2, end2))) =
                (partition.first(), partition.get(1))
            {
                let left_exprs = self.generate_expressions_memo(digits, start1, end1, cache);
                let right_exprs = self.generate_expressions_memo(digits, start2, end2, cache);

                for left in &left_exprs {
                    for right in &right_exprs {
                        expressions.extend([
                            Expression::Add(Box::new(left.clone()), Box::new(right.clone())),
                            Expression::Sub(Box::new(left.clone()), Box::new(right.clone())),
                            Expression::Mul(Box::new(left.clone()), Box::new(right.clone())),
                            Expression::Div(Box::new(left.clone()), Box::new(right.clone())),
                            Expression::Pow(Box::new(left.clone()), Box::new(right.clone())),
                        ]);
                    }
                }
            }
        }
    }

    /// Add nth root operations to expressions
    fn add_nth_root_operations(
        &self,
        digits: &str,
        start: usize,
        end: usize,
        cache: &mut ExprCache,
        expressions: &mut Vec<Expression>,
    ) {
        for partition in generate_partitions(start, end, 2) {
            if let (Some(&(start1, end1)), Some(&(start2, end2))) =
                (partition.first(), partition.get(1))
            {
                // First block must form an integer >= 2 for the root index
                if let Ok(n_num) = digits_to_number(digits, start1, end1) {
                    if n_num >= 2.0 && n_num.fract() == 0.0 && n_num <= self.config.max_root_degree
                    {
                        let n_expr = Expression::Number(n_num);
                        let a_exprs = self.generate_expressions_memo(digits, start2, end2, cache);

                        for a_expr in &a_exprs {
                            expressions.push(Expression::NthRoot(
                                Box::new(n_expr.clone()),
                                Box::new(a_expr.clone()),
                            ));
                        }
                    }
                }
            }
        }
    }

    /// Add unary negation operations to expressions
    fn add_negation_operations(&self, expressions: &mut Vec<Expression>) {
        // Unary negation (only for composite expressions to avoid redundancy)
        let composite_expressions: Vec<_> = expressions.iter().skip(1).cloned().collect();
        for expr in composite_expressions {
            expressions.push(Expression::Neg(Box::new(expr)));
        }
    }

    /// Wrapper function that creates cache and calls memoized version
    fn generate_expressions(&self, digits: &str, start: usize, end: usize) -> Vec<Expression> {
        let mut cache = HashMap::new();
        self.generate_expressions_memo(digits, start, end, &mut cache)
    }
}

impl Default for ExpressionSolver {
    fn default() -> Self {
        Self::new(SolverConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_expression_with_nth_root() {
        // Test that find_expression can find nth root solutions
        // Using digits "327" to find target 3 (cube root of 27 = 3)
        let solver = ExpressionSolver::default();
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
    fn test_solver_with_custom_config() {
        let config = SolverConfig {
            max_root_degree: 5.0,
            epsilon: 1e-6,
        };
        let solver = ExpressionSolver::new(config);

        // Should still find basic solutions
        let result = solver.find_expression("24", 6.0);
        assert!(result.is_some());
    }
}
