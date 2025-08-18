use crate::expression::{Expression, ExpressionError};
use crate::utils::{UtilsError, digits_to_number, generate_partitions};
use log::{debug, info};

use std::collections::{HashMap, VecDeque};
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

/// Iterative expression generator that yields expressions one at a time without storing them
pub struct ExpressionIterator {
    work_queue: VecDeque<WorkItem>,
    digits: String,
    small_cache: HashMap<(usize, usize), Vec<Expression>>, // Cache for small ranges only
}

#[derive(Debug, Clone)]
struct WorkItem {
    start: usize,
    end: usize,
    state: GenerationState,
}

#[derive(Debug, Clone)]
enum GenerationState {
    /// Generate the base number
    BaseNumber,
    /// Generate binary operations for a specific partition
    BinaryOps {
        partition_idx: usize,
        left_exprs: Vec<Expression>,
        right_exprs: Vec<Expression>,
        left_idx: usize,
        right_idx: usize,
        op_idx: usize, // 0=Add, 1=Sub, 2=Mul, 3=Div, 4=Pow
    },
    /// Generate nth root operations for a specific partition
    NthRoots {
        partition_idx: usize,
        n_value: f64,
        a_exprs: Vec<Expression>,
        a_idx: usize,
    },
    /// Generate negation operations
    Negations {
        base_exprs: Vec<Expression>,
        expr_idx: usize,
    },
    /// Initialize binary operations for a partition
    InitBinary { partition_idx: usize },
    /// Initialize nth root operations for a partition
    InitNthRoot { partition_idx: usize },
    /// Initialize negation operations
    InitNegations,
}

impl ExpressionIterator {
    pub fn new(digits: String) -> Self {
        let mut work_queue = VecDeque::new();
        let len = digits.len();

        // Start with base number generation
        work_queue.push_back(WorkItem {
            start: 0,
            end: len,
            state: GenerationState::BaseNumber,
        });

        info!(
            "Initialized iterative expression generator with {} digits",
            len
        );

        Self {
            work_queue,
            digits,
            small_cache: HashMap::new(),
        }
    }

    /// Get expressions for small ranges (≤ 2 digits) with caching
    fn get_small_expressions(&mut self, start: usize, end: usize) -> Vec<Expression> {
        let key = (start, end);
        if let Some(cached) = self.small_cache.get(&key) {
            return cached.clone();
        }

        let mut expressions = Vec::new();

        // Base case: single number
        if let Ok(num) = digits_to_number(&self.digits, start, end) {
            expressions.push(Expression::Number(num));
        }

        let length = end - start;
        if length == 2 {
            let partitions = generate_partitions(start, end, 2);

            // Binary operations for 2-digit combinations
            for partition in &partitions {
                if let (Some(&(start1, end1)), Some(&(start2, end2))) =
                    (partition.first(), partition.get(1))
                {
                    if let (Ok(left_num), Ok(right_num)) = (
                        digits_to_number(&self.digits, start1, end1),
                        digits_to_number(&self.digits, start2, end2),
                    ) {
                        let left = Expression::Number(left_num);
                        let right = Expression::Number(right_num);

                        expressions.push(Expression::Add(
                            Box::new(left.clone()),
                            Box::new(right.clone()),
                        ));
                        expressions.push(Expression::Sub(
                            Box::new(left.clone()),
                            Box::new(right.clone()),
                        ));
                        expressions.push(Expression::Mul(
                            Box::new(left.clone()),
                            Box::new(right.clone()),
                        ));
                        expressions.push(Expression::Div(
                            Box::new(left.clone()),
                            Box::new(right.clone()),
                        ));
                        expressions.push(Expression::Pow(
                            Box::new(left.clone()),
                            Box::new(right.clone()),
                        ));

                        // Nth root if applicable
                        if left_num >= 2.0 && left_num.fract() == 0.0 && left_num <= MAX_ROOT_DEGREE
                        {
                            expressions.push(Expression::NthRoot(
                                Box::new(left.clone()),
                                Box::new(right.clone()),
                            ));
                        }
                    }
                }
            }

            // Negation operations (skip the base number)
            let composite_exprs: Vec<_> = expressions
                .iter()
                .skip(1)
                .filter(|expr| !matches!(expr, Expression::Neg(_)))
                .cloned()
                .collect();

            for expr in composite_exprs {
                expressions.push(Expression::Neg(Box::new(expr)));
            }
        }

        self.small_cache.insert(key, expressions.clone());
        expressions
    }
}

impl Iterator for ExpressionIterator {
    type Item = Expression;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(item) = self.work_queue.pop_front() {
            let length = item.end - item.start;

            match item.state {
                GenerationState::BaseNumber => {
                    // Generate base number
                    if let Ok(num) = digits_to_number(&self.digits, item.start, item.end) {
                        // Queue up binary operations if length >= 2
                        if length >= 2 {
                            self.work_queue.push_back(WorkItem {
                                start: item.start,
                                end: item.end,
                                state: GenerationState::InitBinary { partition_idx: 0 },
                            });
                        }
                        return Some(Expression::Number(num));
                    }
                }

                GenerationState::InitBinary { partition_idx } => {
                    let partitions = generate_partitions(item.start, item.end, 2);
                    if let Some(partition) = partitions.get(partition_idx) {
                        if let (Some(&(start1, end1)), Some(&(start2, end2))) =
                            (partition.first(), partition.get(1))
                        {
                            // Get expressions for left and right parts
                            let left_exprs = if end1 - start1 <= 2 {
                                self.get_small_expressions(start1, end1)
                            } else {
                                // For larger expressions, we'll generate them on demand
                                // This is a simplified approach - in a full implementation,
                                // we'd need a more sophisticated lazy evaluation system
                                vec![]
                            };

                            let right_exprs = if end2 - start2 <= 2 {
                                self.get_small_expressions(start2, end2)
                            } else {
                                vec![]
                            };

                            if !left_exprs.is_empty() && !right_exprs.is_empty() {
                                // Queue binary operations
                                self.work_queue.push_back(WorkItem {
                                    start: item.start,
                                    end: item.end,
                                    state: GenerationState::BinaryOps {
                                        partition_idx,
                                        left_exprs,
                                        right_exprs,
                                        left_idx: 0,
                                        right_idx: 0,
                                        op_idx: 0,
                                    },
                                });
                            }
                        }

                        // Queue next partition
                        self.work_queue.push_back(WorkItem {
                            start: item.start,
                            end: item.end,
                            state: GenerationState::InitBinary {
                                partition_idx: partition_idx + 1,
                            },
                        });
                    } else {
                        // Move to nth root operations
                        self.work_queue.push_back(WorkItem {
                            start: item.start,
                            end: item.end,
                            state: GenerationState::InitNthRoot { partition_idx: 0 },
                        });
                    }
                }

                GenerationState::BinaryOps {
                    partition_idx,
                    left_exprs,
                    right_exprs,
                    left_idx,
                    right_idx,
                    op_idx,
                } => {
                    if let (Some(left), Some(right)) =
                        (left_exprs.get(left_idx), right_exprs.get(right_idx))
                    {
                        let expr = match op_idx {
                            0 => Expression::Add(Box::new(left.clone()), Box::new(right.clone())),
                            1 => Expression::Sub(Box::new(left.clone()), Box::new(right.clone())),
                            2 => Expression::Mul(Box::new(left.clone()), Box::new(right.clone())),
                            3 => Expression::Div(Box::new(left.clone()), Box::new(right.clone())),
                            4 => Expression::Pow(Box::new(left.clone()), Box::new(right.clone())),
                            _ => continue,
                        };

                        // Update indices for next iteration
                        let (next_left_idx, next_right_idx, next_op_idx) = if op_idx < 4 {
                            (left_idx, right_idx, op_idx + 1)
                        } else if right_idx + 1 < right_exprs.len() {
                            (left_idx, right_idx + 1, 0)
                        } else if left_idx + 1 < left_exprs.len() {
                            (left_idx + 1, 0, 0)
                        } else {
                            // Done with this partition, don't queue anything more
                            (left_exprs.len(), right_exprs.len(), 5)
                        };

                        // Queue next iteration if not done
                        if next_left_idx < left_exprs.len()
                            && next_right_idx < right_exprs.len()
                            && next_op_idx <= 4
                        {
                            self.work_queue.push_back(WorkItem {
                                start: item.start,
                                end: item.end,
                                state: GenerationState::BinaryOps {
                                    partition_idx,
                                    left_exprs: left_exprs.clone(),
                                    right_exprs: right_exprs.clone(),
                                    left_idx: next_left_idx,
                                    right_idx: next_right_idx,
                                    op_idx: next_op_idx,
                                },
                            });
                        }

                        return Some(expr);
                    }
                }

                GenerationState::InitNthRoot { partition_idx } => {
                    let partitions = generate_partitions(item.start, item.end, 2);
                    if let Some(partition) = partitions.get(partition_idx) {
                        if let (Some(&(start1, end1)), Some(&(start2, end2))) =
                            (partition.first(), partition.get(1))
                        {
                            // First part must be a valid root index
                            if let Ok(n_num) = digits_to_number(&self.digits, start1, end1) {
                                if n_num >= 2.0 && n_num.fract() == 0.0 && n_num <= MAX_ROOT_DEGREE
                                {
                                    let a_exprs = if end2 - start2 <= 2 {
                                        self.get_small_expressions(start2, end2)
                                    } else {
                                        vec![]
                                    };

                                    if !a_exprs.is_empty() {
                                        self.work_queue.push_back(WorkItem {
                                            start: item.start,
                                            end: item.end,
                                            state: GenerationState::NthRoots {
                                                partition_idx,
                                                n_value: n_num,
                                                a_exprs,
                                                a_idx: 0,
                                            },
                                        });
                                    }
                                }
                            }
                        }

                        // Queue next partition
                        self.work_queue.push_back(WorkItem {
                            start: item.start,
                            end: item.end,
                            state: GenerationState::InitNthRoot {
                                partition_idx: partition_idx + 1,
                            },
                        });
                    } else if length > 1 {
                        // Move to negation operations
                        self.work_queue.push_back(WorkItem {
                            start: item.start,
                            end: item.end,
                            state: GenerationState::InitNegations,
                        });
                    }
                }

                GenerationState::NthRoots {
                    partition_idx,
                    n_value,
                    a_exprs,
                    a_idx,
                } => {
                    if let Some(a_expr) = a_exprs.get(a_idx) {
                        let n_expr = Expression::Number(n_value);

                        // Queue next iteration
                        if a_idx + 1 < a_exprs.len() {
                            self.work_queue.push_back(WorkItem {
                                start: item.start,
                                end: item.end,
                                state: GenerationState::NthRoots {
                                    partition_idx,
                                    n_value,
                                    a_exprs: a_exprs.clone(),
                                    a_idx: a_idx + 1,
                                },
                            });
                        }

                        return Some(Expression::NthRoot(
                            Box::new(n_expr),
                            Box::new(a_expr.clone()),
                        ));
                    }
                }

                GenerationState::InitNegations => {
                    // Get base expressions to negate (simplified - only from small cache for now)
                    let base_exprs = if length <= 2 {
                        let mut all_exprs = self.get_small_expressions(item.start, item.end);
                        // Remove the base number and already negated expressions
                        all_exprs.retain(|expr| {
                            !matches!(expr, Expression::Number(_) | Expression::Neg(_))
                        });
                        all_exprs
                    } else {
                        vec![]
                    };

                    if !base_exprs.is_empty() {
                        self.work_queue.push_back(WorkItem {
                            start: item.start,
                            end: item.end,
                            state: GenerationState::Negations {
                                base_exprs,
                                expr_idx: 0,
                            },
                        });
                    }
                }

                GenerationState::Negations {
                    base_exprs,
                    expr_idx,
                } => {
                    if let Some(expr) = base_exprs.get(expr_idx) {
                        // Queue next iteration
                        if expr_idx + 1 < base_exprs.len() {
                            self.work_queue.push_back(WorkItem {
                                start: item.start,
                                end: item.end,
                                state: GenerationState::Negations {
                                    base_exprs: base_exprs.clone(),
                                    expr_idx: expr_idx + 1,
                                },
                            });
                        }

                        return Some(Expression::Neg(Box::new(expr.clone())));
                    }
                }
            }
        }
        None
    }
}

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
}
