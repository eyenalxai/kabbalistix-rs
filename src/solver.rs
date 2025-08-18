use crate::expression::{Expression, ExpressionError};
use crate::utils::{UtilsError, digits_to_number, generate_partitions};
use dashmap::DashMap;
use log::{debug, info};
use rayon::prelude::*;
use std::collections::VecDeque;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};
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
const GENERATION_BATCH_SIZE: usize = 5000; // Generate expressions in smaller batches

/// Represents a work item for generating expressions
#[derive(Debug, Clone)]
struct GenerationTask {
    digits: String,
    start: usize,
    end: usize,
    depth: usize,
}

/// Streaming expression generator that produces expressions in batches
pub struct ExpressionBatchGenerator {
    work_queue: VecDeque<GenerationTask>,
    solver: Arc<ExpressionSolver>,
    current_batch: Vec<Expression>,
    total_generated: usize,
}

impl ExpressionBatchGenerator {
    fn new(digits: String, solver: Arc<ExpressionSolver>) -> Self {
        let mut work_queue = VecDeque::new();
        work_queue.push_back(GenerationTask {
            digits: digits.clone(),
            start: 0,
            end: digits.len(),
            depth: 0,
        });

        info!(
            "Initialized streaming expression generator with {} digits, batch size: {}",
            digits.len(),
            GENERATION_BATCH_SIZE
        );

        Self {
            work_queue,
            solver,
            current_batch: Vec::new(),
            total_generated: 0,
        }
    }

    /// Generate the next batch of expressions
    fn next_batch(&mut self) -> Option<Vec<Expression>> {
        self.current_batch.clear();

        // Process work items until we have enough expressions for a batch or run out of work
        while self.current_batch.len() < GENERATION_BATCH_SIZE && !self.work_queue.is_empty() {
            if let Some(task) = self.work_queue.pop_front() {
                self.process_task(task);
            }
        }

        if self.current_batch.is_empty() {
            None
        } else {
            self.total_generated += self.current_batch.len();
            info!(
                "Generated batch of {} expressions (total: {})",
                self.current_batch.len(),
                self.total_generated
            );
            Some(self.current_batch.clone())
        }
    }

    fn process_task(&mut self, task: GenerationTask) {
        let length = task.end - task.start;

        // Base case: single number
        if let Ok(num) = digits_to_number(&task.digits, task.start, task.end) {
            self.current_batch.push(Expression::Number(num));
        }

        if length >= 2 {
            // Add binary operations
            self.add_binary_operations_to_batch(&task);
            // Add nth root operations
            self.add_nth_root_operations_to_batch(&task);
            // Add negation operations (only for composite expressions)
            if task.depth > 0 {
                self.add_negation_operations_to_batch();
            }
        }
    }

    fn add_binary_operations_to_batch(&mut self, task: &GenerationTask) {
        let partitions = generate_partitions(task.start, task.end, 2);

        for partition in partitions {
            if let (Some(&(start1, end1)), Some(&(start2, end2))) =
                (partition.first(), partition.get(1))
            {
                // Generate smaller sub-expressions directly if they're small enough
                let left_exprs = if end1 - start1 <= 2 {
                    self.solver
                        .generate_expressions_small(&task.digits, start1, end1)
                } else {
                    // Add to work queue for later processing
                    self.work_queue.push_back(GenerationTask {
                        digits: task.digits.clone(),
                        start: start1,
                        end: end1,
                        depth: task.depth + 1,
                    });
                    continue;
                };

                let right_exprs = if end2 - start2 <= 2 {
                    self.solver
                        .generate_expressions_small(&task.digits, start2, end2)
                } else {
                    // Add to work queue for later processing
                    self.work_queue.push_back(GenerationTask {
                        digits: task.digits.clone(),
                        start: start2,
                        end: end2,
                        depth: task.depth + 1,
                    });
                    continue;
                };

                // Generate binary operations
                for left in left_exprs.iter() {
                    for right in right_exprs.iter() {
                        self.current_batch.extend([
                            Expression::Add(Box::new(left.clone()), Box::new(right.clone())),
                            Expression::Sub(Box::new(left.clone()), Box::new(right.clone())),
                            Expression::Mul(Box::new(left.clone()), Box::new(right.clone())),
                            Expression::Div(Box::new(left.clone()), Box::new(right.clone())),
                            Expression::Pow(Box::new(left.clone()), Box::new(right.clone())),
                        ]);

                        // Stop if we've generated enough for this batch
                        if self.current_batch.len() >= GENERATION_BATCH_SIZE {
                            return;
                        }
                    }
                }
            }
        }
    }

    fn add_nth_root_operations_to_batch(&mut self, task: &GenerationTask) {
        let partitions = generate_partitions(task.start, task.end, 2);

        for partition in partitions {
            if let (Some(&(start1, end1)), Some(&(start2, end2))) =
                (partition.first(), partition.get(1))
            {
                // First block must form an integer >= 2 for the root index
                if let Ok(n_num) = digits_to_number(&task.digits, start1, end1) {
                    if n_num >= 2.0 && n_num.fract() == 0.0 && n_num <= MAX_ROOT_DEGREE {
                        let n_expr = Expression::Number(n_num);

                        let a_exprs = if end2 - start2 <= 2 {
                            self.solver
                                .generate_expressions_small(&task.digits, start2, end2)
                        } else {
                            // For larger expressions, generate a smaller set
                            self.solver
                                .generate_expressions_recursive(&task.digits, start2, end2)
                        };

                        for a_expr in a_exprs.iter() {
                            self.current_batch.push(Expression::NthRoot(
                                Box::new(n_expr.clone()),
                                Box::new(a_expr.clone()),
                            ));

                            // Stop if we've generated enough for this batch
                            if self.current_batch.len() >= GENERATION_BATCH_SIZE {
                                return;
                            }
                        }
                    }
                }
            }
        }
    }

    fn add_negation_operations_to_batch(&mut self) {
        let expressions_to_negate: Vec<_> = self
            .current_batch
            .iter()
            .filter(|expr| !matches!(expr, Expression::Neg(_)))
            .cloned()
            .collect();

        for expr in expressions_to_negate {
            self.current_batch.push(Expression::Neg(Box::new(expr)));
            if self.current_batch.len() >= GENERATION_BATCH_SIZE {
                break;
            }
        }
    }
}

/// Main solver for finding expressions that match a target value
pub struct ExpressionSolver {
    // Memoization cache: (start, end) -> all expressions generated for that slice
    cache: DashMap<(usize, usize), Arc<Vec<Expression>>>,
}

impl ExpressionSolver {
    /// Create a new expression solver
    pub fn new() -> Self {
        Self {
            cache: DashMap::new(),
        }
    }

    /// Generate expressions for small digit ranges (≤ 2 digits) directly
    fn generate_expressions_small(
        &self,
        digits: &str,
        start: usize,
        end: usize,
    ) -> Arc<Vec<Expression>> {
        if end - start <= 2 {
            self.generate_expressions_recursive(digits, start, end)
        } else {
            Arc::new(Vec::new())
        }
    }

    /// Find an expression from the given digits that evaluates to the target
    ///
    /// # Panics
    ///
    /// This function may panic if there are issues with internal mutex locking,
    /// which should be extremely rare in normal operation.
    pub fn find_expression(&self, digits: &str, target: f64) -> Option<Expression> {
        info!("Starting streaming expression generation and evaluation");

        // Create a streaming generator
        let solver_arc = Arc::new(ExpressionSolver::new());
        let mut generator = ExpressionBatchGenerator::new(digits.to_string(), solver_arc);

        // Track overall statistics
        let mut total_evaluated = 0;
        let mut total_valid = 0;
        let mut closest_distance = f64::INFINITY;
        let mut closest_expression: Option<Expression> = None;
        let mut closest_value = 0.0;
        let mut batch_num = 0;

        // Process expressions in streaming batches
        while let Some(batch) = generator.next_batch() {
            batch_num += 1;
            debug!(
                "Processing batch {} ({} expressions)",
                batch_num,
                batch.len()
            );

            info!(
                "Processing batch {} with {} expressions",
                batch_num,
                batch.len()
            );

            // Use Arc<AtomicUsize> to track counts across threads for this batch
            let batch_evaluated_count = Arc::new(AtomicUsize::new(0));
            let batch_valid_count = Arc::new(AtomicUsize::new(0));
            let batch_closest_distance = Arc::new(std::sync::Mutex::new(f64::INFINITY));
            let batch_closest_expr = Arc::new(std::sync::Mutex::new(None::<(Expression, f64)>));

            // Use parallel iterator to find matching expression in this batch
            let batch_result = batch.par_iter().find_map_any(|expr| {
                // Update batch evaluated count
                batch_evaluated_count.fetch_add(1, Ordering::Relaxed);

                if let Ok(value) = expr.evaluate() {
                    // Update batch valid count
                    batch_valid_count.fetch_add(1, Ordering::Relaxed);

                    debug!("Expression {} evaluates to {}", expr, value);

                    // Check if this is an exact match
                    if (value - target).abs() < EPSILON {
                        return Some(expr.clone());
                    }

                    // Track the closest result in this batch
                    let distance = (value - target).abs();
                    if let Ok(mut batch_closest_dist) = batch_closest_distance.lock() {
                        if distance < *batch_closest_dist {
                            *batch_closest_dist = distance;
                            if let Ok(mut batch_closest) = batch_closest_expr.lock() {
                                *batch_closest = Some((expr.clone(), value));
                            }
                        }
                    }
                } else {
                    debug!("Expression {} failed to evaluate", expr);
                }
                None
            });

            // Update totals with batch results
            let batch_evaluated = batch_evaluated_count.load(Ordering::Relaxed);
            let batch_valid = batch_valid_count.load(Ordering::Relaxed);
            total_evaluated += batch_evaluated;
            total_valid += batch_valid;

            // Check if we found an exact match in this batch
            if let Some(exact_match) = batch_result {
                info!(
                    "Found exact match in batch {} after evaluating {} total expressions ({} valid)",
                    batch_num, total_evaluated, total_valid
                );
                return Some(exact_match);
            }

            // Update global closest result and get batch distance for logging
            let batch_closest_dist =
                if let Ok(batch_closest_dist_guard) = batch_closest_distance.lock() {
                    let batch_closest_dist = *batch_closest_dist_guard;
                    if batch_closest_dist < closest_distance {
                        closest_distance = batch_closest_dist;
                        if let Ok(batch_closest_guard) = batch_closest_expr.lock() {
                            if let Some((expr, value)) = batch_closest_guard.as_ref() {
                                closest_expression = Some(expr.clone());
                                closest_value = *value;
                            }
                        }
                    }
                    batch_closest_dist
                } else {
                    f64::INFINITY
                };

            // Log batch results
            info!(
                "Batch {} complete: evaluated {} expressions ({} valid). Closest in batch: distance {:.6}",
                batch_num, batch_evaluated, batch_valid, batch_closest_dist
            );

            // Log overall closest so far
            if let Some(ref closest_expr) = closest_expression {
                info!(
                    "Overall closest so far: {} = {:.6} (distance: {:.6})",
                    closest_expr, closest_value, closest_distance
                );
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

    /// Generate all possible expressions from a digit string
    fn generate_expressions_recursive(
        &self,
        digits: &str,
        start: usize,
        end: usize,
    ) -> Arc<Vec<Expression>> {
        if start >= end || start >= digits.len() || end > digits.len() {
            return Arc::new(Vec::new());
        }

        // Check memoization cache first
        if let Some(found) = self.cache.get(&(start, end)) {
            return Arc::clone(&found);
        }

        let length = end - start;
        let mut expressions: Vec<Expression> = Vec::new();

        // Base case: single number (always include this)
        if let Ok(num) = digits_to_number(digits, start, end) {
            expressions.push(Expression::Number(num));
        }

        if length >= 2 {
            self.add_binary_operations(digits, start, end, &mut expressions);
            self.add_nth_root_operations(digits, start, end, &mut expressions);
            self.add_negation_operations(&mut expressions);
        }

        let arc_vec = Arc::new(expressions);
        // Insert into cache for future reuse
        let _ = self.cache.insert((start, end), Arc::clone(&arc_vec));
        arc_vec
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
        // Unary negation (only for composite expressions and avoid double negatives)
        let composite_expressions: Vec<_> = expressions.iter().skip(1).cloned().collect();
        for expr in composite_expressions {
            match expr {
                Expression::Neg(_) => {
                    // Skip generating -(-x)
                }
                _ => expressions.push(Expression::Neg(Box::new(expr))),
            }
        }
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
