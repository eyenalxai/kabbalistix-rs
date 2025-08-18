use crate::iterator::constants::SMALL_RANGE_THRESHOLD;
use crate::iterator::generator::ExpressionGenerator;
use crate::utils::{digits_to_number, generate_partitions};
use log::info;
use rayon::prelude::*;

use crate::expression::Expression;
// use crate::iterator::ExpressionIterator;
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
        info!("Starting fully parallel partition-based search");

        let len = digits.len();

        // Quick check: base number using all digits
        if let Ok(num) = digits_to_number(digits, 0, len) {
            let expr = Expression::Number(num);
            if let Ok(value) = expr.evaluate()
                && (value - target).abs() < EPSILON
            {
                info!("Found exact match as base number: {}", expr);
                return Some(expr);
            }
        }

        // Parallelize across top-level partitions without pre-allocating the full partition list
        for num_blocks in 2..=len {
            // allow deep partitions; inner generation limits depth for small subranges
            let min_end = 1;
            let max_end = len - (num_blocks - 1);

            if let Some(found) = (min_end..=max_end)
                .into_par_iter()
                .filter_map(|split_point| {
                    let mut prefix = Vec::with_capacity(num_blocks);
                    prefix.push((0, split_point));
                    self.search_partitions_branch(
                        digits,
                        target,
                        split_point,
                        len,
                        num_blocks - 1,
                        prefix,
                    )
                })
                .find_any(|_| true)
            {
                info!("Found exact match in partitioned search: {}", found);
                return Some(found);
            }
        }

        info!("No exact match found");
        None
    }

    fn search_partitions_branch(
        &self,
        digits: &str,
        target: f64,
        start: usize,
        end: usize,
        remaining_blocks: usize,
        current_partition: Vec<(usize, usize)>,
    ) -> Option<Expression> {
        if remaining_blocks == 1 {
            let mut partition = current_partition;
            partition.push((start, end));
            // Evaluate this full partition (streaming generation)
            if partition.len() == 2 {
                return self.search_binary_partition_for_target(digits, &partition, target);
            } else {
                return self.search_nary_partition_for_target(digits, &partition, target);
            }
        }

        let min_end = start + 1;
        let max_end = end - (remaining_blocks - 1);

        (min_end..=max_end)
            .into_par_iter()
            .filter_map(|split_point| {
                let mut next = current_partition.clone();
                next.push((start, split_point));
                self.search_partitions_branch(
                    digits,
                    target,
                    split_point,
                    end,
                    remaining_blocks - 1,
                    next,
                )
            })
            .find_any(|_| true)
    }
    /// Build expressions for a small range iteratively (parallelized) without recursion via generics
    fn build_small_expressions(&self, digits: &str, start: usize, end: usize) -> Vec<Expression> {
        let mut expressions = Vec::new();

        if let Ok(num) = digits_to_number(digits, start, end) {
            expressions.push(Expression::Number(num));
        }

        let length = end - start;
        if length >= 2 && length <= SMALL_RANGE_THRESHOLD {
            self.generate_small_partitioned_expressions(digits, start, end, &mut expressions);

            // add negations of composite expressions (skip base number and already negated)
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

        expressions
    }

    fn generate_small_partitioned_expressions(
        &self,
        digits: &str,
        start: usize,
        end: usize,
        expressions: &mut Vec<Expression>,
    ) {
        let length = end - start;
        let max_blocks = std::cmp::min(length, 4);

        let mut parallel_results: Vec<Expression> = (2..=max_blocks)
            .into_par_iter()
            .map(|num_blocks| {
                let partitions = generate_partitions(start, end, num_blocks);
                partitions
                    .into_par_iter()
                    .map(|partition| {
                        if num_blocks == 2 {
                            self.generate_binary_expressions_small_partition(digits, &partition)
                        } else {
                            self.generate_nary_expressions_small_partition(digits, &partition)
                        }
                    })
                    .reduce(Vec::new, |mut acc, mut v| {
                        acc.append(&mut v);
                        acc
                    })
            })
            .reduce(Vec::new, |mut acc, mut v| {
                acc.append(&mut v);
                acc
            });

        expressions.append(&mut parallel_results);
    }

    fn generate_binary_expressions_small_partition(
        &self,
        digits: &str,
        partition: &[(usize, usize)],
    ) -> Vec<Expression> {
        let mut out = Vec::new();
        if let (Some(&(start1, end1)), Some(&(start2, end2))) =
            (partition.first(), partition.get(1))
        {
            let left_exprs = self.get_sub_expressions(digits, start1, end1);
            let right_exprs = self.get_sub_expressions(digits, start2, end2);

            let mut results: Vec<Expression> = left_exprs
                .par_iter()
                .flat_map_iter(|left| {
                    right_exprs.iter().flat_map(move |right| {
                        let mut local = ExpressionGenerator::generate_binary_ops(left, right);
                        if let Some(nth_root) = ExpressionGenerator::generate_nth_root(left, right)
                        {
                            local.push(nth_root);
                        }
                        local
                    })
                })
                .collect();

            out.append(&mut results);
        }
        out
    }

    fn generate_nary_expressions_small_partition(
        &self,
        digits: &str,
        partition: &[(usize, usize)],
    ) -> Vec<Expression> {
        let mut all_operands = Vec::new();
        for &(start_i, end_i) in partition {
            let sub_exprs = self.get_sub_expressions(digits, start_i, end_i);
            if sub_exprs.is_empty() {
                return Vec::new();
            }
            all_operands.push(sub_exprs);
        }

        let mut results = Vec::new();
        self.generate_nary_combinations_small(&mut results, &all_operands);
        results
    }

    fn get_sub_expressions(&self, digits: &str, start: usize, end: usize) -> Vec<Expression> {
        if end - start == 1 {
            if let Ok(num) = digits_to_number(digits, start, end) {
                vec![Expression::Number(num)]
            } else {
                vec![]
            }
        } else {
            self.build_small_expressions(digits, start, end)
        }
    }

    fn generate_nary_combinations_small(
        &self,
        expressions: &mut Vec<Expression>,
        all_operands: &[Vec<Expression>],
    ) {
        if all_operands.is_empty() {
            return;
        }

        let mut stack: Vec<(usize, Vec<Expression>)> = Vec::new();
        stack.push((0, Vec::new()));

        while let Some((depth, current_combo)) = stack.pop() {
            if depth == all_operands.len() {
                if current_combo.len() >= 2 {
                    expressions.extend(ExpressionGenerator::generate_nary_ops(&current_combo));
                }
                continue;
            }

            if let Some(operands_at_depth) = all_operands.get(depth) {
                for expr in operands_at_depth {
                    let mut new_combo = current_combo.clone();
                    new_combo.push(expr.clone());
                    stack.push((depth + 1, new_combo));
                }
            }
        }
    }

    fn search_binary_partition_for_target(
        &self,
        digits: &str,
        partition: &[(usize, usize)],
        target: f64,
    ) -> Option<Expression> {
        if let (Some(&(s1, e1)), Some(&(s2, e2))) = (partition.first(), partition.get(1)) {
            let left_exprs = self.build_small_expressions(digits, s1, e1);
            let right_exprs = self.build_small_expressions(digits, s2, e2);

            return left_exprs
                .par_iter()
                .filter_map(|left| {
                    // Inner loop can stay sequential to reduce overhead
                    for right in &right_exprs {
                        let mut ops = ExpressionGenerator::generate_binary_ops(left, right);
                        if let Some(root) = ExpressionGenerator::generate_nth_root(left, right) {
                            ops.push(root);
                        }
                        for expr in ops {
                            if let Ok(value) = expr.evaluate()
                                && (value - target).abs() < EPSILON
                            {
                                return Some(expr);
                            }
                        }
                    }
                    None
                })
                .find_any(|_| true);
        }
        None
    }

    fn search_nary_partition_for_target(
        &self,
        digits: &str,
        partition: &[(usize, usize)],
        target: f64,
    ) -> Option<Expression> {
        // Build operand expression lists for each subrange
        let mut all_operands: Vec<Vec<Expression>> = Vec::new();
        for &(s, e) in partition {
            let exprs = self.build_small_expressions(digits, s, e);
            if exprs.is_empty() {
                return None;
            }
            all_operands.push(exprs);
        }

        if all_operands.is_empty() {
            return None;
        }

        // Parallelize by the first operand choices
        let first = all_operands.remove(0);
        first
            .into_par_iter()
            .filter_map(|first_expr| {
                // Iterative stack over remaining operands
                let mut stack: Vec<(usize, Vec<Expression>)> = Vec::new();
                stack.push((0, vec![first_expr.clone()]));

                while let Some((depth, current_combo)) = stack.pop() {
                    if depth == all_operands.len() {
                        if current_combo.len() >= 2 {
                            let ops = ExpressionGenerator::generate_nary_ops(&current_combo);
                            for expr in ops {
                                if let Ok(value) = expr.evaluate()
                                    && (value - target).abs() < EPSILON
                                {
                                    return Some(expr);
                                }
                            }
                        }
                        continue;
                    }

                    if let Some(operands_at_depth) = all_operands.get(depth) {
                        for expr in operands_at_depth {
                            let mut next_combo = current_combo.clone();
                            next_combo.push(expr.clone());
                            stack.push((depth + 1, next_combo));
                        }
                    }
                }
                None
            })
            .find_any(|_| true)
    }
}

impl Default for ExpressionSolver {
    fn default() -> Self {
        Self::new()
    }
}
