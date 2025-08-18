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
            if let Ok(value) = expr.evaluate() {
                if (value - target).abs() < EPSILON {
                    info!("Found exact match as base number: {}", expr);
                    return Some(expr);
                }
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
            // Evaluate this full partition
            let candidates: Vec<Expression> = if partition.len() == 2 {
                self.build_binary_partition_expressions(digits, &partition)
            } else {
                self.build_nary_partition_expressions(digits, &partition)
            };

            for expr in candidates {
                if let Ok(value) = expr.evaluate() {
                    if (value - target).abs() < EPSILON {
                        return Some(expr);
                    }
                }
            }
            return None;
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
    fn build_small_expressions(&self, digits: &str, start: usize, end: usize) -> Vec<Expression> {
        let mut expressions = Vec::new();

        // Base number for the range
        if let Ok(num) = digits_to_number(digits, start, end) {
            expressions.push(Expression::Number(num));
        }

        let length = end - start;
        if (2..=SMALL_RANGE_THRESHOLD).contains(&length) {
            // Parallelize partitions up to SMALL_RANGE_THRESHOLD for subranges
            let max_blocks = std::cmp::min(length, 4);
            let mut extra: Vec<Expression> = (2..=max_blocks)
                .into_par_iter()
                .map(|num_blocks| {
                    let parts = generate_partitions(start, end, num_blocks);
                    parts
                        .into_par_iter()
                        .map(|partition| {
                            if num_blocks == 2 {
                                self.build_binary_partition_expressions(digits, &partition)
                            } else {
                                self.build_nary_partition_expressions(digits, &partition)
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

            // Add negations of composite expressions
            let composite: Vec<Expression> = extra
                .iter()
                .filter(|e| !matches!(e, Expression::Neg(_)))
                .cloned()
                .collect();
            extra.extend(composite.into_iter().map(|e| Expression::Neg(Box::new(e))));

            expressions.append(&mut extra);
        }

        expressions
    }

    fn build_binary_partition_expressions(
        &self,
        digits: &str,
        partition: &[(usize, usize)],
    ) -> Vec<Expression> {
        let mut out = Vec::new();
        if let (Some(&(s1, e1)), Some(&(s2, e2))) = (partition.first(), partition.get(1)) {
            let left = self.build_small_expressions(digits, s1, e1);
            let right = self.build_small_expressions(digits, s2, e2);

            let mut cross: Vec<Expression> = left
                .par_iter()
                .flat_map_iter(|l| {
                    right.iter().flat_map(move |r| {
                        let mut v = ExpressionGenerator::generate_binary_ops(l, r);
                        if let Some(root) = ExpressionGenerator::generate_nth_root(l, r) {
                            v.push(root);
                        }
                        v
                    })
                })
                .collect();
            out.append(&mut cross);
        }
        out
    }

    fn build_nary_partition_expressions(
        &self,
        digits: &str,
        partition: &[(usize, usize)],
    ) -> Vec<Expression> {
        let mut all_operands: Vec<Vec<Expression>> = Vec::new();
        for &(s, e) in partition {
            let sub = self.build_small_expressions(digits, s, e);
            if sub.is_empty() {
                return Vec::new();
            }
            all_operands.push(sub);
        }

        // Generate Cartesian product of operand choices sequentially (depths are small) then map to n-ary ops
        let mut stack: Vec<(usize, Vec<Expression>)> = vec![(0, Vec::new())];
        let mut results: Vec<Expression> = Vec::new();
        while let Some((depth, current)) = stack.pop() {
            if depth == all_operands.len() {
                if current.len() >= 2 {
                    results.extend(ExpressionGenerator::generate_nary_ops(&current));
                }
                continue;
            }
            if let Some(level) = all_operands.get(depth) {
                for expr in level {
                    let mut next = current.clone();
                    next.push(expr.clone());
                    stack.push((depth + 1, next));
                }
            }
        }
        results
    }
}

impl Default for ExpressionSolver {
    fn default() -> Self {
        Self::new()
    }
}
