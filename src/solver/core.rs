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
    /// Enumerate expressions for a given small range without allocating large vectors.
    /// Calls `on_expr` for each generated expression. If `on_expr` returns Some(found), propagation stops and the found value is returned.
    fn enumerate_expressions_for_range(
        &self,
        digits: &str,
        start: usize,
        end: usize,
        mut on_expr: impl FnMut(Expression) -> Option<Expression>,
    ) -> Option<Expression> {
        // Base number for the range
        if let Ok(num) = digits_to_number(digits, start, end)
            && let Some(found) = on_expr(Expression::Number(num))
        {
            return Some(found);
        }

        let length = end - start;
        if (2..=SMALL_RANGE_THRESHOLD).contains(&length) {
            let max_blocks = std::cmp::min(length, 4);
            for num_blocks in 2..=max_blocks {
                let partitions = generate_partitions(start, end, num_blocks);
                for partition in partitions {
                    if num_blocks == 2 {
                        if let (Some(&(s1, e1)), Some(&(s2, e2))) =
                            (partition.first(), partition.get(1))
                        {
                            // Enumerate left then right expressions, combine via binary ops
                            let mut found: Option<Expression> = None;
                            if let Some(_found_inner) = become self.enumerate_expressions_for_range(
                                digits,
                                s1,
                                e1,
                                |left| {
                                    if let Some(found_inner) = become self
                                        .enumerate_expressions_for_range(digits, s2, e2, |right| {
                                            let mut ops = ExpressionGenerator::generate_binary_ops(
                                                &left, &right,
                                            );
                                            if let Some(root) =
                                                ExpressionGenerator::generate_nth_root(
                                                    &left, &right,
                                                )
                                            {
                                                ops.push(root);
                                            }
                                            for expr in ops {
                                                if let Some(x) = on_expr(expr) {
                                                    return Some(x);
                                                }
                                            }
                                            None
                                        })
                                    {
                                        found = Some(found_inner);
                                    }
                                    None
                                },
                            ) {
                                return found;
                            }
                            if found.is_some() {
                                return found;
                            }
                        }
                    } else {
                        // n-ary: enumerate combinations of subrange expressions
                        let mut operands: Vec<Expression> = Vec::new();
                        if let Some(found) = self.enumerate_nary_operands(
                            digits,
                            &partition,
                            0,
                            &mut operands,
                            |ops_vec| {
                                let combo = ExpressionGenerator::generate_nary_ops(ops_vec);
                                for expr in combo {
                                    if let Some(x) = on_expr(expr) {
                                        return Some(x);
                                    }
                                }
                                None
                            },
                        ) {
                            return Some(found);
                        }
                    }
                }
            }
        }
        None
    }

    fn enumerate_nary_operands(
        &self,
        digits: &str,
        partition: &[(usize, usize)],
        idx: usize,
        current: &mut Vec<Expression>,
        mut on_full: impl FnMut(&[Expression]) -> Option<Expression>,
    ) -> Option<Expression> {
        if idx == partition.len() {
            return on_full(current);
        }
        if let Some(&(s, e)) = partition.get(idx) {
            return self.enumerate_expressions_for_range(digits, s, e, |expr| {
                current.push(expr);
                let res =
                    self.enumerate_nary_operands(digits, partition, idx + 1, current, &mut on_full);
                current.pop();
                res
            });
        }
        None
    }

    fn search_binary_partition_for_target(
        &self,
        digits: &str,
        partition: &[(usize, usize)],
        target: f64,
    ) -> Option<Expression> {
        if let (Some(&(s1, e1)), Some(&(s2, e2))) = (partition.first(), partition.get(1)) {
            let mut found: Option<Expression> = None;
            self.enumerate_expressions_for_range(digits, s1, e1, |left| {
                if let Some(x) = self.enumerate_expressions_for_range(digits, s2, e2, |right| {
                    let mut ops = ExpressionGenerator::generate_binary_ops(&left, &right);
                    if let Some(root) = ExpressionGenerator::generate_nth_root(&left, &right) {
                        ops.push(root);
                    }
                    for expr in ops {
                        if let Ok(value) = expr.evaluate()
                            && (value - target).abs() < EPSILON
                        {
                            return Some(expr);
                        }
                    }
                    None
                }) {
                    found = Some(x);
                }
                None
            });
            if found.is_some() {
                return found;
            }
        }
        None
    }

    fn search_nary_partition_for_target(
        &self,
        digits: &str,
        partition: &[(usize, usize)],
        target: f64,
    ) -> Option<Expression> {
        let mut current: Vec<Expression> = Vec::new();
        self.enumerate_nary_operands(digits, partition, 0, &mut current, |ops_vec| {
            let ops = ExpressionGenerator::generate_nary_ops(ops_vec);
            for expr in ops {
                if let Ok(value) = expr.evaluate()
                    && (value - target).abs() < EPSILON
                {
                    return Some(expr);
                }
            }
            None
        })
    }
}

impl Default for ExpressionSolver {
    fn default() -> Self {
        Self::new()
    }
}
