use super::generator::ExpressionGenerator;
use crate::utils::digits_to_number;
use log::info;
use rayon::prelude::*;

use crate::expression::Expression;
use crate::solver::constants::EPSILON;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

type RangeKey = (usize, usize);
type ExprVec = Arc<Vec<crate::expression::Expression>>;
type SmallCache = Arc<Mutex<HashMap<RangeKey, ExprVec>>>;

pub struct ExpressionSolver {}

impl ExpressionSolver {
    pub fn new() -> Self {
        Self {}
    }

    pub fn find_expression(&self, digits: &str, target: f64) -> Option<Expression> {
        info!("Starting fully parallel partition-based search");

        let len = digits.len();

        let cache: SmallCache = Arc::new(Mutex::new(HashMap::new()));

        if let Ok(num) = digits_to_number(digits, 0, len) {
            let expr = Expression::Number(num);
            if let Ok(value) = expr.evaluate()
                && (value - target).abs() < EPSILON
            {
                info!("Found exact match as base number: {}", expr);
                return Some(expr);
            }
        }

        for num_blocks in 2..=len {
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
                        (split_point, len),
                        num_blocks - 1,
                        prefix,
                        Arc::clone(&cache),
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
        range: (usize, usize),
        remaining_blocks: usize,
        current_partition: Vec<(usize, usize)>,
        cache: SmallCache,
    ) -> Option<Expression> {
        let (start, end) = range;
        if remaining_blocks == 1 {
            let mut partition = current_partition;
            partition.push((start, end));
            if partition.len() == 2 {
                return self.search_binary_partition_for_target(digits, &partition, target, cache);
            } else {
                return self.search_nary_partition_for_target(digits, &partition, target, cache);
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
                    (split_point, end),
                    remaining_blocks - 1,
                    next,
                    Arc::clone(&cache),
                )
            })
            .find_any(|_| true)
    }

    fn search_binary_partition_for_target(
        &self,
        digits: &str,
        partition: &[(usize, usize)],
        target: f64,
        cache: SmallCache,
    ) -> Option<Expression> {
        if let (Some(&(s1, e1)), Some(&(s2, e2))) = (partition.first(), partition.get(1)) {
            let left_arc = Self::get_small_expressions_cached(digits, s1, e1, &cache);
            let right_arc = Self::get_small_expressions_cached(digits, s2, e2, &cache);

            let left_vals: Arc<Vec<Option<f64>>> =
                Arc::new(left_arc.par_iter().map(|e| e.evaluate().ok()).collect());
            let right_vals: Arc<Vec<Option<f64>>> =
                Arc::new(right_arc.par_iter().map(|e| e.evaluate().ok()).collect());

            return left_arc
                .par_iter()
                .enumerate()
                .filter_map(|(i, left)| {
                    let left_val = match left_vals.get(i).copied().flatten() {
                        Some(v) if v.is_finite() => v,
                        _ => return None,
                    };

                    right_arc
                        .par_iter()
                        .enumerate()
                        .filter_map(|(j, right)| {
                            let right_val = match right_vals.get(j).copied().flatten() {
                                Some(v) if v.is_finite() => v,
                                _ => return None,
                            };

                            // Add
                            let sum = left_val + right_val;
                            if (sum - target).abs() < EPSILON {
                                return Some(Expression::Add(
                                    Box::new(left.clone()),
                                    Box::new(right.clone()),
                                ));
                            }

                            let diff = left_val - right_val;
                            if (diff - target).abs() < EPSILON {
                                return Some(Expression::Sub(
                                    Box::new(left.clone()),
                                    Box::new(right.clone()),
                                ));
                            }

                            let prod = left_val * right_val;
                            if (prod - target).abs() < EPSILON {
                                return Some(Expression::Mul(
                                    Box::new(left.clone()),
                                    Box::new(right.clone()),
                                ));
                            }

                            if right_val != 0.0 {
                                let quot = left_val / right_val;
                                if (quot - target).abs() < EPSILON {
                                    return Some(Expression::Div(
                                        Box::new(left.clone()),
                                        Box::new(right.clone()),
                                    ));
                                }
                            }

                            if let Some(exp) = Self::as_small_integer(right_val, -6, 6) {
                                let pow_val = left_val.powi(exp);
                                if pow_val.is_finite() && (pow_val - target).abs() < EPSILON {
                                    return Some(Expression::Pow(
                                        Box::new(left.clone()),
                                        Box::new(right.clone()),
                                    ));
                                }
                            }

                            if let Some(root) = ExpressionGenerator::generate_nth_root(left, right)
                                && let Ok(value) = root.evaluate()
                                && (value - target).abs() < EPSILON
                            {
                                return Some(root);
                            }

                            None
                        })
                        .find_any(|_| true)
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
        cache: SmallCache,
    ) -> Option<Expression> {
        let mut all_operands: Vec<Vec<Expression>> = Vec::new();
        for &(s, e) in partition {
            let exprs_arc = Self::get_small_expressions_cached(digits, s, e, &cache);
            let exprs = (*exprs_arc).clone();
            if exprs.is_empty() {
                return None;
            }
            all_operands.push(exprs);
        }

        if all_operands.is_empty() {
            return None;
        }

        Self::search_nary_operands_for_target(&all_operands, target)
    }
}

impl Default for ExpressionSolver {
    fn default() -> Self {
        Self::new()
    }
}

impl ExpressionSolver {
    fn search_nary_operands_for_target(
        all_operands: &[Vec<Expression>],
        target: f64,
    ) -> Option<Expression> {
        if all_operands.is_empty() {
            return None;
        }

        fn recurse(
            all_operands: &[Vec<Expression>],
            depth: usize,
            current_combo: Vec<Expression>,
            target: f64,
        ) -> Option<Expression> {
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
                return None;
            }

            if let Some(operands_at_depth) = all_operands.get(depth) {
                operands_at_depth
                    .par_iter()
                    .filter_map(|expr| {
                        let mut next_combo = current_combo.clone();
                        next_combo.push(expr.clone());
                        recurse(all_operands, depth + 1, next_combo, target)
                    })
                    .find_any(|_| true)
            } else {
                None
            }
        }

        recurse(all_operands, 0, Vec::new(), target)
    }
    fn get_small_expressions_cached(
        digits: &str,
        start: usize,
        end: usize,
        cache: &SmallCache,
    ) -> ExprVec {
        {
            if let Some(found) = cache
                .lock()
                .ok()
                .and_then(|m| m.get(&(start, end)).cloned())
            {
                return found;
            }
        }

        let built = ExpressionGenerator::build_small_expressions(digits, start, end);
        let arc_vec: ExprVec = Arc::new(built);
        if let Ok(mut map) = cache.lock() {
            map.insert((start, end), Arc::clone(&arc_vec));
        }
        arc_vec
    }

    fn as_small_integer(value: f64, min: i32, max: i32) -> Option<i32> {
        if !value.is_finite() {
            return None;
        }
        let rounded = value.round();
        if (rounded - value).abs() < 1e-9 {
            let n = rounded as i32;
            if n >= min && n <= max {
                return Some(n);
            }
        }
        None
    }
}
