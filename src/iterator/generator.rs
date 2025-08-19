use crate::expression::Expression;
use crate::iterator::constants::MAX_ROOT_DEGREE;
use crate::iterator::constants::SMALL_RANGE_THRESHOLD;
use crate::utils::{digits_to_number, generate_partitions};
use rayon::prelude::*;

pub struct ExpressionGenerator;

impl ExpressionGenerator {
    pub fn generate_binary_ops(left: &Expression, right: &Expression) -> Vec<Expression> {
        vec![
            Expression::Add(Box::new(left.clone()), Box::new(right.clone())),
            Expression::Sub(Box::new(left.clone()), Box::new(right.clone())),
            Expression::Mul(Box::new(left.clone()), Box::new(right.clone())),
            Expression::Div(Box::new(left.clone()), Box::new(right.clone())),
            Expression::Pow(Box::new(left.clone()), Box::new(right.clone())),
        ]
    }

    pub fn generate_nth_root(n: &Expression, a: &Expression) -> Option<Expression> {
        if let Expression::Number(n_val) = n
            && *n_val >= 2.0
            && n_val.fract() == 0.0
            && *n_val <= MAX_ROOT_DEGREE
        {
            return Some(Expression::NthRoot(
                Box::new(n.clone()),
                Box::new(a.clone()),
            ));
        }
        None
    }

    /// # Panics
    ///
    /// This function does not panic. If fewer than 2 operands are provided,
    /// an empty vector is returned.
    pub fn generate_nary_ops(operands: &[Expression]) -> Vec<Expression> {
        if operands.len() < 2 {
            return Vec::new();
        }

        let mut results = Vec::with_capacity(4);

        let (first, rest) = match operands.split_first() {
            Some((first, rest)) => (first, rest),
            None => return results,
        };

        let mut add_result = first.clone();
        for operand in rest.iter() {
            add_result = Expression::Add(Box::new(add_result), Box::new(operand.clone()));
        }
        results.push(add_result);

        let mut mul_result = first.clone();
        for operand in rest.iter() {
            mul_result = Expression::Mul(Box::new(mul_result), Box::new(operand.clone()));
        }
        results.push(mul_result);

        let mut sub_result = first.clone();
        for operand in rest.iter() {
            sub_result = Expression::Sub(Box::new(sub_result), Box::new(operand.clone()));
        }
        results.push(sub_result);

        if rest.len() >= 2 {
            let mut mixed_result = Expression::Sub(
                Box::new(first.clone()),
                Box::new(rest.first().cloned().unwrap_or_else(|| first.clone())),
            );
            for operand in rest.iter().skip(1) {
                mixed_result = Expression::Add(Box::new(mixed_result), Box::new(operand.clone()));
            }
            results.push(mixed_result);
        }

        results
    }


    pub fn build_small_expressions(digits: &str, start: usize, end: usize) -> Vec<Expression> {
        let mut expressions = Vec::new();

        if let Ok(num) = digits_to_number(digits, start, end) {
            expressions.push(Expression::Number(num));
        }

        let length = end - start;
        if (2..=SMALL_RANGE_THRESHOLD).contains(&length) {
            Self::generate_small_partitioned_expressions(digits, start, end, &mut expressions);

            let base_len = expressions.len();
            if let Some(slice) = expressions.get(1..base_len) {
                let mut to_negate: Vec<Expression> = Vec::with_capacity(slice.len());
                to_negate.extend(slice.iter().cloned());
                for expr in to_negate {
                    expressions.push(Expression::Neg(Box::new(expr)));
                }
            }
        }

        expressions
    }

    fn generate_small_partitioned_expressions(
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
                            Self::generate_binary_expressions_small_partition(digits, &partition)
                        } else {
                            Self::generate_nary_expressions_small_partition(digits, &partition)
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

    pub fn generate_binary_expressions_small_partition(
        digits: &str,
        partition: &[(usize, usize)],
    ) -> Vec<Expression> {
        let mut out = Vec::new();
        if let (Some(&(start1, end1)), Some(&(start2, end2))) =
            (partition.first(), partition.get(1))
        {
            let left_exprs = Self::get_sub_expressions(digits, start1, end1);
            let right_exprs = Self::get_sub_expressions(digits, start2, end2);

            let mut results: Vec<Expression> = left_exprs
                .par_iter()
                .flat_map_iter(|left| {
                    right_exprs.iter().flat_map(move |right| {
                        let mut local = Self::generate_binary_ops(left, right);
                        if let Some(nth_root) = Self::generate_nth_root(left, right) {
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

    pub fn generate_nary_expressions_small_partition(
        digits: &str,
        partition: &[(usize, usize)],
    ) -> Vec<Expression> {
        let mut all_operands = Vec::new();
        for &(start_i, end_i) in partition {
            let sub_exprs = Self::get_sub_expressions(digits, start_i, end_i);
            if sub_exprs.is_empty() {
                return Vec::new();
            }
            all_operands.push(sub_exprs);
        }

        let mut results = Vec::new();
        Self::generate_nary_combinations_small(&mut results, &all_operands);
        results
    }

    pub fn get_sub_expressions(digits: &str, start: usize, end: usize) -> Vec<Expression> {
        if end - start == 1 {
            if let Ok(num) = digits_to_number(digits, start, end) {
                vec![Expression::Number(num)]
            } else {
                vec![]
            }
        } else {
            Self::build_small_expressions(digits, start, end)
        }
    }

    fn generate_nary_combinations_small(
        expressions: &mut Vec<Expression>,
        all_operands: &[Vec<Expression>],
    ) {
        if all_operands.is_empty() {
            return;
        }

        let mut stack: Vec<(usize, Vec<Expression>)> =
            Vec::with_capacity(all_operands.len().saturating_add(1));
        stack.push((0, Vec::new()));

        while let Some((depth, current_combo)) = stack.pop() {
            if depth == all_operands.len() {
                if current_combo.len() >= 2 {
                    expressions.extend(Self::generate_nary_ops(&current_combo));
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
}
