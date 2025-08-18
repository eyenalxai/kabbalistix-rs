use crate::expression::Expression;
use crate::utils::{digits_to_number, generate_partitions};
use log::{info, warn};
use rayon::prelude::*;
use std::collections::VecDeque;

use super::constants::{MAX_WORK_QUEUE_SIZE, SMALL_RANGE_THRESHOLD};
use super::generator::ExpressionGenerator;
use super::state::ExpressionIteratorState;
use super::types::{GenerationState, WorkItem};

#[derive(Debug, Clone)]
struct BinaryOpState {
    start: usize,
    end: usize,
    is_full_range: bool,
    partition_idx: usize,
    left_range: (usize, usize),
    right_range: (usize, usize),
    left_iterator_state: ExpressionIteratorState,
    right_iterator_state: Option<ExpressionIteratorState>,
    current_left: Option<Expression>,
    op_idx: usize,
}

#[derive(Debug, Clone)]
pub struct ExpressionIterator {
    work_queue: VecDeque<WorkItem>,
    digits: String,
}

impl ExpressionIterator {
    /// Add a work item to the queue if within bounds
    fn try_add_work_item(&mut self, item: WorkItem) -> bool {
        if self.work_queue.len() < MAX_WORK_QUEUE_SIZE {
            self.work_queue.push_back(item);
            true
        } else {
            warn!(
                "Work queue size limit reached ({}), skipping further work items to prevent memory exhaustion",
                MAX_WORK_QUEUE_SIZE
            );
            false
        }
    }

    /// Extract operands from a partition
    fn extract_operands(&self, partition: &[(usize, usize)]) -> Option<Vec<Expression>> {
        let mut operands = Vec::new();
        for &(start, end) in partition {
            if let Ok(num) = digits_to_number(&self.digits, start, end) {
                operands.push(Expression::Number(num));
            } else {
                return None;
            }
        }
        Some(operands)
    }

    /// Check if a range should use small expression generation
    fn is_small_range(&self, start: usize, end: usize) -> bool {
        end - start <= SMALL_RANGE_THRESHOLD
    }

    /// Generate the next expression for a given range and iterator state
    fn generate_next_expression(
        &mut self,
        range: (usize, usize),
        state: &mut ExpressionIteratorState,
    ) -> Option<Expression> {
        if state.exhausted {
            return None;
        }

        let (start, end) = range;

        // For small ranges, generate expressions on the fly (no caching)
        if self.is_small_range(start, end) {
            let expressions = self.build_small_expressions(start, end);
            if let Some(expr) = expressions.get(state.position) {
                let expr = expr.clone();
                state.advance();
                return Some(expr);
            } else {
                state.mark_exhausted();
                return None;
            }
        }

        // For larger ranges, generate base number only
        if state.complexity == 1 && state.position == 0 {
            if let Ok(num) = digits_to_number(&self.digits, start, end) {
                state.advance();
                return Some(Expression::Number(num));
            }
        }

        state.mark_exhausted();
        None
    }

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

        Self { work_queue, digits }
    }

    /// Build expressions for small ranges (no caching)
    fn build_small_expressions(&self, start: usize, end: usize) -> Vec<Expression> {
        let mut expressions = Vec::new();

        // Base case: single number
        if let Ok(num) = digits_to_number(&self.digits, start, end) {
            expressions.push(Expression::Number(num));
        }

        let length = end - start;
        if length >= 2 {
            self.generate_small_partitioned_expressions(start, end, &mut expressions);
            self.add_negation_expressions(&mut expressions);
        }

        expressions
    }

    /// Generate expressions from partitions for small ranges
    fn generate_small_partitioned_expressions(
        &self,
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
                            self.generate_binary_expressions_small_partition(&partition)
                        } else {
                            self.generate_nary_expressions_small_partition(&partition)
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

    /// Generate binary expressions for small ranges
    fn generate_binary_expressions_small_partition(
        &self,
        partition: &[(usize, usize)],
    ) -> Vec<Expression> {
        let mut out = Vec::new();
        if let (Some(&(start1, end1)), Some(&(start2, end2))) =
            (partition.first(), partition.get(1))
        {
            let left_exprs = self.get_sub_expressions(start1, end1);
            let right_exprs = self.get_sub_expressions(start2, end2);

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

    /// Generate n-ary expressions for small ranges
    fn generate_nary_expressions_small_partition(
        &self,
        partition: &[(usize, usize)],
    ) -> Vec<Expression> {
        let mut all_operands = Vec::new();
        for &(start_i, end_i) in partition {
            let sub_exprs = self.get_sub_expressions(start_i, end_i);
            if sub_exprs.is_empty() {
                return Vec::new();
            }
            all_operands.push(sub_exprs);
        }

        let mut results = Vec::new();
        self.generate_nary_combinations_small(&mut results, &all_operands, 0, Vec::new());
        results
    }

    /// Get sub-expressions for a range (single number or recursive call)
    fn get_sub_expressions(&self, start: usize, end: usize) -> Vec<Expression> {
        if end - start == 1 {
            if let Ok(num) = digits_to_number(&self.digits, start, end) {
                vec![Expression::Number(num)]
            } else {
                vec![]
            }
        } else {
            self.build_small_expressions(start, end)
        }
    }

    /// Add negation expressions (skip base number and already negated expressions)
    fn add_negation_expressions(&self, expressions: &mut Vec<Expression>) {
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

    // Caching removed for maximal parallel scalability

    /// Generate all combinations of n-ary operations for small ranges
    fn generate_nary_combinations_small(
        &self,
        expressions: &mut Vec<Expression>,
        all_operands: &[Vec<Expression>],
        _depth: usize,
        _current_combo: Vec<Expression>,
    ) {
        let mut stack: Vec<(usize, Vec<Expression>)> = Vec::new();
        stack.push((0, Vec::new()));

        while let Some((depth, current_combo)) = stack.pop() {
            if depth == all_operands.len() {
                if current_combo.len() >= 2 {
                    expressions.extend(ExpressionGenerator::generate_nary_ops(&current_combo));
                }
                continue;
            }

            // Try each expression at this depth level
            if let Some(operands_at_depth) = all_operands.get(depth) {
                for expr in operands_at_depth {
                    let mut new_combo = current_combo.clone();
                    new_combo.push(expr.clone());
                    stack.push((depth + 1, new_combo));
                }
            }
        }
    }

    // Handler methods for cleaner iterator logic

    /// Handle base number generation
    fn handle_base_number(&mut self, item: &WorkItem, is_full_range: bool) -> Option<Expression> {
        if let Ok(num) = digits_to_number(&self.digits, item.start, item.end) {
            let length = item.end - item.start;

            // Queue up partition processing if length >= 2
            if length >= 2 {
                for num_blocks in 2..=std::cmp::min(length, 7) {
                    if !self.try_add_work_item(WorkItem {
                        start: item.start,
                        end: item.end,
                        state: GenerationState::ProcessPartitions {
                            num_blocks,
                            partition_idx: 0,
                        },
                    }) {
                        break;
                    }
                }
            }

            // Only return expressions that use all digits
            if is_full_range {
                return Some(Expression::Number(num));
            }
        }
        None
    }

    /// Handle partition processing
    fn handle_partition_processing(
        &mut self,
        item: &WorkItem,
        num_blocks: usize,
        partition_idx: usize,
    ) {
        let partitions = generate_partitions(item.start, item.end, num_blocks);

        if let Some(partition) = partitions.get(partition_idx) {
            if num_blocks == 2 {
                self.queue_binary_operations(item, partition, partition_idx);
            } else {
                self.queue_nary_operations(item, partition);
            }

            // Queue next partition
            self.try_add_work_item(WorkItem {
                start: item.start,
                end: item.end,
                state: GenerationState::ProcessPartitions {
                    num_blocks,
                    partition_idx: partition_idx + 1,
                },
            });
        } else {
            // Move to negations if this is a small range
            if item.end - item.start <= SMALL_RANGE_THRESHOLD {
                self.try_add_work_item(WorkItem {
                    start: item.start,
                    end: item.end,
                    state: GenerationState::Negations {
                        base_range: (item.start, item.end),
                        base_iterator_state: ExpressionIteratorState::new(),
                        skip_base_number: true,
                    },
                });
            }
        }
    }

    /// Queue binary operations for a partition
    fn queue_binary_operations(
        &mut self,
        item: &WorkItem,
        partition: &[(usize, usize)],
        partition_idx: usize,
    ) {
        if let (Some(&(start1, end1)), Some(&(start2, end2))) =
            (partition.first(), partition.get(1))
        {
            self.try_add_work_item(WorkItem {
                start: item.start,
                end: item.end,
                state: GenerationState::BinaryOps {
                    partition_idx,
                    left_range: (start1, end1),
                    right_range: (start2, end2),
                    left_iterator_state: ExpressionIteratorState::new(),
                    right_iterator_state: None,
                    current_left: None,
                    op_idx: 0,
                },
            });

            // Queue sub-partitions for larger ranges
            self.queue_sub_partitions(start1, end1);
            self.queue_sub_partitions(start2, end2);
        }
    }

    /// Queue n-ary operations for a partition
    fn queue_nary_operations(&mut self, item: &WorkItem, partition: &[(usize, usize)]) {
        if let Some(_operands) = self.extract_operands(partition) {
            self.try_add_work_item(WorkItem {
                start: item.start,
                end: item.end,
                state: GenerationState::NAryOps {
                    partition: partition.to_vec(),
                    op_idx: 0,
                },
            });
        }
    }

    /// Queue sub-partitions for larger ranges
    fn queue_sub_partitions(&mut self, start: usize, end: usize) {
        if end - start > 2 {
            let sub_length = end - start;
            for sub_blocks in 2..=std::cmp::min(sub_length, 7) {
                self.try_add_work_item(WorkItem {
                    start,
                    end,
                    state: GenerationState::ProcessPartitions {
                        num_blocks: sub_blocks,
                        partition_idx: 0,
                    },
                });
            }
        }
    }

    /// Create a binary expression based on operation index
    fn create_binary_expression(
        &self,
        left: &Expression,
        right: &Expression,
        op_idx: usize,
    ) -> Option<Expression> {
        let expr = match op_idx {
            0 => Expression::Add(Box::new(left.clone()), Box::new(right.clone())),
            1 => Expression::Sub(Box::new(left.clone()), Box::new(right.clone())),
            2 => Expression::Mul(Box::new(left.clone()), Box::new(right.clone())),
            3 => Expression::Div(Box::new(left.clone()), Box::new(right.clone())),
            4 => Expression::Pow(Box::new(left.clone()), Box::new(right.clone())),
            5 => ExpressionGenerator::generate_nth_root(left, right)?,
            _ => return None,
        };
        Some(expr)
    }
}

impl Iterator for ExpressionIterator {
    type Item = Expression;

    fn next(&mut self) -> Option<Self::Item> {
        let full_length = self.digits.len();

        while let Some(item) = self.work_queue.pop_front() {
            let is_full_range = item.start == 0 && item.end == full_length;
            let item_start = item.start;
            let item_end = item.end;

            match item.state {
                GenerationState::BaseNumber => {
                    if let Some(result) = self.handle_base_number(&item, is_full_range) {
                        return Some(result);
                    }
                }

                GenerationState::ProcessPartitions {
                    num_blocks,
                    partition_idx,
                } => {
                    self.handle_partition_processing(&item, num_blocks, partition_idx);
                }

                GenerationState::BinaryOps {
                    partition_idx,
                    left_range,
                    right_range,
                    left_iterator_state,
                    right_iterator_state,
                    current_left,
                    op_idx,
                } => {
                    let s = BinaryOpState {
                        start: item_start,
                        end: item_end,
                        is_full_range,
                        partition_idx,
                        left_range,
                        right_range,
                        left_iterator_state,
                        right_iterator_state,
                        current_left,
                        op_idx,
                    };
                    if let Some(result) = self.handle_binary_ops(s) {
                        return Some(result);
                    }
                }

                GenerationState::NAryOps { partition, op_idx } => {
                    if let Some(result) =
                        self.handle_nary_ops(item_start, item_end, is_full_range, partition, op_idx)
                    {
                        return Some(result);
                    }
                }

                GenerationState::Negations {
                    base_range,
                    base_iterator_state,
                    skip_base_number,
                } => {
                    if let Some(result) = self.handle_negations(
                        item_start,
                        item_end,
                        is_full_range,
                        base_range,
                        base_iterator_state,
                        skip_base_number,
                    ) {
                        return Some(result);
                    }
                }
            }
        }
        None
    }
}

impl ExpressionIterator {
    /// Simplified binary operations handler that doesn't need to move WorkItem
    fn handle_binary_ops(&mut self, mut s: BinaryOpState) -> Option<Expression> {
        // Get current left expression if we don't have one
        if s.current_left.is_none() {
            s.current_left =
                self.generate_next_expression(s.left_range, &mut s.left_iterator_state);
            s.current_left.as_ref()?; // Return None if left iterator exhausted
            s.right_iterator_state = Some(ExpressionIteratorState::new());
        }

        // Get next right expression
        if s.right_iterator_state.is_none() {
            s.right_iterator_state = Some(ExpressionIteratorState::new());
        }

        if let Some(ref mut right_state) = s.right_iterator_state {
            if let Some(right) = self.generate_next_expression(s.right_range, right_state) {
                if let Some(ref left) = s.current_left {
                    let expr = self.create_binary_expression(left, &right, s.op_idx)?;

                    // Queue next iteration
                    let is_full = s.is_full_range;
                    self.queue_next_binary_iteration(s);

                    // Only return expressions that use all digits
                    if is_full {
                        return Some(expr);
                    }
                }
            } else {
                // Right iterator exhausted for current operation
                if s.op_idx < 5 {
                    // Try next operation with same left expression
                    self.try_add_work_item(WorkItem {
                        start: s.start,
                        end: s.end,
                        state: GenerationState::BinaryOps {
                            partition_idx: s.partition_idx,
                            left_range: s.left_range,
                            right_range: s.right_range,
                            left_iterator_state: s.left_iterator_state,
                            right_iterator_state: Some(ExpressionIteratorState::new()),
                            current_left: s.current_left,
                            op_idx: s.op_idx + 1,
                        },
                    });
                } else {
                    // All operations exhausted for current left, get next left
                    self.try_add_work_item(WorkItem {
                        start: s.start,
                        end: s.end,
                        state: GenerationState::BinaryOps {
                            partition_idx: s.partition_idx,
                            left_range: s.left_range,
                            right_range: s.right_range,
                            left_iterator_state: s.left_iterator_state,
                            right_iterator_state: None,
                            current_left: None,
                            op_idx: 0,
                        },
                    });
                }
            }
        }
        None
    }

    /// Simplified queue next binary operation iteration
    fn queue_next_binary_iteration(&mut self, s: BinaryOpState) {
        // Continue with same operation and advance right iterator
        self.try_add_work_item(WorkItem {
            start: s.start,
            end: s.end,
            state: GenerationState::BinaryOps {
                partition_idx: s.partition_idx,
                left_range: s.left_range,
                right_range: s.right_range,
                left_iterator_state: s.left_iterator_state,
                right_iterator_state: s.right_iterator_state,
                current_left: s.current_left,
                op_idx: s.op_idx,
            },
        });
    }

    /// Simplified n-ary operations handler
    fn handle_nary_ops(
        &mut self,
        start: usize,
        end: usize,
        is_full_range: bool,
        partition: Vec<(usize, usize)>,
        op_idx: usize,
    ) -> Option<Expression> {
        // Generate operands on-demand from partition
        let operands = self.extract_operands(&partition)?;

        if operands.is_empty() {
            return None;
        }

        let nary_results = ExpressionGenerator::generate_nary_ops(&operands);
        let result = nary_results.get(op_idx)?;

        // Queue next operation type if available
        if op_idx + 1 < nary_results.len() {
            self.try_add_work_item(WorkItem {
                start,
                end,
                state: GenerationState::NAryOps {
                    partition,
                    op_idx: op_idx + 1,
                },
            });
        }

        // Only return expressions that use all digits
        if is_full_range {
            Some(result.clone())
        } else {
            None
        }
    }

    /// Simplified negation operations handler
    fn handle_negations(
        &mut self,
        start: usize,
        end: usize,
        is_full_range: bool,
        base_range: (usize, usize),
        mut base_iterator_state: ExpressionIteratorState,
        skip_base_number: bool,
    ) -> Option<Expression> {
        // Skip the base number if requested
        if skip_base_number {
            let _ = self.generate_next_expression(base_range, &mut base_iterator_state);
        }

        if let Some(expr) = self.generate_next_expression(base_range, &mut base_iterator_state) {
            // Skip already negated expressions
            if matches!(expr, Expression::Neg(_)) {
                // Queue next iteration without returning anything
                if !base_iterator_state.exhausted {
                    self.try_add_work_item(WorkItem {
                        start,
                        end,
                        state: GenerationState::Negations {
                            base_range,
                            base_iterator_state,
                            skip_base_number: false,
                        },
                    });
                }
                return None;
            }

            // Queue next iteration if there are more expressions
            if !base_iterator_state.exhausted {
                self.try_add_work_item(WorkItem {
                    start,
                    end,
                    state: GenerationState::Negations {
                        base_range,
                        base_iterator_state,
                        skip_base_number: false,
                    },
                });
            }

            // Only return expressions that use all digits
            if is_full_range {
                return Some(Expression::Neg(Box::new(expr)));
            }
        }
        None
    }
}
