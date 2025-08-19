use crate::expression::Expression;
use crate::utils::{digits_to_number, generate_partitions};
use log::{info, warn};
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

    fn is_small_range(&self, start: usize, end: usize) -> bool {
        end - start <= SMALL_RANGE_THRESHOLD
    }

    fn generate_next_expression(
        &mut self,
        range: (usize, usize),
        state: &mut ExpressionIteratorState,
    ) -> Option<Expression> {
        if state.exhausted {
            return None;
        }

        let (start, end) = range;

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

        if state.complexity == 1
            && state.position == 0
            && let Ok(num) = digits_to_number(&self.digits, start, end)
        {
            state.advance();
            return Some(Expression::Number(num));
        }

        state.mark_exhausted();
        None
    }

    pub fn new(digits: String) -> Self {
        let mut work_queue = VecDeque::new();
        let len = digits.len();

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

    pub fn from_digits(digits: &str) -> Self {
        Self::new(digits.to_string())
    }

    fn build_small_expressions(&self, start: usize, end: usize) -> Vec<Expression> {
        ExpressionGenerator::build_small_expressions(&self.digits, start, end)
    }

    fn handle_base_number(&mut self, item: &WorkItem, is_full_range: bool) -> Option<Expression> {
        if let Ok(num) = digits_to_number(&self.digits, item.start, item.end) {
            let length = item.end - item.start;

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

            if is_full_range {
                return Some(Expression::Number(num));
            }
        }
        None
    }

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

            self.try_add_work_item(WorkItem {
                start: item.start,
                end: item.end,
                state: GenerationState::ProcessPartitions {
                    num_blocks,
                    partition_idx: partition_idx + 1,
                },
            });
        } else if item.end - item.start <= SMALL_RANGE_THRESHOLD {
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

            self.queue_sub_partitions(start1, end1);
            self.queue_sub_partitions(start2, end2);
        }
    }

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
    fn handle_binary_ops(&mut self, mut s: BinaryOpState) -> Option<Expression> {
        if s.current_left.is_none() {
            s.current_left =
                self.generate_next_expression(s.left_range, &mut s.left_iterator_state);
            s.current_left.as_ref()?;
            s.right_iterator_state = Some(ExpressionIteratorState::new());
        }

        if s.right_iterator_state.is_none() {
            s.right_iterator_state = Some(ExpressionIteratorState::new());
        }

        if let Some(ref mut right_state) = s.right_iterator_state {
            if let Some(right) = self.generate_next_expression(s.right_range, right_state) {
                if let Some(ref left) = s.current_left {
                    let expr = self.create_binary_expression(left, &right, s.op_idx)?;

                    let is_full = s.is_full_range;
                    self.queue_next_binary_iteration(s);

                    if is_full {
                        return Some(expr);
                    }
                }
            } else if s.op_idx < 5 {
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
        None
    }

    fn queue_next_binary_iteration(&mut self, s: BinaryOpState) {
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

    fn handle_nary_ops(
        &mut self,
        start: usize,
        end: usize,
        is_full_range: bool,
        partition: Vec<(usize, usize)>,
        op_idx: usize,
    ) -> Option<Expression> {
        let operands = self.extract_operands(&partition)?;

        if operands.is_empty() {
            return None;
        }

        let nary_results = ExpressionGenerator::generate_nary_ops(&operands);
        let result = nary_results.get(op_idx)?;

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

        if is_full_range {
            Some(result.clone())
        } else {
            None
        }
    }

    fn handle_negations(
        &mut self,
        start: usize,
        end: usize,
        is_full_range: bool,
        base_range: (usize, usize),
        mut base_iterator_state: ExpressionIteratorState,
        skip_base_number: bool,
    ) -> Option<Expression> {
        if skip_base_number {
            let _ = self.generate_next_expression(base_range, &mut base_iterator_state);
        }

        if let Some(expr) = self.generate_next_expression(base_range, &mut base_iterator_state) {
            if matches!(expr, Expression::Neg(_)) {
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

            if is_full_range {
                return Some(Expression::Neg(Box::new(expr)));
            }
        }
        None
    }
}
