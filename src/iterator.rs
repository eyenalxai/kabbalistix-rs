use crate::expression::Expression;
use crate::utils::{digits_to_number, generate_partitions};
use log::info;

use std::collections::{HashMap, VecDeque};

// Configuration constants
const MAX_ROOT_DEGREE: f64 = 10.0;
const MAX_CACHE_SIZE: usize = 1000;
const MAX_WORK_QUEUE_SIZE: usize = 100_000;
const SMALL_RANGE_THRESHOLD: usize = 4;

/// Lightweight state for generating expressions from a range without storing them
#[derive(Debug, Clone)]
pub struct ExpressionIteratorState {
    /// Current complexity level being generated (1=base numbers, 2=binary ops, etc.)
    complexity: usize,
    /// Position within current complexity level
    position: usize,
    /// Whether this iterator is exhausted
    exhausted: bool,
}

impl ExpressionIteratorState {
    fn new() -> Self {
        Self {
            complexity: 1,
            position: 0,
            exhausted: false,
        }
    }

    fn advance(&mut self) {
        self.position += 1;
    }

    fn mark_exhausted(&mut self) {
        self.exhausted = true;
    }
}

/// Helper for generating expressions from operands
struct ExpressionGenerator;

impl ExpressionGenerator {
    /// Generate binary operations from two expressions
    fn generate_binary_ops(left: &Expression, right: &Expression) -> Vec<Expression> {
        vec![
            Expression::Add(Box::new(left.clone()), Box::new(right.clone())),
            Expression::Sub(Box::new(left.clone()), Box::new(right.clone())),
            Expression::Mul(Box::new(left.clone()), Box::new(right.clone())),
            Expression::Div(Box::new(left.clone()), Box::new(right.clone())),
            Expression::Pow(Box::new(left.clone()), Box::new(right.clone())),
        ]
    }

    /// Generate nth root if left is a valid root index
    fn generate_nth_root(n: &Expression, a: &Expression) -> Option<Expression> {
        if let Expression::Number(n_val) = n {
            if *n_val >= 2.0 && n_val.fract() == 0.0 && *n_val <= MAX_ROOT_DEGREE {
                return Some(Expression::NthRoot(
                    Box::new(n.clone()),
                    Box::new(a.clone()),
                ));
            }
        }
        None
    }

    /// Generate n-ary operations from a list of operands
    fn generate_nary_ops(operands: &[Expression]) -> Vec<Expression> {
        if operands.len() < 2 {
            return Vec::new();
        }

        let mut results = Vec::new();
        let first = operands.first().unwrap_or(&Expression::Number(0.0));

        // N-ary addition
        let mut add_result = first.clone();
        for operand in operands.iter().skip(1) {
            add_result = Expression::Add(Box::new(add_result), Box::new(operand.clone()));
        }
        results.push(add_result);

        // N-ary multiplication
        let mut mul_result = first.clone();
        for operand in operands.iter().skip(1) {
            mul_result = Expression::Mul(Box::new(mul_result), Box::new(operand.clone()));
        }
        results.push(mul_result);

        // N-ary subtraction (first - rest)
        let mut sub_result = first.clone();
        for operand in operands.iter().skip(1) {
            sub_result = Expression::Sub(Box::new(sub_result), Box::new(operand.clone()));
        }
        results.push(sub_result);

        // Mixed operation (first - second + rest) for 3+ operands
        if operands.len() >= 3 {
            let mut mixed_result = Expression::Sub(
                Box::new(first.clone()),
                Box::new(operands.get(1).unwrap_or(&Expression::Number(0.0)).clone()),
            );
            for operand in operands.iter().skip(2) {
                mixed_result = Expression::Add(Box::new(mixed_result), Box::new(operand.clone()));
            }
            results.push(mixed_result);
        }

        results
    }
}

/// Iterative expression generator that yields expressions one at a time without storing them
#[derive(Debug, Clone)]
pub struct ExpressionIterator {
    work_queue: VecDeque<WorkItem>,
    digits: String,
    small_cache: HashMap<(usize, usize), Vec<Expression>>, // Cache for small ranges only
}

#[derive(Debug, Clone)]
pub struct WorkItem {
    pub start: usize,
    pub end: usize,
    pub state: GenerationState,
}

#[derive(Debug, Clone)]
pub enum GenerationState {
    /// Generate the base number
    BaseNumber,
    /// Process partitions for a given number of blocks
    ProcessPartitions {
        num_blocks: usize,
        partition_idx: usize,
    },
    /// Generate binary operations
    BinaryOps {
        partition_idx: usize,
        left_range: (usize, usize),
        right_range: (usize, usize),
        left_iterator_state: ExpressionIteratorState,
        right_iterator_state: Option<ExpressionIteratorState>,
        current_left: Option<Expression>,
        op_idx: usize, // 0=Add, 1=Sub, 2=Mul, 3=Div, 4=Pow, 5=NthRoot
    },
    /// Generate n-ary operations
    NAryOps {
        partition: Vec<(usize, usize)>,
        op_idx: usize, // 0=Add, 1=Mul, 2=Sub, 3=Mixed
    },
    /// Generate negation operations
    Negations {
        base_range: (usize, usize),
        base_iterator_state: ExpressionIteratorState,
        skip_base_number: bool,
    },
}

impl ExpressionIterator {
    /// Add a work item to the queue if within bounds
    fn try_add_work_item(&mut self, item: WorkItem) -> bool {
        if self.work_queue.len() < MAX_WORK_QUEUE_SIZE {
            self.work_queue.push_back(item);
            true
        } else {
            log::warn!(
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

        // For small ranges, use cached expressions
        if self.is_small_range(start, end) {
            let expressions = self.get_small_expressions(start, end);
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

        Self {
            work_queue,
            digits,
            small_cache: HashMap::new(),
        }
    }

    /// Get expressions for small ranges with caching
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
        if length >= 2 {
            self.generate_small_partitioned_expressions(start, end, &mut expressions);
            self.add_negation_expressions(&mut expressions);
        }

        self.cache_expressions(key, &expressions);
        expressions
    }

    /// Generate expressions from partitions for small ranges
    fn generate_small_partitioned_expressions(
        &mut self,
        start: usize,
        end: usize,
        expressions: &mut Vec<Expression>,
    ) {
        let length = end - start;
        for num_blocks in 2..=std::cmp::min(length, 4) {
            let partitions = generate_partitions(start, end, num_blocks);

            for partition in &partitions {
                if num_blocks == 2 {
                    self.generate_binary_expressions_small(partition, expressions);
                } else {
                    self.generate_nary_expressions_small(partition, expressions);
                }
            }
        }
    }

    /// Generate binary expressions for small ranges
    fn generate_binary_expressions_small(
        &mut self,
        partition: &[(usize, usize)],
        expressions: &mut Vec<Expression>,
    ) {
        if let (Some(&(start1, end1)), Some(&(start2, end2))) =
            (partition.first(), partition.get(1))
        {
            let left_exprs = self.get_sub_expressions(start1, end1);
            let right_exprs = self.get_sub_expressions(start2, end2);

            for left in &left_exprs {
                for right in &right_exprs {
                    // Generate all binary operations
                    expressions.extend(ExpressionGenerator::generate_binary_ops(left, right));

                    // Generate nth root if applicable
                    if let Some(nth_root) = ExpressionGenerator::generate_nth_root(left, right) {
                        expressions.push(nth_root);
                    }
                }
            }
        }
    }

    /// Generate n-ary expressions for small ranges
    fn generate_nary_expressions_small(
        &mut self,
        partition: &[(usize, usize)],
        expressions: &mut Vec<Expression>,
    ) {
        let mut all_operands = Vec::new();
        for &(start_i, end_i) in partition {
            let sub_exprs = self.get_sub_expressions(start_i, end_i);
            if sub_exprs.is_empty() {
                return; // Invalid partition
            }
            all_operands.push(sub_exprs);
        }

        self.generate_nary_combinations_small(expressions, &all_operands, 0, Vec::new());
    }

    /// Get sub-expressions for a range (single number or recursive call)
    fn get_sub_expressions(&mut self, start: usize, end: usize) -> Vec<Expression> {
        if end - start == 1 {
            if let Ok(num) = digits_to_number(&self.digits, start, end) {
                vec![Expression::Number(num)]
            } else {
                vec![]
            }
        } else {
            self.get_small_expressions(start, end)
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

    /// Cache expressions with size limit management
    fn cache_expressions(&mut self, key: (usize, usize), expressions: &[Expression]) {
        if self.small_cache.len() < MAX_CACHE_SIZE {
            self.small_cache.insert(key, expressions.to_vec());
        } else if self.small_cache.len() > MAX_CACHE_SIZE * 2 {
            self.small_cache.clear();
        }
    }

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
                    if let Some(result) = self.handle_binary_ops_simple(
                        item_start,
                        item_end,
                        is_full_range,
                        partition_idx,
                        left_range,
                        right_range,
                        left_iterator_state,
                        right_iterator_state,
                        current_left,
                        op_idx,
                    ) {
                        return Some(result);
                    }
                }

                GenerationState::NAryOps { partition, op_idx } => {
                    if let Some(result) = self.handle_nary_ops_simple(
                        item_start,
                        item_end,
                        is_full_range,
                        partition,
                        op_idx,
                    ) {
                        return Some(result);
                    }
                }

                GenerationState::Negations {
                    base_range,
                    base_iterator_state,
                    skip_base_number,
                } => {
                    if let Some(result) = self.handle_negations_simple(
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
    fn handle_binary_ops_simple(
        &mut self,
        start: usize,
        end: usize,
        is_full_range: bool,
        partition_idx: usize,
        left_range: (usize, usize),
        right_range: (usize, usize),
        mut left_iterator_state: ExpressionIteratorState,
        mut right_iterator_state: Option<ExpressionIteratorState>,
        mut current_left: Option<Expression>,
        op_idx: usize,
    ) -> Option<Expression> {
        // Get current left expression if we don't have one
        if current_left.is_none() {
            current_left = self.generate_next_expression(left_range, &mut left_iterator_state);
            current_left.as_ref()?; // Return None if left iterator exhausted
            right_iterator_state = Some(ExpressionIteratorState::new());
        }

        // Get next right expression
        if right_iterator_state.is_none() {
            right_iterator_state = Some(ExpressionIteratorState::new());
        }

        if let Some(ref mut right_state) = right_iterator_state {
            if let Some(right) = self.generate_next_expression(right_range, right_state) {
                if let Some(ref left) = current_left {
                    let expr = self.create_binary_expression(left, &right, op_idx)?;

                    // Queue next iteration
                    let right_state_exhausted = right_state.exhausted;
                    self.queue_next_binary_iteration_simple(
                        start,
                        end,
                        partition_idx,
                        left_range,
                        right_range,
                        left_iterator_state,
                        right_iterator_state.clone(),
                        current_left.clone(),
                        op_idx,
                        right_state_exhausted,
                    );

                    // Only return expressions that use all digits
                    if is_full_range {
                        return Some(expr);
                    }
                }
            } else {
                // Right iterator exhausted for current operation
                if op_idx < 5 {
                    // Try next operation with same left expression
                    self.try_add_work_item(WorkItem {
                        start,
                        end,
                        state: GenerationState::BinaryOps {
                            partition_idx,
                            left_range,
                            right_range,
                            left_iterator_state,
                            right_iterator_state: Some(ExpressionIteratorState::new()),
                            current_left,
                            op_idx: op_idx + 1,
                        },
                    });
                } else {
                    // All operations exhausted for current left, get next left
                    self.try_add_work_item(WorkItem {
                        start,
                        end,
                        state: GenerationState::BinaryOps {
                            partition_idx,
                            left_range,
                            right_range,
                            left_iterator_state,
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
    fn queue_next_binary_iteration_simple(
        &mut self,
        start: usize,
        end: usize,
        partition_idx: usize,
        left_range: (usize, usize),
        right_range: (usize, usize),
        left_iterator_state: ExpressionIteratorState,
        right_iterator_state: Option<ExpressionIteratorState>,
        current_left: Option<Expression>,
        op_idx: usize,
        _right_state_exhausted: bool,
    ) {
        // Continue with same operation and advance right iterator
        self.try_add_work_item(WorkItem {
            start,
            end,
            state: GenerationState::BinaryOps {
                partition_idx,
                left_range,
                right_range,
                left_iterator_state,
                right_iterator_state,
                current_left,
                op_idx,
            },
        });
    }

    /// Simplified n-ary operations handler
    fn handle_nary_ops_simple(
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
    fn handle_negations_simple(
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
