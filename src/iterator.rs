use crate::expression::Expression;
use crate::utils::{digits_to_number, generate_partitions};
use log::info;

use std::collections::{HashMap, VecDeque};

// Default configuration constants
const MAX_ROOT_DEGREE: f64 = 10.0;
const MAX_CACHE_SIZE: usize = 1000; // Limit cache size to prevent unbounded growth

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

    #[allow(dead_code)]
    fn next_complexity(&mut self) {
        self.complexity += 1;
        self.position = 0;
    }

    fn mark_exhausted(&mut self) {
        self.exhausted = true;
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
    /// Generate binary operations for a specific partition
    BinaryOps {
        partition_idx: usize,
        left_range: (usize, usize),
        right_range: (usize, usize),
        left_iterator_state: ExpressionIteratorState,
        right_iterator_state: Option<ExpressionIteratorState>,
        current_left: Option<Expression>,
        op_idx: usize, // 0=Add, 1=Sub, 2=Mul, 3=Div, 4=Pow
    },

    /// Generate nth root operations for a specific partition
    NthRoots {
        partition_idx: usize,
        n_value: f64,
        a_range: (usize, usize),
        a_iterator_state: ExpressionIteratorState,
    },
    /// Generate negation operations
    Negations {
        base_range: (usize, usize),
        base_iterator_state: ExpressionIteratorState,
        skip_base_number: bool, // Skip the first base number
    },
    /// Generate n-ary operations (for more than 2 operands)
    NAryOps {
        partition_idx: usize,
        partition: Vec<(usize, usize)>,
        op_idx: usize, // 0=Add, 1=Mul, 2=Sub (first - rest), 3=Mixed (first - second + rest)
    },
    /// Initialize binary operations for a partition with specific number of blocks
    InitBinary {
        num_blocks: usize,
        partition_idx: usize,
    },
    /// Initialize nth root operations for a partition
    InitNthRoot { partition_idx: usize },
    /// Initialize negation operations
    InitNegations,
}

impl ExpressionIterator {
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

        // For small ranges, use the existing small cache method
        if end - start <= 4 {
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

        // For larger ranges, generate base number only for now
        // (This is a simplified approach - we could extend this later)
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

    /// Get expressions for small ranges (≤ 4 digits) with caching
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
            // Generate all possible partitions up to the range size
            for num_blocks in 2..=std::cmp::min(length, 4) {
                let partitions = generate_partitions(start, end, num_blocks);

                for partition in &partitions {
                    if num_blocks == 2 {
                        // Binary operations
                        if let (Some(&(start1, end1)), Some(&(start2, end2))) =
                            (partition.first(), partition.get(1))
                        {
                            let left_exprs = if end1 - start1 == 1 {
                                if let Ok(num) = digits_to_number(&self.digits, start1, end1) {
                                    vec![Expression::Number(num)]
                                } else {
                                    vec![]
                                }
                            } else {
                                // Recursively get expressions for this sub-range
                                self.get_small_expressions(start1, end1)
                            };

                            let right_exprs = if end2 - start2 == 1 {
                                if let Ok(num) = digits_to_number(&self.digits, start2, end2) {
                                    vec![Expression::Number(num)]
                                } else {
                                    vec![]
                                }
                            } else {
                                self.get_small_expressions(start2, end2)
                            };

                            // Combine all left and right expressions
                            for left in &left_exprs {
                                for right in &right_exprs {
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

                                    // Nth root if left is a valid root index
                                    if let Expression::Number(n) = left {
                                        if *n >= 2.0 && n.fract() == 0.0 && *n <= MAX_ROOT_DEGREE {
                                            expressions.push(Expression::NthRoot(
                                                Box::new(left.clone()),
                                                Box::new(right.clone()),
                                            ));
                                        }
                                    }
                                }
                            }
                        }
                    } else {
                        // N-ary operations for 3+ operands
                        let mut operands = Vec::new();
                        let mut all_valid = true;

                        for &(start_i, end_i) in partition {
                            if end_i - start_i == 1 {
                                if let Ok(num) = digits_to_number(&self.digits, start_i, end_i) {
                                    operands.push(vec![Expression::Number(num)]);
                                } else {
                                    all_valid = false;
                                    break;
                                }
                            } else {
                                let sub_exprs = self.get_small_expressions(start_i, end_i);
                                if sub_exprs.is_empty() {
                                    all_valid = false;
                                    break;
                                }
                                operands.push(sub_exprs);
                            }
                        }

                        if all_valid {
                            // Generate n-ary combinations
                            self.generate_nary_combinations_small(
                                &mut expressions,
                                &operands,
                                0,
                                Vec::new(),
                            );
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

        // Only cache if we haven't exceeded the size limit
        if self.small_cache.len() < MAX_CACHE_SIZE {
            self.small_cache.insert(key, expressions.clone());
        } else {
            // If cache is full, occasionally clear old entries
            if self.small_cache.len() > MAX_CACHE_SIZE * 2 {
                self.small_cache.clear();
            }
        }
        expressions
    }

    /// Generate all combinations of n-ary operations for small ranges
    #[allow(clippy::only_used_in_recursion)]
    fn generate_nary_combinations_small(
        &self,
        expressions: &mut Vec<Expression>,
        all_operands: &[Vec<Expression>],
        depth: usize,
        current_combo: Vec<Expression>,
    ) {
        if depth == all_operands.len() {
            if current_combo.len() >= 2 {
                if let Some(first) = current_combo.first() {
                    // N-ary addition
                    let mut result = first.clone();
                    for operand in current_combo.iter().skip(1) {
                        result = Expression::Add(Box::new(result), Box::new(operand.clone()));
                    }
                    expressions.push(result);

                    // N-ary multiplication
                    let mut result = first.clone();
                    for operand in current_combo.iter().skip(1) {
                        result = Expression::Mul(Box::new(result), Box::new(operand.clone()));
                    }
                    expressions.push(result);

                    // N-ary subtraction
                    let mut result = first.clone();
                    for operand in current_combo.iter().skip(1) {
                        result = Expression::Sub(Box::new(result), Box::new(operand.clone()));
                    }
                    expressions.push(result);
                }
            }
            return;
        }

        // Try each expression at this depth level
        if let Some(operands_at_depth) = all_operands.get(depth) {
            for expr in operands_at_depth {
                let mut new_combo = current_combo.clone();
                new_combo.push(expr.clone());
                self.generate_nary_combinations_small(
                    expressions,
                    all_operands,
                    depth + 1,
                    new_combo,
                );
            }
        }
    }
}

impl Iterator for ExpressionIterator {
    type Item = Expression;

    fn next(&mut self) -> Option<Self::Item> {
        let full_length = self.digits.len();

        while let Some(item) = self.work_queue.pop_front() {
            let length = item.end - item.start;
            let is_full_range = item.start == 0 && item.end == full_length;

            match item.state {
                GenerationState::BaseNumber => {
                    // Generate base number
                    if let Ok(num) = digits_to_number(&self.digits, item.start, item.end) {
                        // Queue up binary operations if length >= 2
                        if length >= 2 {
                            // Try different numbers of blocks, starting with 2
                            for num_blocks in 2..=std::cmp::min(length, 7) {
                                self.work_queue.push_back(WorkItem {
                                    start: item.start,
                                    end: item.end,
                                    state: GenerationState::InitBinary {
                                        num_blocks,
                                        partition_idx: 0,
                                    },
                                });
                            }
                        }

                        // Only return expressions that use all digits
                        if is_full_range {
                            return Some(Expression::Number(num));
                        }
                    }
                }

                GenerationState::InitBinary {
                    num_blocks,
                    partition_idx,
                } => {
                    let partitions = generate_partitions(item.start, item.end, num_blocks);
                    if let Some(partition) = partitions.get(partition_idx) {
                        if num_blocks == 2 {
                            // Handle binary operations
                            if let (Some(&(start1, end1)), Some(&(start2, end2))) =
                                (partition.first(), partition.get(1))
                            {
                                // Queue binary operations with range info instead of storing expressions
                                self.work_queue.push_back(WorkItem {
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

                                // Also queue work items for the left and right parts if they're large
                                if end1 - start1 > 2 {
                                    let sub_length = end1 - start1;
                                    for sub_blocks in 2..=std::cmp::min(sub_length, 7) {
                                        self.work_queue.push_back(WorkItem {
                                            start: start1,
                                            end: end1,
                                            state: GenerationState::InitBinary {
                                                num_blocks: sub_blocks,
                                                partition_idx: 0,
                                            },
                                        });
                                    }
                                }
                                if end2 - start2 > 2 {
                                    let sub_length = end2 - start2;
                                    for sub_blocks in 2..=std::cmp::min(sub_length, 7) {
                                        self.work_queue.push_back(WorkItem {
                                            start: start2,
                                            end: end2,
                                            state: GenerationState::InitBinary {
                                                num_blocks: sub_blocks,
                                                partition_idx: 0,
                                            },
                                        });
                                    }
                                }
                            }
                        } else {
                            // Handle n-ary operations (num_blocks > 2)
                            let mut operands = Vec::new();
                            let mut all_valid = true;

                            // Collect operands from all blocks in the partition
                            for &(start_i, end_i) in partition {
                                if let Ok(num) = digits_to_number(&self.digits, start_i, end_i) {
                                    operands.push(Expression::Number(num));
                                } else {
                                    all_valid = false;
                                    break;
                                }
                            }

                            // Generate n-ary operations for all valid partitions
                            if all_valid && operands.len() == num_blocks {
                                self.work_queue.push_back(WorkItem {
                                    start: item.start,
                                    end: item.end,
                                    state: GenerationState::NAryOps {
                                        partition_idx,
                                        partition: partition.clone(),
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
                                num_blocks,
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
                    left_range,
                    right_range,
                    mut left_iterator_state,
                    mut right_iterator_state,
                    mut current_left,
                    op_idx,
                } => {
                    // Get current left expression if we don't have one
                    if current_left.is_none() {
                        current_left =
                            self.generate_next_expression(left_range, &mut left_iterator_state);
                        if current_left.is_none() {
                            // Left iterator exhausted, move on
                            continue;
                        }
                        // Reset right iterator for new left expression
                        right_iterator_state = Some(ExpressionIteratorState::new());
                    }

                    // Get next right expression
                    if right_iterator_state.is_none() {
                        right_iterator_state = Some(ExpressionIteratorState::new());
                    }

                    if let Some(ref mut right_state) = right_iterator_state {
                        if let Some(right) = self.generate_next_expression(right_range, right_state)
                        {
                            if let Some(ref left) = current_left {
                                let expr = match op_idx {
                                    0 => Expression::Add(
                                        Box::new(left.clone()),
                                        Box::new(right.clone()),
                                    ),
                                    1 => Expression::Sub(
                                        Box::new(left.clone()),
                                        Box::new(right.clone()),
                                    ),
                                    2 => Expression::Mul(
                                        Box::new(left.clone()),
                                        Box::new(right.clone()),
                                    ),
                                    3 => Expression::Div(
                                        Box::new(left.clone()),
                                        Box::new(right.clone()),
                                    ),
                                    4 => Expression::Pow(
                                        Box::new(left.clone()),
                                        Box::new(right.clone()),
                                    ),
                                    _ => continue,
                                };

                                // Queue next iteration - advance operation or right iterator
                                let next_op_idx = if op_idx < 4 { op_idx + 1 } else { 0 };
                                let continue_with_same_left = if op_idx < 4 {
                                    true
                                } else {
                                    // Check if right iterator has more expressions
                                    !right_state.exhausted
                                };

                                if continue_with_same_left || !left_iterator_state.exhausted {
                                    let next_current_left = if continue_with_same_left {
                                        current_left.clone()
                                    } else {
                                        None // Will get next left expression
                                    };

                                    self.work_queue.push_back(WorkItem {
                                        start: item.start,
                                        end: item.end,
                                        state: GenerationState::BinaryOps {
                                            partition_idx,
                                            left_range,
                                            right_range,
                                            left_iterator_state,
                                            right_iterator_state: if continue_with_same_left {
                                                right_iterator_state
                                            } else {
                                                Some(ExpressionIteratorState::new())
                                            },
                                            current_left: next_current_left,
                                            op_idx: next_op_idx,
                                        },
                                    });
                                }

                                // Only return expressions that use all digits
                                if is_full_range {
                                    return Some(expr);
                                }
                            }
                        } else {
                            // Right iterator exhausted, get next left
                            self.work_queue.push_back(WorkItem {
                                start: item.start,
                                end: item.end,
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

                GenerationState::NAryOps {
                    partition_idx,
                    partition,
                    op_idx,
                } => {
                    // Generate operands on-demand from partition
                    let mut operands = Vec::new();
                    for &(start_i, end_i) in &partition {
                        if let Ok(num) = digits_to_number(&self.digits, start_i, end_i) {
                            operands.push(Expression::Number(num));
                        } else {
                            // Invalid operand, skip this n-ary operation
                            continue;
                        }
                    }

                    if operands.is_empty() {
                        continue;
                    }

                    if op_idx == 0 {
                        // N-ary addition: sum all operands
                        if let Some(first) = operands.first() {
                            let mut result = first.clone();
                            for operand in operands.iter().skip(1) {
                                result =
                                    Expression::Add(Box::new(result), Box::new(operand.clone()));
                            }

                            // Queue next operation type
                            self.work_queue.push_back(WorkItem {
                                start: item.start,
                                end: item.end,
                                state: GenerationState::NAryOps {
                                    partition_idx,
                                    partition: partition.clone(),
                                    op_idx: 1,
                                },
                            });

                            // Only return expressions that use all digits
                            if is_full_range {
                                return Some(result);
                            }
                        }
                    } else if op_idx == 1 {
                        // N-ary multiplication: multiply all operands
                        if let Some(first) = operands.first() {
                            let mut result = first.clone();
                            for operand in operands.iter().skip(1) {
                                result =
                                    Expression::Mul(Box::new(result), Box::new(operand.clone()));
                            }

                            // Queue next operation type (subtraction)
                            self.work_queue.push_back(WorkItem {
                                start: item.start,
                                end: item.end,
                                state: GenerationState::NAryOps {
                                    partition_idx,
                                    partition: partition.clone(),
                                    op_idx: 2,
                                },
                            });

                            // Only return expressions that use all digits
                            if is_full_range {
                                return Some(result);
                            }
                        }
                    } else if op_idx == 2 {
                        // N-ary subtraction: first - (sum of rest)
                        if let Some(first) = operands.first() {
                            let mut result = first.clone();
                            for operand in operands.iter().skip(1) {
                                result =
                                    Expression::Sub(Box::new(result), Box::new(operand.clone()));
                            }

                            // Queue mixed operation (first - second + rest)
                            if operands.len() >= 3 {
                                self.work_queue.push_back(WorkItem {
                                    start: item.start,
                                    end: item.end,
                                    state: GenerationState::NAryOps {
                                        partition_idx,
                                        partition: partition.clone(),
                                        op_idx: 3,
                                    },
                                });
                            }

                            // Only return expressions that use all digits
                            if is_full_range {
                                return Some(result);
                            }
                        }
                    } else if op_idx == 3 {
                        // Mixed operation: first - second + (sum of rest)
                        if operands.len() >= 3 {
                            if let (Some(first), Some(second)) = (operands.first(), operands.get(1))
                            {
                                let mut result = Expression::Sub(
                                    Box::new(first.clone()),
                                    Box::new(second.clone()),
                                );
                                for operand in operands.iter().skip(2) {
                                    result = Expression::Add(
                                        Box::new(result),
                                        Box::new(operand.clone()),
                                    );
                                }
                                // Only return expressions that use all digits
                                if is_full_range {
                                    return Some(result);
                                }
                            }
                        }
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
                                    self.work_queue.push_back(WorkItem {
                                        start: item.start,
                                        end: item.end,
                                        state: GenerationState::NthRoots {
                                            partition_idx,
                                            n_value: n_num,
                                            a_range: (start2, end2),
                                            a_iterator_state: ExpressionIteratorState::new(),
                                        },
                                    });
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
                    a_range,
                    mut a_iterator_state,
                } => {
                    if let Some(a_expr) =
                        self.generate_next_expression(a_range, &mut a_iterator_state)
                    {
                        let n_expr = Expression::Number(n_value);

                        // Queue next iteration if there are more expressions
                        if !a_iterator_state.exhausted {
                            self.work_queue.push_back(WorkItem {
                                start: item.start,
                                end: item.end,
                                state: GenerationState::NthRoots {
                                    partition_idx,
                                    n_value,
                                    a_range,
                                    a_iterator_state,
                                },
                            });
                        }

                        // Only return expressions that use all digits
                        if is_full_range {
                            return Some(Expression::NthRoot(Box::new(n_expr), Box::new(a_expr)));
                        }
                    }
                }

                GenerationState::InitNegations => {
                    // Only negate expressions for small ranges to keep it simple
                    if length <= 4 {
                        self.work_queue.push_back(WorkItem {
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

                GenerationState::Negations {
                    base_range,
                    mut base_iterator_state,
                    skip_base_number,
                } => {
                    // Skip the base number if requested
                    if skip_base_number {
                        // Skip the first expression (base number)
                        let _ = self.generate_next_expression(base_range, &mut base_iterator_state);
                    }

                    if let Some(expr) =
                        self.generate_next_expression(base_range, &mut base_iterator_state)
                    {
                        // Skip already negated expressions
                        if matches!(expr, Expression::Neg(_)) {
                            // Queue next iteration without returning anything
                            if !base_iterator_state.exhausted {
                                self.work_queue.push_back(WorkItem {
                                    start: item.start,
                                    end: item.end,
                                    state: GenerationState::Negations {
                                        base_range,
                                        base_iterator_state,
                                        skip_base_number: false,
                                    },
                                });
                            }
                            continue;
                        }

                        // Queue next iteration if there are more expressions
                        if !base_iterator_state.exhausted {
                            self.work_queue.push_back(WorkItem {
                                start: item.start,
                                end: item.end,
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
                }
            }
        }
        None
    }
}
