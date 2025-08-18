use crate::expression::Expression;
use crate::utils::{digits_to_number, generate_partitions};
use log::info;

use std::collections::{HashMap, VecDeque};

// Default configuration constants
const MAX_ROOT_DEGREE: f64 = 10.0;

/// Iterative expression generator that yields expressions one at a time without storing them
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
        left_exprs: Vec<Expression>,
        right_exprs: Vec<Expression>,
        left_idx: usize,
        right_idx: usize,
        op_idx: usize, // 0=Add, 1=Sub, 2=Mul, 3=Div, 4=Pow
    },
    /// Generate nth root operations for a specific partition
    NthRoots {
        partition_idx: usize,
        n_value: f64,
        a_exprs: Vec<Expression>,
        a_idx: usize,
    },
    /// Generate negation operations
    Negations {
        base_exprs: Vec<Expression>,
        expr_idx: usize,
    },
    /// Generate n-ary operations (for more than 2 operands)
    NAryOps {
        partition_idx: usize,
        operands: Vec<Expression>,
        op_idx: usize, // 0=Add, 1=Mul (only commutative ops make sense for n-ary)
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

    /// Get expressions for small ranges (≤ 2 digits) with caching
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
        if length == 2 {
            let partitions = generate_partitions(start, end, 2);

            // Binary operations for 2-digit combinations
            for partition in &partitions {
                if let (Some(&(start1, end1)), Some(&(start2, end2))) =
                    (partition.first(), partition.get(1))
                {
                    if let (Ok(left_num), Ok(right_num)) = (
                        digits_to_number(&self.digits, start1, end1),
                        digits_to_number(&self.digits, start2, end2),
                    ) {
                        let left = Expression::Number(left_num);
                        let right = Expression::Number(right_num);

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

                        // Nth root if applicable
                        if left_num >= 2.0 && left_num.fract() == 0.0 && left_num <= MAX_ROOT_DEGREE
                        {
                            expressions.push(Expression::NthRoot(
                                Box::new(left.clone()),
                                Box::new(right.clone()),
                            ));
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

        self.small_cache.insert(key, expressions.clone());
        expressions
    }
}

impl Iterator for ExpressionIterator {
    type Item = Expression;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(item) = self.work_queue.pop_front() {
            let length = item.end - item.start;

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
                        return Some(Expression::Number(num));
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
                                // Get expressions for left and right parts
                                let left_exprs = if end1 - start1 <= 2 {
                                    self.get_small_expressions(start1, end1)
                                } else {
                                    // For larger ranges, generate a single base number expression
                                    // The recursive partitioning will be handled by separate work items
                                    if let Ok(num) = digits_to_number(&self.digits, start1, end1) {
                                        vec![Expression::Number(num)]
                                    } else {
                                        vec![]
                                    }
                                };

                                let right_exprs = if end2 - start2 <= 2 {
                                    self.get_small_expressions(start2, end2)
                                } else {
                                    // For larger ranges, generate a single base number expression
                                    if let Ok(num) = digits_to_number(&self.digits, start2, end2) {
                                        vec![Expression::Number(num)]
                                    } else {
                                        vec![]
                                    }
                                };

                                if !left_exprs.is_empty() && !right_exprs.is_empty() {
                                    // Queue binary operations
                                    self.work_queue.push_back(WorkItem {
                                        start: item.start,
                                        end: item.end,
                                        state: GenerationState::BinaryOps {
                                            partition_idx,
                                            left_exprs,
                                            right_exprs,
                                            left_idx: 0,
                                            right_idx: 0,
                                            op_idx: 0,
                                        },
                                    });
                                }

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
                            let mut all_single_digits = true;

                            // Collect operands from all blocks in the partition
                            for &(start_i, end_i) in partition {
                                if end_i - start_i == 1 {
                                    // Single digit - convert directly to number
                                    if let Ok(num) = digits_to_number(&self.digits, start_i, end_i)
                                    {
                                        operands.push(Expression::Number(num));
                                    }
                                } else {
                                    all_single_digits = false;
                                    break;
                                }
                            }

                            // Only generate n-ary operations if all operands are single digits
                            if all_single_digits && operands.len() == num_blocks {
                                self.work_queue.push_back(WorkItem {
                                    start: item.start,
                                    end: item.end,
                                    state: GenerationState::NAryOps {
                                        partition_idx,
                                        operands,
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
                    left_exprs,
                    right_exprs,
                    left_idx,
                    right_idx,
                    op_idx,
                } => {
                    if let (Some(left), Some(right)) =
                        (left_exprs.get(left_idx), right_exprs.get(right_idx))
                    {
                        let expr = match op_idx {
                            0 => Expression::Add(Box::new(left.clone()), Box::new(right.clone())),
                            1 => Expression::Sub(Box::new(left.clone()), Box::new(right.clone())),
                            2 => Expression::Mul(Box::new(left.clone()), Box::new(right.clone())),
                            3 => Expression::Div(Box::new(left.clone()), Box::new(right.clone())),
                            4 => Expression::Pow(Box::new(left.clone()), Box::new(right.clone())),
                            _ => continue,
                        };

                        // Update indices for next iteration
                        let (next_left_idx, next_right_idx, next_op_idx) = if op_idx < 4 {
                            (left_idx, right_idx, op_idx + 1)
                        } else if right_idx + 1 < right_exprs.len() {
                            (left_idx, right_idx + 1, 0)
                        } else if left_idx + 1 < left_exprs.len() {
                            (left_idx + 1, 0, 0)
                        } else {
                            // Done with this partition, don't queue anything more
                            (left_exprs.len(), right_exprs.len(), 5)
                        };

                        // Queue next iteration if not done
                        if next_left_idx < left_exprs.len()
                            && next_right_idx < right_exprs.len()
                            && next_op_idx <= 4
                        {
                            self.work_queue.push_back(WorkItem {
                                start: item.start,
                                end: item.end,
                                state: GenerationState::BinaryOps {
                                    partition_idx,
                                    left_exprs: left_exprs.clone(),
                                    right_exprs: right_exprs.clone(),
                                    left_idx: next_left_idx,
                                    right_idx: next_right_idx,
                                    op_idx: next_op_idx,
                                },
                            });
                        }

                        return Some(expr);
                    }
                }

                GenerationState::NAryOps {
                    partition_idx,
                    operands,
                    op_idx,
                } => {
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
                                    operands,
                                    op_idx: 1,
                                },
                            });

                            return Some(result);
                        }
                    } else if op_idx == 1 {
                        // N-ary multiplication: multiply all operands
                        if let Some(first) = operands.first() {
                            let mut result = first.clone();
                            for operand in operands.iter().skip(1) {
                                result =
                                    Expression::Mul(Box::new(result), Box::new(operand.clone()));
                            }
                            return Some(result);
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
                                    let a_exprs = if end2 - start2 <= 2 {
                                        self.get_small_expressions(start2, end2)
                                    } else {
                                        vec![]
                                    };

                                    if !a_exprs.is_empty() {
                                        self.work_queue.push_back(WorkItem {
                                            start: item.start,
                                            end: item.end,
                                            state: GenerationState::NthRoots {
                                                partition_idx,
                                                n_value: n_num,
                                                a_exprs,
                                                a_idx: 0,
                                            },
                                        });
                                    }
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
                    a_exprs,
                    a_idx,
                } => {
                    if let Some(a_expr) = a_exprs.get(a_idx) {
                        let n_expr = Expression::Number(n_value);

                        // Queue next iteration
                        if a_idx + 1 < a_exprs.len() {
                            self.work_queue.push_back(WorkItem {
                                start: item.start,
                                end: item.end,
                                state: GenerationState::NthRoots {
                                    partition_idx,
                                    n_value,
                                    a_exprs: a_exprs.clone(),
                                    a_idx: a_idx + 1,
                                },
                            });
                        }

                        return Some(Expression::NthRoot(
                            Box::new(n_expr),
                            Box::new(a_expr.clone()),
                        ));
                    }
                }

                GenerationState::InitNegations => {
                    // Get base expressions to negate (simplified - only from small cache for now)
                    let base_exprs = if length <= 2 {
                        let mut all_exprs = self.get_small_expressions(item.start, item.end);
                        // Remove the base number and already negated expressions
                        all_exprs.retain(|expr| {
                            !matches!(expr, Expression::Number(_) | Expression::Neg(_))
                        });
                        all_exprs
                    } else {
                        vec![]
                    };

                    if !base_exprs.is_empty() {
                        self.work_queue.push_back(WorkItem {
                            start: item.start,
                            end: item.end,
                            state: GenerationState::Negations {
                                base_exprs,
                                expr_idx: 0,
                            },
                        });
                    }
                }

                GenerationState::Negations {
                    base_exprs,
                    expr_idx,
                } => {
                    if let Some(expr) = base_exprs.get(expr_idx) {
                        // Queue next iteration
                        if expr_idx + 1 < base_exprs.len() {
                            self.work_queue.push_back(WorkItem {
                                start: item.start,
                                end: item.end,
                                state: GenerationState::Negations {
                                    base_exprs: base_exprs.clone(),
                                    expr_idx: expr_idx + 1,
                                },
                            });
                        }

                        return Some(Expression::Neg(Box::new(expr.clone())));
                    }
                }
            }
        }
        None
    }
}
