use crate::expression::Expression;

use super::state::ExpressionIteratorState;

#[derive(Debug, Clone)]
pub struct WorkItem {
    pub start: usize,
    pub end: usize,
    pub state: GenerationState,
}

#[derive(Debug, Clone)]
pub enum GenerationState {
    BaseNumber,
    ProcessPartitions {
        num_blocks: usize,
        partition_idx: usize,
    },
    BinaryOps {
        partition_idx: usize,
        left_range: (usize, usize),
        right_range: (usize, usize),
        left_iterator_state: ExpressionIteratorState,
        right_iterator_state: Option<ExpressionIteratorState>,
        current_left: Option<Expression>,
        op_idx: usize,
    },
    NAryOps {
        partition: Vec<(usize, usize)>,
        op_idx: usize,
    },
    Negations {
        base_range: (usize, usize),
        base_iterator_state: ExpressionIteratorState,
        skip_base_number: bool,
    },
}
