#[derive(Debug, Clone)]
pub struct ExpressionIteratorState {
    pub(crate) complexity: usize,
    pub(crate) position: usize,
    pub(crate) exhausted: bool,
}

impl ExpressionIteratorState {
    pub fn new() -> Self {
        Self {
            complexity: 1,
            position: 0,
            exhausted: false,
        }
    }

    pub fn advance(&mut self) {
        self.position += 1;
    }
    pub fn mark_exhausted(&mut self) {
        self.exhausted = true;
    }
}

impl Default for ExpressionIteratorState {
    fn default() -> Self {
        Self::new()
    }
}
