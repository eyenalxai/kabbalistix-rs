use thiserror::Error;

#[derive(Error, Debug, Clone, PartialEq)]
pub enum ExpressionError {
    #[error("Division by zero")]
    DivisionByZero,
    #[error("Complex result from negative base with fractional exponent")]
    ComplexResult,
    #[error("Root index must be an integer >= 2")]
    InvalidRootIndex,
    #[error("Even root of negative number")]
    EvenRootOfNegative,
}
