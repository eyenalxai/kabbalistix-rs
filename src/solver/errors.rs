use thiserror::Error;

use crate::expression::ExpressionError;
use crate::utils::UtilsError;


#[derive(Error, Debug)]
pub enum SolverError {
    #[error("Expression evaluation error: {0}")]
    ExpressionError(#[from] ExpressionError),
    #[error("Utils error: {0}")]
    UtilsError(#[from] UtilsError),
}
