use thiserror::Error;

/// Errors that can occur in utility functions
#[derive(Error, Debug, Clone, PartialEq)]
pub enum UtilsError {
    #[error("Digit string cannot be empty")]
    EmptyDigitString,
    #[error("Digit string must contain only digits: {0}")]
    InvalidDigitString(String),
    #[error("Invalid range: start={start}, end={end}, length={length}")]
    InvalidRange {
        start: usize,
        end: usize,
        length: usize,
    },
}
