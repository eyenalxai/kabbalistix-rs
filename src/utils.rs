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

/// Generate all possible ways to partition a range of digits into consecutive blocks
pub fn generate_partitions(
    start: usize,
    end: usize,
    num_blocks: usize,
) -> Vec<Vec<(usize, usize)>> {
    if num_blocks == 1 {
        return vec![vec![(start, end)]];
    }

    let mut result = Vec::new();
    for split_point in start + 1..end {
        let first_block = (start, split_point);
        for mut rest in generate_partitions(split_point, end, num_blocks - 1) {
            let mut partition = vec![first_block];
            partition.append(&mut rest);
            result.push(partition);
        }
    }
    result
}

/// Convert a digit range to a number
pub fn digits_to_number(digits: &str, start: usize, end: usize) -> Result<f64, UtilsError> {
    if start >= digits.len() || end > digits.len() || start >= end {
        return Err(UtilsError::InvalidRange {
            start,
            end,
            length: digits.len(),
        });
    }

    let slice = digits.get(start..end).ok_or(UtilsError::InvalidRange {
        start,
        end,
        length: digits.len(),
    })?;

    Ok(slice.parse::<f64>().unwrap_or(0.0))
}

/// Validates that a string contains only ASCII digits
pub fn validate_digit_string(digit_string: &str) -> Result<(), UtilsError> {
    if digit_string.is_empty() {
        return Err(UtilsError::EmptyDigitString);
    }

    if !digit_string.chars().all(|c| c.is_ascii_digit()) {
        return Err(UtilsError::InvalidDigitString(digit_string.to_string()));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_partitions_single_block() {
        let partitions = generate_partitions(0, 3, 1);
        assert_eq!(partitions, vec![vec![(0, 3)]]);
    }

    #[test]
    fn test_generate_partitions_two_blocks() {
        let partitions = generate_partitions(0, 3, 2);
        let expected = vec![vec![(0, 1), (1, 3)], vec![(0, 2), (2, 3)]];
        assert_eq!(partitions, expected);
    }

    #[test]
    fn test_digits_to_number() {
        let result = digits_to_number("12345", 0, 3);
        assert!(result.is_ok());
        if let Ok(value) = result {
            assert_eq!(value, 123.0);
        }

        let result = digits_to_number("12345", 2, 5);
        assert!(result.is_ok());
        if let Ok(value) = result {
            assert_eq!(value, 345.0);
        }

        let result = digits_to_number("12345", 1, 4);
        assert!(result.is_ok());
        if let Ok(value) = result {
            assert_eq!(value, 234.0);
        }
    }

    #[test]
    fn test_digits_to_number_invalid_range() {
        let result = digits_to_number("12345", 0, 10);
        assert!(result.is_err());

        let result = digits_to_number("12345", 5, 3);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_digit_string_valid() {
        assert!(validate_digit_string("12345").is_ok());
        assert!(validate_digit_string("0").is_ok());
        assert!(validate_digit_string("999").is_ok());
    }

    #[test]
    fn test_validate_digit_string_invalid() {
        assert!(validate_digit_string("").is_err());
        assert!(validate_digit_string("12a45").is_err());
        assert!(validate_digit_string("12.45").is_err());
        assert!(validate_digit_string("12-45").is_err());
    }
}
