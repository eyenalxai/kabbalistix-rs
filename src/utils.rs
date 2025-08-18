use log::{debug, warn};
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
///
/// This uses an iterative approach to avoid stack overflow with large ranges.
pub fn generate_partitions(
    start: usize,
    end: usize,
    num_blocks: usize,
) -> Vec<Vec<(usize, usize)>> {
    debug!(
        "Generating partitions for range {}..{} with {} blocks",
        start, end, num_blocks
    );

    if num_blocks == 1 {
        return vec![vec![(start, end)]];
    }

    if num_blocks > (end - start) {
        // Can't have more blocks than positions
        return vec![];
    }

    let mut result = Vec::new();

    // Use iterative approach with a stack to avoid recursion
    // Stack contains: (current_partition, remaining_start, remaining_blocks)
    let mut stack = Vec::new();
    stack.push((Vec::new(), start, num_blocks));

    while let Some((current_partition, remaining_start, remaining_blocks)) = stack.pop() {
        if remaining_blocks == 1 {
            // Last block takes the rest of the range
            let mut partition = current_partition;
            partition.push((remaining_start, end));
            result.push(partition);
            continue;
        }

        // Try all possible split points for the next block
        // Ensure we leave enough positions for the remaining blocks
        let min_end = remaining_start + 1;
        let max_end = end - (remaining_blocks - 1); // Leave at least 1 position per remaining block

        for split_point in min_end..=max_end {
            let mut new_partition = current_partition.clone();
            new_partition.push((remaining_start, split_point));
            stack.push((new_partition, split_point, remaining_blocks - 1));
        }
    }

    debug!("Generated {} partitions", result.len());
    result
}

/// Convert a digit range to a number
///
/// # Errors
///
/// This function will return an error if:
/// * The range is invalid (start >= end, or out of bounds)
/// * The digit slice cannot be parsed as a valid number
pub fn digits_to_number(digits: &str, start: usize, end: usize) -> Result<f64, UtilsError> {
    debug!("Converting digits[{}..{}] from '{}'", start, end, digits);

    if start >= digits.len() || end > digits.len() || start >= end {
        warn!(
            "Invalid range: start={}, end={}, length={}",
            start,
            end,
            digits.len()
        );
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

    let result = slice
        .parse::<f64>()
        .map_err(|_| UtilsError::InvalidDigitString(slice.to_string()))?;
    debug!("Converted '{}' to {}", slice, result);
    Ok(result)
}

/// Validates that a string contains only ASCII digits
///
/// # Errors
///
/// This function will return an error if:
/// * The digit string is empty
/// * The digit string contains non-ASCII-digit characters
pub fn validate_digit_string(digit_string: &str) -> Result<(), UtilsError> {
    debug!("Validating digit string: '{}'", digit_string);

    if digit_string.is_empty() {
        warn!("Digit string is empty");
        return Err(UtilsError::EmptyDigitString);
    }

    if !digit_string.chars().all(|c| c.is_ascii_digit()) {
        warn!(
            "Digit string contains non-digit characters: '{}'",
            digit_string
        );
        return Err(UtilsError::InvalidDigitString(digit_string.to_string()));
    }

    debug!("Digit string validation successful");
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
        let expected = vec![vec![(0, 2), (2, 3)], vec![(0, 1), (1, 3)]];
        assert_eq!(partitions, expected);
    }

    #[test]
    fn test_generate_partitions_three_blocks() {
        let partitions = generate_partitions(0, 4, 3);
        let expected = vec![
            vec![(0, 2), (2, 3), (3, 4)],
            vec![(0, 1), (1, 3), (3, 4)],
            vec![(0, 1), (1, 2), (2, 4)],
        ];
        assert_eq!(partitions, expected);
    }

    #[test]
    fn test_generate_partitions_impossible() {
        // Can't partition 3 positions into 4 blocks
        let partitions = generate_partitions(0, 3, 4);
        assert_eq!(partitions, Vec::<Vec<(usize, usize)>>::new());
    }

    #[test]
    fn test_generate_partitions_large_range() {
        // Test that large ranges don't cause stack overflow
        let partitions = generate_partitions(0, 20, 5);
        assert!(!partitions.is_empty());

        // Verify all partitions are valid
        for partition in &partitions {
            assert_eq!(partition.len(), 5);
            if let (Some(first), Some(last)) = (partition.first(), partition.last()) {
                assert_eq!(first.0, 0);
                assert_eq!(last.1, 20);
            }

            // Check continuity
            for i in 0..4 {
                if let (Some(current), Some(next)) = (partition.get(i), partition.get(i + 1)) {
                    assert_eq!(current.1, next.0);
                }
            }
        }
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
