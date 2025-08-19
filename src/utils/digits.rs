use log::{debug, warn};

use crate::utils::errors::UtilsError;

/// # Errors
///
/// Returns an error if the provided indices are out of bounds or invalid,
/// or if the selected slice cannot be parsed into a numeric value.
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

    if slice.len() > 1 && slice.starts_with('0') {
        debug!("Rejecting number with leading zero: '{}'", slice);
        return Err(UtilsError::InvalidDigitString(slice.to_string()));
    }

    let result = slice
        .parse::<f64>()
        .map_err(|_| UtilsError::InvalidDigitString(slice.to_string()))?;
    debug!("Converted '{}' to {}", slice, result);
    Ok(result)
}
