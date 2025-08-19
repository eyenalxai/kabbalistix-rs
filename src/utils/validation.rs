use log::{debug, warn};

use crate::utils::errors::UtilsError;

/// # Errors
///
/// Returns an error if the string is empty or contains any non-ASCII-digit characters.
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
