use log::debug;
use std::fmt;
use thiserror::Error;

/// Errors that can occur during expression evaluation
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

/// Represents mathematical expressions that can be built from digits
#[derive(Debug, Clone)]
pub enum Expression {
    Number(f64),
    Add(Box<Expression>, Box<Expression>),
    Sub(Box<Expression>, Box<Expression>),
    Mul(Box<Expression>, Box<Expression>),
    Div(Box<Expression>, Box<Expression>),
    Pow(Box<Expression>, Box<Expression>),
    Neg(Box<Expression>),
    NthRoot(Box<Expression>, Box<Expression>), // NthRoot(n, a) = a^(1/n)
}

impl fmt::Display for Expression {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Expression::Number(n) => write!(f, "{}", n),
            Expression::Add(l, r) => write!(f, "({} + {})", l, r),
            Expression::Sub(l, r) => write!(f, "({} - {})", l, r),
            Expression::Mul(l, r) => write!(f, "({} × {})", l, r),
            Expression::Div(l, r) => write!(f, "({} ÷ {})", l, r),
            Expression::Pow(l, r) => write!(f, "({} ^ {})", l, r),
            Expression::Neg(e) => write!(f, "(-{})", e),
            Expression::NthRoot(n, a) => write!(f, "√{}({})", n, a),
        }
    }
}

/// Check if a floating-point number is effectively zero
fn is_zero(value: f64) -> bool {
    value.abs() < f64::EPSILON
}

/// Check if a floating-point number is effectively an integer
fn is_integer(value: f64) -> bool {
    // Handle edge cases for very large numbers where fract() might not work properly
    if value.abs() > 2_f64.powi(52) {
        // For numbers larger than 2^52, floating-point precision means they're effectively integers
        true
    } else {
        (value - value.round()).abs() < f64::EPSILON
    }
}

/// Check if a floating-point number is an even integer
fn is_even_integer(value: f64) -> bool {
    if !is_integer(value) {
        return false;
    }

    // For very large numbers, use modular arithmetic carefully
    if value.abs() > 2_f64.powi(52) {
        // For numbers this large, we can't reliably determine even/odd
        // Conservative approach: assume even to be safe
        true
    } else {
        let rounded = value.round();
        (rounded % 2.0).abs() < f64::EPSILON
    }
}

impl Expression {
    /// Evaluates the expression and returns the result
    ///
    /// # Errors
    ///
    /// This function will return an error if:
    /// * Division by zero is attempted
    /// * A negative base is raised to a fractional exponent (would result in complex number)
    /// * An invalid root index is used (< 2 or fractional)
    /// * An even root of a negative number is attempted
    pub fn evaluate(&self) -> Result<f64, ExpressionError> {
        debug!("Evaluating expression: {}", self);

        let result = match self {
            Expression::Number(n) => Ok(*n),
            Expression::Add(l, r) => {
                let left = l.evaluate()?;
                let right = r.evaluate()?;
                Ok(left + right)
            }
            Expression::Sub(l, r) => {
                let left = l.evaluate()?;
                let right = r.evaluate()?;
                Ok(left - right)
            }
            Expression::Mul(l, r) => {
                let left = l.evaluate()?;
                let right = r.evaluate()?;
                Ok(left * right)
            }
            Expression::Div(l, r) => {
                let left = l.evaluate()?;
                let right = r.evaluate()?;
                if is_zero(right) {
                    debug!("Division by zero attempted");
                    Err(ExpressionError::DivisionByZero)
                } else {
                    Ok(left / right)
                }
            }
            Expression::Pow(l, r) => {
                let left = l.evaluate()?;
                let right = r.evaluate()?;
                if left < 0.0 && !is_integer(right) {
                    debug!(
                        "Complex result from negative base with fractional exponent: {}^{}",
                        left, right
                    );
                    Err(ExpressionError::ComplexResult)
                } else {
                    Ok(left.powf(right))
                }
            }
            Expression::Neg(e) => {
                let val = e.evaluate()?;
                Ok(-val)
            }
            Expression::NthRoot(n, a) => {
                let n_val = n.evaluate()?;
                let a_val = a.evaluate()?;

                if n_val < 2.0 || !is_integer(n_val) {
                    debug!("Invalid root index: {}", n_val);
                    Err(ExpressionError::InvalidRootIndex)
                } else if a_val < 0.0 && is_even_integer(n_val) {
                    debug!(
                        "Even root of negative number: {}th root of {}",
                        n_val, a_val
                    );
                    Err(ExpressionError::EvenRootOfNegative)
                } else if a_val < 0.0 && !is_even_integer(n_val) {
                    // Odd root of negative number: compute as -((-a)^(1/n))
                    debug!(
                        "Computing odd root of negative: {}th root of {}",
                        n_val, a_val
                    );
                    Ok(-((-a_val).powf(1.0 / n_val)))
                } else {
                    Ok(a_val.powf(1.0 / n_val))
                }
            }
        };

        match &result {
            Ok(value) => debug!("Expression evaluated to: {}", value),
            Err(e) => debug!("Expression evaluation failed: {}", e),
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_zero() {
        assert!(is_zero(0.0));
        assert!(is_zero(f64::EPSILON / 2.0));
        assert!(!is_zero(f64::EPSILON * 2.0));
        assert!(!is_zero(1.0));
    }

    #[test]
    fn test_is_integer() {
        assert!(is_integer(1.0));
        assert!(is_integer(42.0));
        assert!(is_integer(-17.0));
        assert!(!is_integer(1.5));
        assert!(!is_integer(1.234_567));

        // Test large numbers
        assert!(is_integer(2_f64.powi(53))); // Should be treated as integer
        assert!(is_integer(1e15)); // Large but representable exactly
    }

    #[test]
    fn test_is_even_integer() {
        assert!(is_even_integer(2.0));
        assert!(is_even_integer(4.0));
        assert!(is_even_integer(-6.0));
        assert!(!is_even_integer(1.0));
        assert!(!is_even_integer(3.0));
        assert!(!is_even_integer(1.5));
    }

    #[test]
    fn test_nth_root_cube_root() {
        // Test cube root: 3rd root of 27 = 3
        let expr = Expression::NthRoot(
            Box::new(Expression::Number(3.0)),
            Box::new(Expression::Number(27.0)),
        );
        let result = expr.evaluate();
        assert!(
            result.is_ok(),
            "Expression should evaluate successfully but got: {:?}",
            result.err()
        );
        if let Ok(value) = result {
            assert!((value - 3.0).abs() < 1e-9, "Expected 3.0, got {}", value);
        }
    }

    #[test]
    fn test_nth_root_square_root() {
        // Test square root: 2nd root of 100 = 10
        let expr = Expression::NthRoot(
            Box::new(Expression::Number(2.0)),
            Box::new(Expression::Number(100.0)),
        );
        let result = expr.evaluate();
        assert!(
            result.is_ok(),
            "Expression should evaluate successfully but got: {:?}",
            result.err()
        );
        if let Ok(value) = result {
            assert!((value - 10.0).abs() < 1e-9, "Expected 10.0, got {}", value);
        }
    }

    #[test]
    fn test_nth_root_fourth_root() {
        // Test 4th root: 4th root of 81 = 3
        let expr = Expression::NthRoot(
            Box::new(Expression::Number(4.0)),
            Box::new(Expression::Number(81.0)),
        );
        let result = expr.evaluate();
        assert!(
            result.is_ok(),
            "Expression should evaluate successfully but got: {:?}",
            result.err()
        );
        if let Ok(value) = result {
            assert!((value - 3.0).abs() < 1e-9, "Expected 3.0, got {}", value);
        }
    }

    #[test]
    fn test_nth_root_square_root_of_49() {
        // Test square root: 2nd root of 49 = 7
        let expr = Expression::NthRoot(
            Box::new(Expression::Number(2.0)),
            Box::new(Expression::Number(49.0)),
        );
        let result = expr.evaluate();
        assert!(
            result.is_ok(),
            "Expression should evaluate successfully but got: {:?}",
            result.err()
        );
        if let Ok(value) = result {
            assert!((value - 7.0).abs() < 1e-9, "Expected 7.0, got {}", value);
        }
    }

    #[test]
    fn test_nth_root_invalid_root_index_fractional() {
        // Test error: fractional root index
        let expr = Expression::NthRoot(
            Box::new(Expression::Number(2.5)),
            Box::new(Expression::Number(16.0)),
        );
        let result = expr.evaluate();
        assert!(
            result.is_err(),
            "Expected error but got success: {:?}",
            result.ok()
        );
        if let Err(e) = result {
            assert_eq!(
                e,
                ExpressionError::InvalidRootIndex,
                "Expected InvalidRootIndex error but got: {:?}",
                e
            );
        }
    }

    #[test]
    fn test_nth_root_invalid_root_index_too_small() {
        // Test error: root index < 2
        let expr = Expression::NthRoot(
            Box::new(Expression::Number(1.0)),
            Box::new(Expression::Number(16.0)),
        );
        let result = expr.evaluate();
        assert!(
            result.is_err(),
            "Expected error but got success: {:?}",
            result.ok()
        );
        if let Err(e) = result {
            assert_eq!(
                e,
                ExpressionError::InvalidRootIndex,
                "Expected InvalidRootIndex error but got: {:?}",
                e
            );
        }
    }

    #[test]
    fn test_nth_root_even_root_of_negative() {
        // Test error: even root of negative number
        let expr = Expression::NthRoot(
            Box::new(Expression::Number(2.0)),
            Box::new(Expression::Number(-16.0)),
        );
        let result = expr.evaluate();
        assert!(
            result.is_err(),
            "Expected error but got success: {:?}",
            result.ok()
        );
        if let Err(e) = result {
            assert_eq!(
                e,
                ExpressionError::EvenRootOfNegative,
                "Expected EvenRootOfNegative error but got: {:?}",
                e
            );
        }
    }

    #[test]
    fn test_nth_root_odd_root_of_negative() {
        // Test odd root of negative number (should work)
        let expr = Expression::NthRoot(
            Box::new(Expression::Number(3.0)),
            Box::new(Expression::Number(-27.0)),
        );
        let result = expr.evaluate();
        assert!(
            result.is_ok(),
            "Expression should evaluate successfully but got: {:?}",
            result.err()
        );
        if let Ok(value) = result {
            assert!(
                (value - (-3.0)).abs() < 1e-9,
                "Expected -3.0, got {}",
                value
            );
        }
    }

    #[test]
    fn test_nth_root_fifth_root() {
        // Test 5th root: 5th root of 32 = 2
        let expr = Expression::NthRoot(
            Box::new(Expression::Number(5.0)),
            Box::new(Expression::Number(32.0)),
        );
        let result = expr.evaluate();
        assert!(
            result.is_ok(),
            "Expression should evaluate successfully but got: {:?}",
            result.err()
        );
        if let Ok(value) = result {
            assert!((value - 2.0).abs() < 1e-9, "Expected 2.0, got {}", value);
        }
    }

    #[test]
    fn test_expression_display_nth_root() {
        // Test the display formatting for nth root
        let expr = Expression::NthRoot(
            Box::new(Expression::Number(3.0)),
            Box::new(Expression::Number(27.0)),
        );
        let display = format!("{}", expr);
        assert_eq!(display, "√3(27)");
    }

    #[test]
    fn test_division_by_small_number() {
        // Test division by very small number (near zero)
        let expr = Expression::Div(
            Box::new(Expression::Number(1.0)),
            Box::new(Expression::Number(f64::EPSILON / 2.0)),
        );
        let result = expr.evaluate();
        assert!(
            result.is_err(),
            "Expected division by zero error for very small divisor"
        );
    }

    #[test]
    fn test_power_with_large_fractional_exponent() {
        // Test that large fractional numbers are handled correctly
        let expr = Expression::Pow(
            Box::new(Expression::Number(-2.0)),
            Box::new(Expression::Number(1000000.1)), // Large fractional
        );
        let result = expr.evaluate();
        assert!(
            result.is_err(),
            "Expected complex result error for negative base with fractional exponent"
        );
    }
}
