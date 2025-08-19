use log::debug;

use crate::expression::ast::Expression;
use crate::expression::errors::ExpressionError;

#[inline]
fn is_zero(value: f64) -> bool {
    value.abs() < f64::EPSILON
}

#[inline]
fn is_integer(value: f64) -> bool {
    if value.abs() > 2_f64.powi(52) {
        true
    } else {
        (value - value.round()).abs() < f64::EPSILON
    }
}

#[inline]
fn is_even_integer(value: f64) -> bool {
    if !is_integer(value) {
        return false;
    }

    if value.abs() > 2_f64.powi(52) {
        true
    } else {
        let rounded = value.round();
        (rounded % 2.0).abs() < f64::EPSILON
    }
}

impl Expression {
    /// # Errors
    ///
    /// Returns an error when attempting:
    /// - Division by zero
    /// - Raising a negative base to a fractional exponent (complex result)
    /// - Using an invalid root index (< 2 or fractional)
    /// - Taking an even root of a negative number
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
mod tests_inner_helpers {
    use super::{is_even_integer, is_integer, is_zero};

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

        assert!(is_integer(2_f64.powi(53)));
        assert!(is_integer(1e15));
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
}
