use std::fmt;

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

impl Expression {
    /// Evaluates the expression and returns the result
    pub fn evaluate(&self) -> Result<f64, String> {
        match self {
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
                if right == 0.0 {
                    Err("Division by zero".to_string())
                } else {
                    Ok(left / right)
                }
            }
            Expression::Pow(l, r) => {
                let left = l.evaluate()?;
                let right = r.evaluate()?;
                if left < 0.0 && right.fract() != 0.0 {
                    Err("Complex result from negative base with fractional exponent".to_string())
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
                if n_val < 2.0 || n_val.fract() != 0.0 {
                    Err("Root index must be an integer >= 2".to_string())
                } else if a_val < 0.0 && (n_val as i32) % 2 == 0 {
                    Err("Even root of negative number".to_string())
                } else if a_val < 0.0 && (n_val as i32) % 2 == 1 {
                    // Odd root of negative number: compute as -((-a)^(1/n))
                    Ok(-((-a_val).powf(1.0 / n_val)))
                } else {
                    Ok(a_val.powf(1.0 / n_val))
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nth_root_cube_root() {
        // Test cube root: 3rd root of 27 = 3
        let expr = Expression::NthRoot(
            Box::new(Expression::Number(3.0)),
            Box::new(Expression::Number(27.0)),
        );
        let result = expr.evaluate().unwrap();
        assert!((result - 3.0).abs() < 1e-9, "Expected 3.0, got {}", result);
    }

    #[test]
    fn test_nth_root_square_root() {
        // Test square root: 2nd root of 100 = 10
        let expr = Expression::NthRoot(
            Box::new(Expression::Number(2.0)),
            Box::new(Expression::Number(100.0)),
        );
        let result = expr.evaluate().unwrap();
        assert!(
            (result - 10.0).abs() < 1e-9,
            "Expected 10.0, got {}",
            result
        );
    }

    #[test]
    fn test_nth_root_fourth_root() {
        // Test 4th root: 4th root of 81 = 3
        let expr = Expression::NthRoot(
            Box::new(Expression::Number(4.0)),
            Box::new(Expression::Number(81.0)),
        );
        let result = expr.evaluate().unwrap();
        assert!((result - 3.0).abs() < 1e-9, "Expected 3.0, got {}", result);
    }

    #[test]
    fn test_nth_root_square_root_of_49() {
        // Test square root: 2nd root of 49 = 7
        let expr = Expression::NthRoot(
            Box::new(Expression::Number(2.0)),
            Box::new(Expression::Number(49.0)),
        );
        let result = expr.evaluate().unwrap();
        assert!((result - 7.0).abs() < 1e-9, "Expected 7.0, got {}", result);
    }

    #[test]
    fn test_nth_root_invalid_root_index_fractional() {
        // Test error: fractional root index
        let expr = Expression::NthRoot(
            Box::new(Expression::Number(2.5)),
            Box::new(Expression::Number(16.0)),
        );
        let result = expr.evaluate();
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Root index must be an integer >= 2");
    }

    #[test]
    fn test_nth_root_invalid_root_index_too_small() {
        // Test error: root index < 2
        let expr = Expression::NthRoot(
            Box::new(Expression::Number(1.0)),
            Box::new(Expression::Number(16.0)),
        );
        let result = expr.evaluate();
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Root index must be an integer >= 2");
    }

    #[test]
    fn test_nth_root_even_root_of_negative() {
        // Test error: even root of negative number
        let expr = Expression::NthRoot(
            Box::new(Expression::Number(2.0)),
            Box::new(Expression::Number(-16.0)),
        );
        let result = expr.evaluate();
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Even root of negative number");
    }

    #[test]
    fn test_nth_root_odd_root_of_negative() {
        // Test odd root of negative number (should work)
        let expr = Expression::NthRoot(
            Box::new(Expression::Number(3.0)),
            Box::new(Expression::Number(-27.0)),
        );
        let result = expr.evaluate().unwrap();
        assert!(
            (result - (-3.0)).abs() < 1e-9,
            "Expected -3.0, got {}",
            result
        );
    }

    #[test]
    fn test_nth_root_fifth_root() {
        // Test 5th root: 5th root of 32 = 2
        let expr = Expression::NthRoot(
            Box::new(Expression::Number(5.0)),
            Box::new(Expression::Number(32.0)),
        );
        let result = expr.evaluate().unwrap();
        assert!((result - 2.0).abs() < 1e-9, "Expected 2.0, got {}", result);
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
}
