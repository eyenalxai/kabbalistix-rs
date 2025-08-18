use crate::expression::ast::Expression;
use crate::expression::errors::ExpressionError;

#[test]
fn test_nth_root_cube_root() {
    let expr = Expression::NthRoot(
        Box::new(Expression::Number(3.0)),
        Box::new(Expression::Number(27.0)),
    );
    let result = expr.evaluate();
    assert!(result.is_ok());
    if let Ok(value) = result {
        assert!((value - 3.0).abs() < 1e-9);
    }
}

#[test]
fn test_nth_root_square_root() {
    let expr = Expression::NthRoot(
        Box::new(Expression::Number(2.0)),
        Box::new(Expression::Number(100.0)),
    );
    let result = expr.evaluate();
    assert!(result.is_ok());
    if let Ok(value) = result {
        assert!((value - 10.0).abs() < 1e-9);
    }
}

#[test]
fn test_nth_root_fourth_root() {
    let expr = Expression::NthRoot(
        Box::new(Expression::Number(4.0)),
        Box::new(Expression::Number(81.0)),
    );
    let result = expr.evaluate();
    assert!(result.is_ok());
    if let Ok(value) = result {
        assert!((value - 3.0).abs() < 1e-9);
    }
}

#[test]
fn test_nth_root_square_root_of_49() {
    let expr = Expression::NthRoot(
        Box::new(Expression::Number(2.0)),
        Box::new(Expression::Number(49.0)),
    );
    let result = expr.evaluate();
    assert!(result.is_ok());
    if let Ok(value) = result {
        assert!((value - 7.0).abs() < 1e-9);
    }
}

#[test]
fn test_nth_root_invalid_root_index_fractional() {
    let expr = Expression::NthRoot(
        Box::new(Expression::Number(2.5)),
        Box::new(Expression::Number(16.0)),
    );
    let result = expr.evaluate();
    assert!(result.is_err());
    if let Err(e) = result {
        assert_eq!(e, ExpressionError::InvalidRootIndex);
    }
}

#[test]
fn test_nth_root_invalid_root_index_too_small() {
    let expr = Expression::NthRoot(
        Box::new(Expression::Number(1.0)),
        Box::new(Expression::Number(16.0)),
    );
    let result = expr.evaluate();
    assert!(result.is_err());
    if let Err(e) = result {
        assert_eq!(e, ExpressionError::InvalidRootIndex);
    }
}

#[test]
fn test_nth_root_even_root_of_negative() {
    let expr = Expression::NthRoot(
        Box::new(Expression::Number(2.0)),
        Box::new(Expression::Number(-16.0)),
    );
    let result = expr.evaluate();
    assert!(result.is_err());
    if let Err(e) = result {
        assert_eq!(e, ExpressionError::EvenRootOfNegative);
    }
}

#[test]
fn test_nth_root_odd_root_of_negative() {
    let expr = Expression::NthRoot(
        Box::new(Expression::Number(3.0)),
        Box::new(Expression::Number(-27.0)),
    );
    let result = expr.evaluate();
    assert!(result.is_ok());
    if let Ok(value) = result {
        assert!((value - (-3.0)).abs() < 1e-9);
    }
}

#[test]
fn test_nth_root_fifth_root() {
    let expr = Expression::NthRoot(
        Box::new(Expression::Number(5.0)),
        Box::new(Expression::Number(32.0)),
    );
    let result = expr.evaluate();
    assert!(result.is_ok());
    if let Ok(value) = result {
        assert!((value - 2.0).abs() < 1e-9);
    }
}

#[test]
fn test_expression_display_nth_root() {
    let expr = Expression::NthRoot(
        Box::new(Expression::Number(3.0)),
        Box::new(Expression::Number(27.0)),
    );
    let display = format!("{}", expr);
    assert_eq!(display, "√3(27)");
}

#[test]
fn test_division_by_small_number() {
    let expr = Expression::Div(
        Box::new(Expression::Number(1.0)),
        Box::new(Expression::Number(f64::EPSILON / 2.0)),
    );
    let result = expr.evaluate();
    assert!(result.is_err());
}

#[test]
fn test_power_with_large_fractional_exponent() {
    let expr = Expression::Pow(
        Box::new(Expression::Number(-2.0)),
        Box::new(Expression::Number(1000000.1)),
    );
    let result = expr.evaluate();
    assert!(result.is_err());
}
