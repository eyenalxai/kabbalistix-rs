use crate::expression::Expression;
use crate::iterator::ExpressionIterator;
use crate::solver::ExpressionSolver;

#[test]
fn test_find_expression_with_nth_root() {
    let solver = ExpressionSolver::new();
    let result = solver.find_expression("327", 3.0);
    assert!(result.is_some());

    if let Some(expr) = result {
        let result = expr.evaluate();
        assert!(result.is_ok());
        if let Ok(value) = result {
            assert!((value - 3.0).abs() < 1e-9);
        }

        let expr_str = format!("{}", expr);
        assert!(expr_str.contains("√3(27)"));
    }
}

#[test]
fn test_solver_creation() {
    let solver = ExpressionSolver::new();
    let result = solver.find_expression("24", 6.0);
    assert!(result.is_some());
}

#[test]
fn test_expression_iterator() {
    let iter = ExpressionIterator::new("12".to_string());
    let expressions: Vec<_> = iter.collect();
    assert!(!expressions.is_empty());
    assert!(
        expressions
            .iter()
            .any(|e| matches!(e, Expression::Number(n) if (*n - 12.0).abs() < 1e-9))
    );
}

#[test]
fn test_seven_twos_equals_fourteen() {
    let solver = ExpressionSolver::new();
    let result = solver.find_expression("2222222", 14.0);
    assert!(result.is_some());
    if let Some(expr) = result {
        assert!(expr.evaluate().is_ok());
        let value = expr.evaluate().unwrap_or(0.0);
        assert!((value - 14.0).abs() < 1e-9);
        let expr_str = format!("{}", expr);
        let two_count = expr_str.matches('2').count();
        assert_eq!(two_count, 7);
    }
}

#[test]
fn test_addition_case() {
    let solver = ExpressionSolver::new();
    let result = solver.find_expression("222", 6.0);
    assert!(result.is_some());
    if let Some(expr) = result {
        assert!(expr.evaluate().is_ok());
        let value = expr.evaluate().unwrap_or(0.0);
        assert!((value - 6.0).abs() < 1e-9);
    }
}

#[test]
fn test_all_digits_must_be_used() {
    let solver = ExpressionSolver::new();
    let result = solver.find_expression("1111", 4.0);
    assert!(result.is_some());
    if let Some(expr) = result {
        assert!(expr.evaluate().is_ok());
        if let Ok(value) = expr.evaluate() {
            assert!((value - 4.0).abs() < 1e-9);
        }
    }
}

#[test]
fn test_six_twos_equals_four() {
    let solver = ExpressionSolver::new();
    let result = solver.find_expression("222222", 4.0);
    assert!(result.is_some());
    if let Some(expr) = result {
        assert!(expr.evaluate().is_ok());
        let value = expr.evaluate().unwrap_or(0.0);
        assert!((value - 4.0).abs() < 1e-9);
        let expr_str = format!("{}", expr);
        let two_count = expr_str.matches('2').count();
        assert_eq!(two_count, 6);
    }
}

#[test]
fn test_five_twos_equals_1024() {
    let solver = ExpressionSolver::new();
    let result = solver.find_expression("22222", 1024.0);
    assert!(result.is_some());
    if let Some(expr) = result {
        assert!(expr.evaluate().is_ok());
        let value = expr.evaluate().unwrap_or(0.0);
        assert!((value - 1024.0).abs() < 1e-9);
        let expr_str = format!("{}", expr);
        let two_count = expr_str.matches('2').count();
        assert_eq!(two_count, 5);
    }
}
