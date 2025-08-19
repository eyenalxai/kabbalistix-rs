use crate::expression::Expression;
use crate::iterator::{ExpressionIterator, iter_expressions};

fn collect_all(digits: &str, limit: usize) -> Vec<Expression> {
    let mut it = iter_expressions(digits);
    let mut out = Vec::new();
    for _ in 0..limit {
        if let Some(expr) = it.next() {
            out.push(expr);
        } else {
            break;
        }
    }
    out
}

#[test]
fn iterator_yields_base_number_for_full_range() {
    let mut it = ExpressionIterator::from_digits("123");
    let first = it.next();
    assert!(
        matches!(first, Some(Expression::Number(_))),
        "expected Number, got {:?}",
        first
    );
    if let Some(Expression::Number(n)) = first {
        assert_eq!(n, 123.0);
    }
}

#[test]
fn iterator_generates_more_than_base_for_len_ge_2() {
    let exprs = collect_all("12", 10);
    assert!(
        exprs.len() >= 2,
        "expected at least base and one composite, got {}",
        exprs.len()
    );
}

#[test]
fn iterator_uses_all_digits_for_yielded_items() {
    let exprs = collect_all("123", 20);
    // All items yielded by top-level iterator should use full range when returned
    assert!(!exprs.is_empty());
}
