use crate::expression::Expression;
use crate::iterator::constants::MAX_ROOT_DEGREE;

pub struct ExpressionGenerator;

impl ExpressionGenerator {
    pub fn generate_binary_ops(left: &Expression, right: &Expression) -> Vec<Expression> {
        vec![
            Expression::Add(Box::new(left.clone()), Box::new(right.clone())),
            Expression::Sub(Box::new(left.clone()), Box::new(right.clone())),
            Expression::Mul(Box::new(left.clone()), Box::new(right.clone())),
            Expression::Div(Box::new(left.clone()), Box::new(right.clone())),
            Expression::Pow(Box::new(left.clone()), Box::new(right.clone())),
        ]
    }

    pub fn generate_nth_root(n: &Expression, a: &Expression) -> Option<Expression> {
        if let Expression::Number(n_val) = n {
            if *n_val >= 2.0 && n_val.fract() == 0.0 && *n_val <= MAX_ROOT_DEGREE {
                return Some(Expression::NthRoot(
                    Box::new(n.clone()),
                    Box::new(a.clone()),
                ));
            }
        }
        None
    }

    pub fn generate_nary_ops(operands: &[Expression]) -> Vec<Expression> {
        if operands.len() < 2 {
            return Vec::new();
        }

        let mut results = Vec::new();
        let first = operands.first().unwrap_or(&Expression::Number(0.0));

        let mut add_result = first.clone();
        for operand in operands.iter().skip(1) {
            add_result = Expression::Add(Box::new(add_result), Box::new(operand.clone()));
        }
        results.push(add_result);

        let mut mul_result = first.clone();
        for operand in operands.iter().skip(1) {
            mul_result = Expression::Mul(Box::new(mul_result), Box::new(operand.clone()));
        }
        results.push(mul_result);

        let mut sub_result = first.clone();
        for operand in operands.iter().skip(1) {
            sub_result = Expression::Sub(Box::new(sub_result), Box::new(operand.clone()));
        }
        results.push(sub_result);

        if operands.len() >= 3 {
            let mut mixed_result = Expression::Sub(
                Box::new(first.clone()),
                Box::new(operands.get(1).unwrap_or(&Expression::Number(0.0)).clone()),
            );
            for operand in operands.iter().skip(2) {
                mixed_result = Expression::Add(Box::new(mixed_result), Box::new(operand.clone()));
            }
            results.push(mixed_result);
        }

        results
    }
}
