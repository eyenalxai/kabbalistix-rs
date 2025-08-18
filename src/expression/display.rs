use std::fmt;

use crate::expression::ast::Expression;

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
