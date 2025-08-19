use std::fmt;

use crate::expression::ast::Expression;

impl fmt::Display for Expression {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fn precedence(expr: &Expression) -> u8 {
            match expr {
                Expression::Add(_, _) | Expression::Sub(_, _) => 1,
                Expression::Mul(_, _) | Expression::Div(_, _) => 2,
                Expression::Neg(_) => 3,
                Expression::Pow(_, _) => 4,
                Expression::Number(_) | Expression::NthRoot(_, _) => 5,
            }
        }

        fn is_literal_zero(expr: &Expression) -> bool {
            match expr {
                Expression::Number(n) => *n == 0.0,
                Expression::Neg(inner) => {
                    matches!(inner.as_ref(), Expression::Number(n) if *n == 0.0)
                }
                _ => false,
            }
        }

        fn leading_is_unary_minus(expr: &Expression) -> bool {
            match expr {
                Expression::Neg(_) => true,
                Expression::Number(n) => *n < 0.0,
                Expression::Add(l, _)
                | Expression::Sub(l, _)
                | Expression::Mul(l, _)
                | Expression::Div(l, _)
                | Expression::Pow(l, _) => leading_is_unary_minus(l),
                Expression::NthRoot(_, a) => leading_is_unary_minus(a),
            }
        }

        fn write_with_parens(
            f: &mut fmt::Formatter,
            expr: &Expression,
            need_parens: bool,
        ) -> fmt::Result {
            if need_parens {
                write!(f, "(")?;
                fmt_expression(f, expr)?;
                write!(f, ")")
            } else {
                fmt_expression(f, expr)
            }
        }

        fn fmt_expression(f: &mut fmt::Formatter, expr: &Expression) -> fmt::Result {
            match expr {
                Expression::Number(n) => write!(f, "{}", n),
                Expression::Add(l, r) => {
                    if let Expression::Neg(inner) = r.as_ref() {
                        let lp = precedence(l);
                        let need_l = lp < 1;
                        write_with_parens(f, l, need_l)?;
                        let rp = precedence(inner);
                        let need_r = rp <= 1;
                        if need_r && leading_is_unary_minus(inner) {
                            write!(f, " -")?;
                        } else {
                            write!(f, " - ")?;
                        }
                        write_with_parens(f, inner, need_r)
                    } else {
                        let lp = precedence(l);
                        let rp = precedence(r);
                        let need_l = lp < 1;
                        let need_r = rp < 1;
                        write_with_parens(f, l, need_l)?;
                        write!(f, " + ")?;
                        write_with_parens(f, r, need_r)
                    }
                }
                Expression::Sub(l, r) => {
                    if let Expression::Neg(inner) = r.as_ref() {
                        let lp = precedence(l);
                        let need_l = lp < 1;
                        write_with_parens(f, l, need_l)?;
                        write!(f, " + ")?;
                        let rp = precedence(inner);
                        let need_r = rp < 1;
                        write_with_parens(f, inner, need_r)
                    } else {
                        let lp = precedence(l);
                        let rp = precedence(r);
                        let need_l = lp < 1;
                        let need_r = rp <= 1;
                        write_with_parens(f, l, need_l)?;
                        if need_r && leading_is_unary_minus(r) {
                            write!(f, " -")?;
                        } else {
                            write!(f, " - ")?;
                        }
                        write_with_parens(f, r, need_r)
                    }
                }
                Expression::Mul(l, r) => {
                    if let Expression::Number(n) = l.as_ref() {
                        if *n == 1.0 && !is_literal_zero(r) {
                            return fmt_expression(f, r);
                        }
                        if *n == -1.0 {
                            if is_literal_zero(r) {
                                return fmt_expression(f, r);
                            }
                            let need = !matches!(r.as_ref(), Expression::Number(_));
                            write!(f, "-")?;
                            return write_with_parens(f, r, need);
                        }
                    }
                    if let Expression::Number(n) = r.as_ref() {
                        if *n == 1.0 && !is_literal_zero(l) {
                            return fmt_expression(f, l);
                        }
                        if *n == -1.0 {
                            if is_literal_zero(l) {
                                return fmt_expression(f, l);
                            }
                            let need = !matches!(l.as_ref(), Expression::Number(_));
                            write!(f, "-")?;
                            return write_with_parens(f, l, need);
                        }
                    }
                    let lp = precedence(l);
                    let rp = precedence(r);
                    let need_l = lp < 2;
                    let need_r = rp < 2;
                    write_with_parens(f, l, need_l)?;
                    write!(f, " * ")?;
                    write_with_parens(f, r, need_r)
                }
                Expression::Div(l, r) => {
                    let lp = precedence(l);
                    let rp = precedence(r);
                    let need_l = lp < 2;
                    let need_r = rp <= 2;
                    write_with_parens(f, l, need_l)?;
                    write!(f, " / ")?;
                    write_with_parens(f, r, need_r)
                }
                Expression::Pow(l, r) => {
                    let lp = precedence(l);
                    let rp = precedence(r);
                    let need_l = lp < 4;
                    let need_r = rp < 4;
                    write_with_parens(f, l, need_l)?;
                    write!(f, " ^ ")?;
                    write_with_parens(f, r, need_r)
                }
                Expression::Neg(e) => {
                    if let Expression::Number(n) = e.as_ref()
                        && *n == 0.0
                    {
                        return write!(f, "0");
                    }
                    if let Expression::Neg(inner) = e.as_ref() {
                        fmt_expression(f, inner)
                    } else {
                        let need = !matches!(e.as_ref(), Expression::Number(_));
                        write!(f, "-")?;
                        write_with_parens(f, e, need)
                    }
                }
                Expression::NthRoot(n, a) => {
                    let lp = precedence(a);
                    let need_l = lp < 4;
                    write_with_parens(f, a, need_l)?;
                    write!(f, " ^ (")?;
                    write!(f, "1 / ")?;
                    let rp = precedence(n);
                    let need_n = rp <= 2;
                    write_with_parens(f, n, need_n)?;
                    write!(f, ")")
                }
            }
        }

        fmt_expression(f, self)
    }
}
