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
                // Treat numbers and functions as highest precedence (bind tightest)
                Expression::Number(_) | Expression::NthRoot(_, _) => 5,
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
                    // Rewrite l + (-r) as l - r
                    if let Expression::Neg(inner) = r.as_ref() {
                        let lp = precedence(l);
                        let need_l = lp < 1;
                        write_with_parens(f, l, need_l)?;
                        write!(f, " - ")?;
                        // Right side of subtraction may need parens
                        let rp = precedence(inner);
                        let need_r = rp <= 1;
                        write_with_parens(f, inner, need_r)
                    } else {
                        let lp = precedence(l);
                        let rp = precedence(r);
                        let need_l = lp < 1; // never true; kept for symmetry
                        let need_r = rp < 1; // never true; addition is associative
                        write_with_parens(f, l, need_l)?;
                        write!(f, " + ")?;
                        write_with_parens(f, r, need_r)
                    }
                }
                Expression::Sub(l, r) => {
                    // Rewrite l - (-r) as l + r
                    if let Expression::Neg(inner) = r.as_ref() {
                        let lp = precedence(l);
                        let need_l = lp < 1;
                        write_with_parens(f, l, need_l)?;
                        write!(f, " + ")?;
                        // For addition, parentheses generally not needed
                        let rp = precedence(inner);
                        let need_r = rp < 1; // never true
                        write_with_parens(f, inner, need_r)
                    } else {
                        let lp = precedence(l);
                        let rp = precedence(r);
                        let need_l = lp < 1; // never true
                        // Right child requires parens for same or lower precedence to preserve order
                        let need_r = rp <= 1;
                        write_with_parens(f, l, need_l)?;
                        write!(f, " - ")?;
                        write_with_parens(f, r, need_r)
                    }
                }
                Expression::Mul(l, r) => {
                    let lp = precedence(l);
                    let rp = precedence(r);
                    // Parenthesize children with lower precedence (add/sub)
                    let need_l = lp < 2;
                    let need_r = rp < 2;
                    write_with_parens(f, l, need_l)?;
                    write!(f, " * ")?;
                    write_with_parens(f, r, need_r)
                }
                Expression::Div(l, r) => {
                    let lp = precedence(l);
                    let rp = precedence(r);
                    // Left child: parens for lower precedence (add/sub)
                    let need_l = lp < 2;
                    // Right child: parens for same or lower precedence (to keep non-assoc semantics)
                    let need_r = rp <= 2;
                    write_with_parens(f, l, need_l)?;
                    write!(f, " / ")?;
                    write_with_parens(f, r, need_r)
                }
                Expression::Pow(l, r) => {
                    let lp = precedence(l);
                    let rp = precedence(r);
                    // Base: parens for lower precedence (neg, add, mul, div)
                    let need_l = lp < 4;
                    // Exponent: parens for lower precedence; equal precedence (power) is fine (right-assoc)
                    let need_r = rp < 4;
                    write_with_parens(f, l, need_l)?;
                    write!(f, " ^ ")?;
                    write_with_parens(f, r, need_r)
                }
                Expression::Neg(e) => {
                    // Simplify --x to x
                    if let Expression::Neg(inner) = e.as_ref() {
                        fmt_expression(f, inner)
                    } else {
                        // Parenthesize non-atomic expressions for clarity
                        let need = !matches!(e.as_ref(), Expression::Number(_));
                        write!(f, "-")?;
                        write_with_parens(f, e, need)
                    }
                }
                Expression::NthRoot(n, a) => {
                    // Render nth-root as a ^ (1 / n) using ASCII
                    let lp = precedence(a);
                    let need_l = lp < 4; // same rule as Pow base
                    write_with_parens(f, a, need_l)?;
                    write!(f, " ^ (")?;
                    write!(f, "1 / ")?;
                    let rp = precedence(n);
                    let need_n = rp <= 2; // divisor needs parens for same/lower precedence
                    write_with_parens(f, n, need_n)?;
                    write!(f, ")")
                }
            }
        }

        fmt_expression(f, self)
    }
}
