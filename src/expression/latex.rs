use crate::expression::ast::Expression;

impl Expression {
    /// Render the expression as LaTeX without hiding numeric factors.
    /// - Always shows coefficients (including 1 and -1) explicitly
    /// - Uses \cdot for multiplication
    /// - Uses \frac for division
    /// - Renders nth roots as radical with explicit index
    pub fn to_latex(&self) -> String {
        fn precedence(expr: &Expression) -> u8 {
            match expr {
                Expression::Add(_, _) | Expression::Sub(_, _) => 1,
                Expression::Mul(_, _) | Expression::Div(_, _) => 2,
                Expression::Neg(_) => 3,
                Expression::Pow(_, _) | Expression::NthRoot(_, _) => 4,
                Expression::Number(_) => 5,
            }
        }

        fn wrap_parens(s: String) -> String {
            format!("\\left({}\\right)", s)
        }

        fn fmt(expr: &Expression) -> String {
            match expr {
                Expression::Number(n) => number_to_string(*n),
                Expression::Add(l, r) => {
                    let lp = precedence(l);
                    let rp = precedence(r);
                    let mut ls = fmt(l);
                    let mut rs = fmt(r);
                    if lp < 1 {
                        ls = wrap_parens(ls);
                    }
                    if rp < 1 {
                        rs = wrap_parens(rs);
                    }
                    format!("{} + {}", ls, rs)
                }
                Expression::Sub(l, r) => {
                    let lp = precedence(l);
                    let rp = precedence(r);
                    let mut ls = fmt(l);
                    let mut rs = fmt(r);
                    if lp < 1 {
                        ls = wrap_parens(ls);
                    }
                    if rp <= 1 {
                        rs = wrap_parens(rs);
                    }
                    format!("{} - {}", ls, rs)
                }
                Expression::Mul(l, r) => {
                    let lp = precedence(l);
                    let rp = precedence(r);
                    let mut ls = fmt(l);
                    let mut rs = fmt(r);
                    if lp < 2 {
                        ls = wrap_parens(ls);
                    }
                    if rp < 2 {
                        rs = wrap_parens(rs);
                    }
                    format!("{} \\cdot {}", ls, rs)
                }
                Expression::Div(l, r) => {
                    let num = fmt(l);
                    let den = fmt(r);
                    format!("\\frac{{{}}}{{{}}}", num, den)
                }
                Expression::Pow(l, r) => {
                    let lp = precedence(l);
                    let mut base = fmt(l);
                    let exp = fmt(r);
                    if lp < 4 {
                        base = wrap_parens(base);
                    }
                    format!("{}^{{{}}}", base, exp)
                }
                Expression::Neg(e) => {
                    let ep = precedence(e);
                    let mut inner = fmt(e);
                    if ep < 3 {
                        inner = wrap_parens(inner);
                    }
                    // Do not hide the -1 factor
                    format!("-1 \\cdot {}", inner)
                }
                Expression::NthRoot(n, a) => {
                    // Render as radical with explicit index for all n (including 2)
                    let n_str = fmt(n);
                    let arg = fmt(a);
                    format!("\\sqrt[{}]{{{}}}", n_str, arg)
                }
            }
        }

        fn number_to_string(n: f64) -> String {
            // Preserve integers without trailing .0, keep others as-is
            if (n.fract() == 0.0) && n.is_finite() {
                format!("{}", n.trunc() as i128)
            } else if n.is_infinite() {
                if n.is_sign_positive() {
                    String::from("\\infty")
                } else {
                    String::from("-\\infty")
                }
            } else if n.is_nan() {
                String::from("\\mathrm{NaN}")
            } else {
                let s = format!("{}", n);
                s
            }
        }

        fmt(self)
    }
}
