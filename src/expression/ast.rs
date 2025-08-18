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
