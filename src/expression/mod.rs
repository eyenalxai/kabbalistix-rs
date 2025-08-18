//! Expression module split into submodules for clarity

mod ast;
mod display;
mod errors;
mod eval;

pub use ast::Expression;
pub use errors::ExpressionError;

#[cfg(test)]
mod tests;
