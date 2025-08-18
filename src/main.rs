#![deny(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

mod cli;
mod expression;
mod solver;
mod utils;

fn main() {
    if let Err(err) = cli::run() {
        eprintln!("Error: {}", err);
        std::process::exit(1);
    }
}
