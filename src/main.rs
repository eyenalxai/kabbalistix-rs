mod cli;
mod expression;
mod iterator;
mod solver;
mod utils;

fn main() {
    if let Err(err) = cli::run() {
        eprintln!("Error: {}", err);
        std::process::exit(1);
    }
}
