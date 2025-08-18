mod cli;
mod expression;
mod solver;
mod utils;

fn main() {
    if let Err(err) = cli::run() {
        eprintln!("Error: {}", err);
        #[allow(clippy::exit)]
        std::process::exit(1);
    }
}
