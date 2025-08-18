#![deny(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::unwrap_or_default,
    clippy::get_unwrap,
    clippy::map_unwrap_or,
    clippy::unnecessary_unwrap,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::todo,
    clippy::unimplemented,
    clippy::unreachable,
    clippy::exit,
    clippy::mem_forget,
    clippy::clone_on_ref_ptr,
    clippy::mutex_atomic,
    clippy::rc_mutex
)]

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
