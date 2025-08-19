use kabbalistix::solver::ExpressionSolver;
use kabbalistix::utils::validate_digit_string;

use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use log::{info, warn};

#[derive(Debug, Clone, ValueEnum)]
pub enum LogLevel {
    Error,
    Warn,
    Info,
    Debug,
    Trace,
}

impl LogLevel {
    pub fn to_log_level_filter(&self) -> log::LevelFilter {
        match self {
            LogLevel::Error => log::LevelFilter::Error,
            LogLevel::Warn => log::LevelFilter::Warn,
            LogLevel::Info => log::LevelFilter::Info,
            LogLevel::Debug => log::LevelFilter::Debug,
            LogLevel::Trace => log::LevelFilter::Trace,
        }
    }
}

#[derive(Parser, Debug)]
#[command(name = "kabbalistix")]
#[command(
    about = "Find mathematical expressions from digit strings that evaluate to a target value"
)]
#[command(version)]
pub struct CliArgs {
    pub digit_string: String,
    pub target: f64,
    #[arg(short, long, value_enum, default_value = "warn")]
    pub log_level: LogLevel,
}

pub struct CliConfig {
    pub digit_string: String,
    pub target: f64,
    pub log_level: LogLevel,
}

fn parse_args() -> Result<CliConfig> {
    let args = CliArgs::parse();
    validate_digit_string(&args.digit_string).context("Invalid digit string")?;
    Ok(CliConfig {
        digit_string: args.digit_string,
        target: args.target,
        log_level: args.log_level,
    })
}

fn init_logging(log_level: &LogLevel) -> Result<()> {
    env_logger::Builder::from_default_env()
        .filter_level(log_level.to_log_level_filter())
        .init();
    Ok(())
}

fn run() -> Result<()> {
    let config = parse_args()?;
    init_logging(&config.log_level)?;
    let solver = ExpressionSolver::new();
    info!(
        "Searching for expressions using digits '{}' that equal {}",
        config.digit_string, config.target
    );
    match solver.find_expression(&config.digit_string, config.target) {
        Some(expr) => {
            match expr.evaluate() {
                Ok(value) => {
                    let display_value = if value == 0.0 { 0.0 } else { value };
                    println!("{} = {}", expr, display_value)
                }
                Err(_) => println!("{}", expr),
            }
            Ok(())
        }
        None => {
            warn!("No matching expression found");
            println!("Unknown.");
            Ok(())
        }
    }
}

fn main() {
    if let Err(err) = run() {
        eprintln!("Error: {}", err);
        std::process::exit(1);
    }
}
