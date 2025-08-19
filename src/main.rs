use kabbalistix::solver::ExpressionSolver;
use kabbalistix::solver::constants::EPSILON;
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

#[derive(Debug, Clone, ValueEnum)]
pub enum OutputFormat {
    Usual,
    Latex,
    Both,
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
    #[arg(short = 'f', long, value_enum, default_value = "both")]
    pub output_format: OutputFormat,
}

pub struct CliConfig {
    pub digit_string: String,
    pub target: f64,
    pub log_level: LogLevel,
    pub output_format: OutputFormat,
}

fn parse_args() -> Result<CliConfig> {
    let args = CliArgs::parse();
    validate_digit_string(&args.digit_string).context("Invalid digit string")?;
    Ok(CliConfig {
        digit_string: args.digit_string,
        target: args.target,
        log_level: args.log_level,
        output_format: args.output_format,
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
                    let error = (value - config.target).abs();

                    // If error is within epsilon, round to target for display
                    let display_value = if error < EPSILON {
                        config.target
                    } else if value == 0.0 {
                        0.0
                    } else {
                        value
                    };

                    match config.output_format {
                        OutputFormat::Usual => {
                            println!("{} = {}", expr, display_value);
                        }
                        OutputFormat::Latex => {
                            println!("${} = {}$", expr.to_latex(), number_to_latex(display_value));
                        }
                        OutputFormat::Both => {
                            println!("{} = {}", expr, display_value);
                            println!(
                                "LaTeX: ${} = {}$",
                                expr.to_latex(),
                                number_to_latex(display_value)
                            );
                        }
                    }
                }
                Err(_) => match config.output_format {
                    OutputFormat::Usual => {
                        println!("{}", expr);
                    }
                    OutputFormat::Latex => {
                        println!("${}$", expr.to_latex());
                    }
                    OutputFormat::Both => {
                        println!("{}", expr);
                        println!("LaTeX: ${}$", expr.to_latex());
                    }
                },
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

fn number_to_latex(n: f64) -> String {
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
        format!("{}", n)
    }
}
