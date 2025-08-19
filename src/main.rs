use kabbalistix::solver::ExpressionSolver;
use kabbalistix::solver::constants::EPSILON;
use kabbalistix::utils::validate_digit_string;

use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use log::{info, warn};
use serde::{Serialize, Serializer};

#[derive(Debug, Clone, ValueEnum)]
pub enum LogLevel {
    Off,
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
            LogLevel::Off => log::LevelFilter::Off,
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
    #[arg(short, long, help = "Output result as JSON")]
    pub json: bool,
}

pub struct CliConfig {
    pub digit_string: String,
    pub target: f64,
    pub log_level: LogLevel,
    pub output_format: OutputFormat,
    pub json: bool,
}

fn parse_args() -> Result<CliConfig> {
    let args = CliArgs::parse();
    validate_digit_string(&args.digit_string).context("Invalid digit string")?;
    Ok(CliConfig {
        digit_string: args.digit_string,
        target: args.target,
        log_level: args.log_level,
        output_format: args.output_format,
        json: args.json,
    })
}

fn init_logging(log_level: &LogLevel) -> Result<()> {
    env_logger::Builder::from_default_env()
        .filter_level(log_level.to_log_level_filter())
        .init();
    Ok(())
}

fn serialize_numeric_value<S>(value: &Option<f64>, serializer: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    match value {
        Some(v) if v.fract() == 0.0 && v.is_finite() => serializer.serialize_i64(*v as i64),
        Some(v) => serializer.serialize_f64(*v),
        None => serializer.serialize_none(),
    }
}

#[derive(Serialize)]
struct ExpressionResult {
    #[serde(skip_serializing_if = "Option::is_none")]
    expression: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    latex: Option<String>,
    #[serde(
        skip_serializing_if = "Option::is_none",
        serialize_with = "serialize_numeric_value"
    )]
    value: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
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
            if config.json {
                output_json_result(&expr, &config)?;
            } else {
                output_text_result(&expr, &config)?;
            }
            Ok(())
        }
        None => {
            if config.json {
                let result = ExpressionResult {
                    expression: None,
                    latex: None,
                    value: None,
                    error: Some("No matching expression found".to_string()),
                };
                println!("{}", serde_json::to_string(&result)?);
            } else {
                warn!("No matching expression found");
                println!("Unknown.");
            }
            Ok(())
        }
    }
}

fn output_json_result(
    expr: &kabbalistix::expression::Expression,
    config: &CliConfig,
) -> Result<()> {
    match expr.evaluate() {
        Ok(value) => {
            let error = (value - config.target).abs();
            let display_value = if error < EPSILON {
                config.target
            } else if value == 0.0 {
                0.0
            } else {
                value
            };

            let result = match config.output_format {
                OutputFormat::Usual => ExpressionResult {
                    expression: Some(format!("{} = {}", expr, display_value)),
                    latex: None,
                    value: Some(display_value),
                    error: None,
                },
                OutputFormat::Latex => ExpressionResult {
                    expression: None,
                    latex: Some(format!(
                        "{} = {}",
                        expr.to_latex(),
                        number_to_latex(display_value)
                    )),
                    value: Some(display_value),
                    error: None,
                },
                OutputFormat::Both => ExpressionResult {
                    expression: Some(format!("{} = {}", expr, display_value)),
                    latex: Some(format!(
                        "{} = {}",
                        expr.to_latex(),
                        number_to_latex(display_value)
                    )),
                    value: Some(display_value),
                    error: None,
                },
            };
            println!("{}", serde_json::to_string(&result)?);
        }
        Err(eval_error) => {
            let result = match config.output_format {
                OutputFormat::Usual => ExpressionResult {
                    expression: Some(expr.to_string()),
                    latex: None,
                    value: None,
                    error: Some(eval_error.to_string()),
                },
                OutputFormat::Latex => ExpressionResult {
                    expression: None,
                    latex: Some(expr.to_latex()),
                    value: None,
                    error: Some(eval_error.to_string()),
                },
                OutputFormat::Both => ExpressionResult {
                    expression: Some(expr.to_string()),
                    latex: Some(expr.to_latex()),
                    value: None,
                    error: Some(eval_error.to_string()),
                },
            };
            println!("{}", serde_json::to_string(&result)?);
        }
    }
    Ok(())
}

fn output_text_result(
    expr: &kabbalistix::expression::Expression,
    config: &CliConfig,
) -> Result<()> {
    match expr.evaluate() {
        Ok(value) => {
            let error = (value - config.target).abs();
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
