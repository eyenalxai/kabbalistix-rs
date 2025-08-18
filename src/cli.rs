use crate::solver::ExpressionSolver;
use crate::utils::validate_digit_string;
use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use log::{info, warn};

/// Log level for the application
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

/// Kabbalistix - Find mathematical expressions from digit strings
#[derive(Parser, Debug)]
#[command(name = "kabbalistix")]
#[command(
    about = "Find mathematical expressions from digit strings that evaluate to a target value"
)]
#[command(version)]
pub struct CliArgs {
    /// String of digits to use in the expression
    pub digit_string: String,

    /// Target value to match
    pub target: f64,

    /// Log level (default: warn)
    #[arg(short, long, value_enum, default_value = "warn")]
    pub log_level: LogLevel,
}

/// Configuration for the CLI application
pub struct CliConfig {
    pub digit_string: String,
    pub target: f64,
    pub log_level: LogLevel,
}

/// Parse command line arguments and return configuration
pub fn parse_args() -> Result<CliConfig> {
    let args = CliArgs::parse();

    // Validate digit string
    validate_digit_string(&args.digit_string).context("Invalid digit string")?;

    Ok(CliConfig {
        digit_string: args.digit_string,
        target: args.target,
        log_level: args.log_level,
    })
}

/// Initialize logging based on the provided log level
pub fn init_logging(log_level: &LogLevel) -> Result<()> {
    env_logger::Builder::from_default_env()
        .filter_level(log_level.to_log_level_filter())
        .init();
    Ok(())
}

/// Run the main application logic
pub fn run() -> Result<()> {
    let config = parse_args()?;

    // Initialize logging
    init_logging(&config.log_level)?;

    let solver = ExpressionSolver::new();

    info!(
        "Searching for expressions using digits '{}' that equal {}",
        config.digit_string, config.target
    );

    match solver.find_expression(&config.digit_string, config.target) {
        Some(expr) => {
            println!("{}", expr);
            Ok(())
        }
        None => {
            warn!("No matching expression found");
            println!("Unknown.");
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_digit_string() {
        // Test digit string validation
        let result = validate_digit_string("123");
        assert!(result.is_ok());

        let result = validate_digit_string("12a3");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_target_number() {
        let target: Result<f64, _> = "42.5".parse();
        assert!(target.is_ok());
        if let Ok(value) = target {
            assert_eq!(value, 42.5);
        }
    }

    #[test]
    fn test_cli_args_parsing() {
        // Test that we can create CliArgs with valid values
        let args = CliArgs {
            digit_string: "123".to_string(),
            target: 6.0,
            log_level: LogLevel::Warn,
        };

        assert_eq!(args.digit_string, "123");
        assert_eq!(args.target, 6.0);
        assert!(matches!(args.log_level, LogLevel::Warn));
    }

    #[test]
    fn test_log_level_conversion() {
        assert_eq!(
            LogLevel::Error.to_log_level_filter(),
            log::LevelFilter::Error
        );
        assert_eq!(LogLevel::Warn.to_log_level_filter(), log::LevelFilter::Warn);
        assert_eq!(LogLevel::Info.to_log_level_filter(), log::LevelFilter::Info);
        assert_eq!(
            LogLevel::Debug.to_log_level_filter(),
            log::LevelFilter::Debug
        );
        assert_eq!(
            LogLevel::Trace.to_log_level_filter(),
            log::LevelFilter::Trace
        );
    }
}
