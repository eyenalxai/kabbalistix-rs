#![deny(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

use crate::solver::{ExpressionSolver, SolverConfig};
use crate::utils::validate_digit_string;
use anyhow::{Context, Result};
use clap::Parser;

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

    /// Maximum degree for nth root operations (default: 10)
    #[arg(long, default_value = "10.0")]
    pub max_root_degree: f64,

    /// Epsilon for floating point comparison (default: 1e-9)
    #[arg(long, default_value = "1e-9")]
    pub epsilon: f64,

    /// Enable verbose output
    #[arg(short, long)]
    pub verbose: bool,
}

/// Configuration for the CLI application
pub struct CliConfig {
    pub digit_string: String,
    pub target: f64,
    pub solver_config: SolverConfig,
    pub verbose: bool,
}

/// Parse command line arguments and return configuration
pub fn parse_args() -> Result<CliConfig> {
    let args = CliArgs::parse();

    // Validate digit string
    validate_digit_string(&args.digit_string).context("Invalid digit string")?;

    let solver_config = SolverConfig {
        max_root_degree: args.max_root_degree,
        epsilon: args.epsilon,
    };

    Ok(CliConfig {
        digit_string: args.digit_string,
        target: args.target,
        solver_config,
        verbose: args.verbose,
    })
}

/// Run the main application logic
pub fn run() -> Result<()> {
    let config = parse_args()?;
    let solver_config = config.solver_config;

    // Pass verbose flag to solver (we'll need to modify solver to handle this)
    let solver = ExpressionSolver::new(solver_config);

    if config.verbose {
        eprintln!(
            "Searching for expressions using digits '{}' that equal {}",
            config.digit_string, config.target
        );
        eprintln!(
            "Configuration: max_root_degree={}, epsilon={}",
            solver.config().max_root_degree,
            solver.config().epsilon
        );
    }

    match solver.find_expression_with_verbose(&config.digit_string, config.target, config.verbose) {
        Some(expr) => {
            println!("{}", expr);
            Ok(())
        }
        None => {
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
            max_root_degree: 10.0,
            epsilon: 1e-9,
            verbose: false,
        };

        assert_eq!(args.digit_string, "123");
        assert_eq!(args.target, 6.0);
        assert!(!args.verbose);
    }

    #[test]
    fn test_solver_config_creation() {
        let solver_config = SolverConfig {
            max_root_degree: 5.0,
            epsilon: 1e-6,
        };

        assert_eq!(solver_config.max_root_degree, 5.0);
        assert_eq!(solver_config.epsilon, 1e-6);
    }
}
