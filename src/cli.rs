#![deny(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

use crate::solver::{ExpressionSolver, SolverConfig};
use crate::utils::validate_digit_string;
use anyhow::{Context, Result};
use std::env;

/// Configuration for the CLI application
pub struct CliConfig {
    pub digit_string: String,
    pub target: f64,
    pub solver_config: SolverConfig,
}

/// Parse command line arguments and return configuration
pub fn parse_args() -> Result<CliConfig> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        let program_name = args.first().map(|s| s.as_str()).unwrap_or("program");
        anyhow::bail!("Usage: {} <digit_string> <target_number>", program_name);
    }

    let digit_string = args
        .get(1)
        .context("Missing digit string argument")?
        .clone();
    let target_str = args.get(2).context("Missing target number argument")?;
    let target: f64 = target_str
        .parse()
        .with_context(|| format!("Invalid target number: {}", target_str))?;

    // Validate digit string
    validate_digit_string(&digit_string).context("Invalid digit string")?;

    Ok(CliConfig {
        digit_string,
        target,
        solver_config: SolverConfig::default(),
    })
}

/// Run the main application logic
pub fn run() -> Result<()> {
    let config = parse_args()?;
    let solver = ExpressionSolver::new(config.solver_config);

    match solver.find_expression(&config.digit_string, config.target) {
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
    fn test_parse_args_valid() {
        // This test is conceptual since we can't easily mock env::args()
        // In a real scenario, you might want to refactor parse_args to accept
        // arguments as parameters for better testability
        let result = validate_digit_string("123");
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_target_number() {
        let target: Result<f64, _> = "42.5".parse();
        assert!(target.is_ok());
        if let Ok(value) = target {
            assert_eq!(value, 42.5);
        }
    }
}
