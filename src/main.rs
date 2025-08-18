use std::collections::HashMap;
use std::env;
use std::fmt;

#[derive(Debug, Clone)]
enum Expression {
    Number(f64),
    Add(Box<Expression>, Box<Expression>),
    Sub(Box<Expression>, Box<Expression>),
    Mul(Box<Expression>, Box<Expression>),
    Div(Box<Expression>, Box<Expression>),
    Pow(Box<Expression>, Box<Expression>),
    Neg(Box<Expression>),
    NthRoot(Box<Expression>, Box<Expression>), // NthRoot(n, a) = a^(1/n)
}

impl fmt::Display for Expression {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Expression::Number(n) => write!(f, "{}", n),
            Expression::Add(l, r) => write!(f, "({} + {})", l, r),
            Expression::Sub(l, r) => write!(f, "({} - {})", l, r),
            Expression::Mul(l, r) => write!(f, "({} × {})", l, r),
            Expression::Div(l, r) => write!(f, "({} ÷ {})", l, r),
            Expression::Pow(l, r) => write!(f, "({} ^ {})", l, r),
            Expression::Neg(e) => write!(f, "(-{})", e),
            Expression::NthRoot(n, a) => write!(f, "√{}({})", n, a),
        }
    }
}

impl Expression {
    fn evaluate(&self) -> Result<f64, String> {
        match self {
            Expression::Number(n) => Ok(*n),
            Expression::Add(l, r) => {
                let left = l.evaluate()?;
                let right = r.evaluate()?;
                Ok(left + right)
            }
            Expression::Sub(l, r) => {
                let left = l.evaluate()?;
                let right = r.evaluate()?;
                Ok(left - right)
            }
            Expression::Mul(l, r) => {
                let left = l.evaluate()?;
                let right = r.evaluate()?;
                Ok(left * right)
            }
            Expression::Div(l, r) => {
                let left = l.evaluate()?;
                let right = r.evaluate()?;
                if right == 0.0 {
                    Err("Division by zero".to_string())
                } else {
                    Ok(left / right)
                }
            }
            Expression::Pow(l, r) => {
                let left = l.evaluate()?;
                let right = r.evaluate()?;
                if left < 0.0 && right.fract() != 0.0 {
                    Err("Complex result from negative base with fractional exponent".to_string())
                } else {
                    Ok(left.powf(right))
                }
            }
            Expression::Neg(e) => {
                let val = e.evaluate()?;
                Ok(-val)
            }
            Expression::NthRoot(n, a) => {
                let n_val = n.evaluate()?;
                let a_val = a.evaluate()?;
                if n_val < 2.0 || n_val.fract() != 0.0 {
                    Err("Root index must be an integer >= 2".to_string())
                } else if a_val < 0.0 && (n_val as i32) % 2 == 0 {
                    Err("Even root of negative number".to_string())
                } else {
                    Ok(a_val.powf(1.0 / n_val))
                }
            }
        }
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        eprintln!("Usage: {} <digit_string> <target_number>", args[0]);
        std::process::exit(1);
    }

    let digit_string = &args[1];
    let target: f64 = match args[2].parse() {
        Ok(t) => t,
        Err(_) => {
            eprintln!("Invalid target number: {}", args[2]);
            std::process::exit(1);
        }
    };

    // Validate digit string contains only digits
    if !digit_string.chars().all(|c| c.is_ascii_digit()) {
        eprintln!("Digit string must contain only digits: {}", digit_string);
        std::process::exit(1);
    }

    if digit_string.is_empty() {
        eprintln!("Digit string cannot be empty");
        std::process::exit(1);
    }

    match find_expression(digit_string, target) {
        Some(expr) => println!("{}", expr),
        None => println!("Unknown."),
    }
}

// Generate all possible ways to partition a range of digits into consecutive blocks
fn generate_partitions(start: usize, end: usize, num_blocks: usize) -> Vec<Vec<(usize, usize)>> {
    if num_blocks == 1 {
        return vec![vec![(start, end)]];
    }

    let mut result = Vec::new();
    for split_point in start + 1..end {
        let first_block = (start, split_point);
        for mut rest in generate_partitions(split_point, end, num_blocks - 1) {
            let mut partition = vec![first_block];
            partition.append(&mut rest);
            result.push(partition);
        }
    }
    result
}

// Convert a digit range to a number
fn digits_to_number(digits: &str, start: usize, end: usize) -> f64 {
    digits[start..end].parse::<f64>().unwrap_or(0.0)
}

// Memoization cache for expressions
type ExprCache = HashMap<(usize, usize), Vec<Expression>>;

// Generate all possible expressions from a digit string with memoization
fn generate_expressions_memo(
    digits: &str,
    start: usize,
    end: usize,
    cache: &mut ExprCache,
) -> Vec<Expression> {
    if start >= end || start >= digits.len() || end > digits.len() {
        return Vec::new();
    }

    // Check cache first
    if let Some(cached) = cache.get(&(start, end)) {
        return cached.clone();
    }

    let length = end - start;
    let mut expressions = Vec::new();

    // Base case: single number (always include this)
    let num = digits_to_number(digits, start, end);
    expressions.push(Expression::Number(num));

    if length >= 2 {
        // Binary operations (need 2 blocks)
        for partition in generate_partitions(start, end, 2) {
            let (start1, end1) = partition[0];
            let (start2, end2) = partition[1];

            let left_exprs = generate_expressions_memo(digits, start1, end1, cache);
            let right_exprs = generate_expressions_memo(digits, start2, end2, cache);

            for left in &left_exprs {
                for right in &right_exprs {
                    expressions.push(Expression::Add(
                        Box::new(left.clone()),
                        Box::new(right.clone()),
                    ));
                    expressions.push(Expression::Sub(
                        Box::new(left.clone()),
                        Box::new(right.clone()),
                    ));
                    expressions.push(Expression::Mul(
                        Box::new(left.clone()),
                        Box::new(right.clone()),
                    ));
                    expressions.push(Expression::Div(
                        Box::new(left.clone()),
                        Box::new(right.clone()),
                    ));
                    expressions.push(Expression::Pow(
                        Box::new(left.clone()),
                        Box::new(right.clone()),
                    ));
                }
            }
        }

        // Nth root (need 2 blocks: first for n, second for a)
        for partition in generate_partitions(start, end, 2) {
            let (start1, end1) = partition[0];
            let (start2, end2) = partition[1];

            // First block must form an integer >= 2 for the root index
            let n_num = digits_to_number(digits, start1, end1);
            if n_num >= 2.0 && n_num.fract() == 0.0 && n_num <= 10.0 {
                // Limit root degree to prevent excessive computation
                let n_expr = Expression::Number(n_num);
                let a_exprs = generate_expressions_memo(digits, start2, end2, cache);

                for a_expr in &a_exprs {
                    expressions.push(Expression::NthRoot(
                        Box::new(n_expr.clone()),
                        Box::new(a_expr.clone()),
                    ));
                }
            }
        }

        // Unary negation (only for composite expressions to avoid redundancy)
        for i in 1..expressions.len() {
            // Skip the first expression which is just the number
            let expr = &expressions[i];
            expressions.push(Expression::Neg(Box::new(expr.clone())));
        }
    }

    // Cache the result
    cache.insert((start, end), expressions.clone());
    expressions
}

// Wrapper function that creates cache and calls memoized version
fn generate_expressions(digits: &str, start: usize, end: usize) -> Vec<Expression> {
    let mut cache = HashMap::new();
    generate_expressions_memo(digits, start, end, &mut cache)
}

fn find_expression(digits: &str, target: f64) -> Option<Expression> {
    const EPSILON: f64 = 1e-9;

    let all_expressions = generate_expressions(digits, 0, digits.len());
    let total_expressions = all_expressions.len();

    eprintln!("Generated {} expressions to evaluate", total_expressions);

    let mut evaluated_count = 0;
    let mut valid_count = 0;

    for expr in all_expressions {
        evaluated_count += 1;
        if let Ok(value) = expr.evaluate() {
            valid_count += 1;
            if (value - target).abs() < EPSILON {
                eprintln!(
                    "Found match after evaluating {} expressions ({} valid)",
                    evaluated_count, valid_count
                );
                return Some(expr);
            }
        }
    }

    eprintln!(
        "No match found. Evaluated {} expressions ({} valid)",
        evaluated_count, valid_count
    );
    None
}
