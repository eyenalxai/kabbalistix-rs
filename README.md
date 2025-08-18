# Kabbalistix

A Rust command-line tool that finds mathematical expressions from digit strings that evaluate to a target value.

## Usage

```bash
kabbalistix <digit_string> <target> [--log-level <level>]
```

## Examples

```bash
# Find expressions using digits "123" that equal 6
kabbalistix 123 6.0
# Output: (1 + 2) × 3 = 6

# Find expressions using digits "327" that equal 3
kabbalistix 327 3.0
# Output: √3(27) = 3

# Find expressions using digits "2222222" that equal 14
kabbalistix 2222222 14.0
# Output: (((((2 + 2) + 2) + 2) + 2) + 2) + 2 = 14
```

## Supported Operations

- Addition (+), Subtraction (−), Multiplication (×), Division (÷)
- Exponentiation (^), Nth roots (√)
- Negation (−)
- Grouping with parentheses

The program generates all possible mathematical expressions using the provided digits and finds one that matches the target value.
