mod digits;
mod errors;
mod partitions;
mod validation;

pub use digits::digits_to_number;
pub use partitions::generate_partitions;
pub use validation::validate_digit_string;

#[cfg(test)]
mod tests;
