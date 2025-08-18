use crate::utils::{digits_to_number, generate_partitions, validate_digit_string};

#[test]
fn test_generate_partitions_single_block() {
    let partitions = generate_partitions(0, 3, 1);
    assert_eq!(partitions, vec![vec![(0, 3)]]);
}

#[test]
fn test_generate_partitions_two_blocks() {
    let partitions = generate_partitions(0, 3, 2);
    let expected = vec![vec![(0, 2), (2, 3)], vec![(0, 1), (1, 3)]];
    assert_eq!(partitions, expected);
}

#[test]
fn test_generate_partitions_three_blocks() {
    let partitions = generate_partitions(0, 4, 3);
    let expected = vec![
        vec![(0, 2), (2, 3), (3, 4)],
        vec![(0, 1), (1, 3), (3, 4)],
        vec![(0, 1), (1, 2), (2, 4)],
    ];
    assert_eq!(partitions, expected);
}

#[test]
fn test_generate_partitions_impossible() {
    let partitions = generate_partitions(0, 3, 4);
    assert_eq!(partitions, Vec::<Vec<(usize, usize)>>::new());
}

#[test]
fn test_generate_partitions_large_range() {
    let partitions = generate_partitions(0, 20, 5);
    assert!(!partitions.is_empty());
    for partition in &partitions {
        assert_eq!(partition.len(), 5);
        if let (Some(first), Some(last)) = (partition.first(), partition.last()) {
            assert_eq!(first.0, 0);
            assert_eq!(last.1, 20);
        }
        for i in 0..4 {
            if let (Some(current), Some(next)) = (partition.get(i), partition.get(i + 1)) {
                assert_eq!(current.1, next.0);
            }
        }
    }
}

#[test]
fn test_digits_to_number() {
    let result = digits_to_number("12345", 0, 3);
    assert!(result.is_ok());
    if let Ok(value) = result {
        assert_eq!(value, 123.0);
    }

    let result = digits_to_number("12345", 2, 5);
    assert!(result.is_ok());
    if let Ok(value) = result {
        assert_eq!(value, 345.0);
    }

    let result = digits_to_number("12345", 1, 4);
    assert!(result.is_ok());
    if let Ok(value) = result {
        assert_eq!(value, 234.0);
    }
}

#[test]
fn test_digits_to_number_invalid_range() {
    let result = digits_to_number("12345", 0, 10);
    assert!(result.is_err());
    let result = digits_to_number("12345", 5, 3);
    assert!(result.is_err());
}

#[test]
fn test_validate_digit_string_valid() {
    assert!(validate_digit_string("12345").is_ok());
    assert!(validate_digit_string("0").is_ok());
    assert!(validate_digit_string("999").is_ok());
}

#[test]
fn test_validate_digit_string_invalid() {
    assert!(validate_digit_string("").is_err());
    assert!(validate_digit_string("12a45").is_err());
    assert!(validate_digit_string("12.45").is_err());
    assert!(validate_digit_string("12-45").is_err());
}
