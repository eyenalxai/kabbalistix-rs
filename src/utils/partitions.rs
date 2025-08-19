use log::debug;

/// This uses an iterative approach to avoid stack overflow with large ranges.
pub fn generate_partitions(
    start: usize,
    end: usize,
    num_blocks: usize,
) -> Vec<Vec<(usize, usize)>> {
    debug!(
        "Generating partitions for range {}..{} with {} blocks",
        start, end, num_blocks
    );

    if num_blocks == 1 {
        return vec![vec![(start, end)]];
    }

    if num_blocks > (end - start) {
        return vec![];
    }

    let mut result = Vec::new();

    let mut stack = Vec::new();
    stack.push((Vec::new(), start, num_blocks));

    while let Some((current_partition, remaining_start, remaining_blocks)) = stack.pop() {
        if remaining_blocks == 1 {
            let mut partition = current_partition;
            partition.push((remaining_start, end));
            result.push(partition);
            continue;
        }

        let min_end = remaining_start + 1;
        let max_end = end - (remaining_blocks - 1);

        for split_point in min_end..=max_end {
            let mut new_partition = current_partition.clone();
            new_partition.push((remaining_start, split_point));
            stack.push((new_partition, split_point, remaining_blocks - 1));
        }
    }

    debug!("Generated {} partitions", result.len());
    result
}
