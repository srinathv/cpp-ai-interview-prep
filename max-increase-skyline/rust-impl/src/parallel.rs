//! Multi-threaded parallel implementation using Rayon

use rayon::prelude::*;
use std::cmp::{max, min};

/// Multi-threaded implementation using Rayon
pub fn max_increase_parallel(grid: &[Vec<i32>]) -> i32 {
    if grid.is_empty() || grid[0].is_empty() {
        return 0;
    }

    let n = grid.len();
    
    // Calculate row maximums in parallel
    let row_max: Vec<i32> = grid
        .par_iter()
        .map(|row| row.iter().copied().max().unwrap_or(0))
        .collect();
    
    // Calculate column maximums in parallel
    let col_max: Vec<i32> = (0..n)
        .into_par_iter()
        .map(|j| (0..n).map(|i| grid[i][j]).max().unwrap_or(0))
        .collect();
    
    // Calculate total increase in parallel with reduction
    (0..n)
        .into_par_iter()
        .flat_map(|i| (0..n).into_par_iter().map(move |j| (i, j)))
        .map(|(i, j)| min(row_max[i], col_max[j]) - grid[i][j])
        .sum()
}

/// Advanced parallel version with custom chunking for better performance
pub fn max_increase_parallel_chunked(grid: &[Vec<i32>]) -> i32 {
    if grid.is_empty() || grid[0].is_empty() {
        return 0;
    }

    let n = grid.len();
    
    // Calculate row maximums in parallel
    let row_max: Vec<i32> = grid
        .par_iter()
        .map(|row| {
            // Process each row
            row.iter().copied().max().unwrap_or(0)
        })
        .collect();
    
    // Calculate column maximums in parallel
    let col_max: Vec<i32> = (0..n)
        .into_par_iter()
        .map(|j| {
            // Process each column
            grid.iter()
                .map(|row| row[j])
                .max()
                .unwrap_or(0)
        })
        .collect();
    
    // Calculate total using parallel chunks for better cache locality
    grid.par_iter()
        .enumerate()
        .map(|(i, row)| {
            // Process each row in parallel
            row.iter()
                .enumerate()
                .map(|(j, &val)| min(row_max[i], col_max[j]) - val)
                .sum::<i32>()
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn get_test_grid() -> Vec<Vec<i32>> {
        vec![
            vec![3, 0, 8, 4],
            vec![2, 4, 5, 7],
            vec![9, 2, 6, 3],
            vec![0, 3, 1, 0],
        ]
    }

    #[test]
    fn test_parallel() {
        let grid = get_test_grid();
        assert_eq!(max_increase_parallel(&grid), 35);
    }

    #[test]
    fn test_parallel_chunked() {
        let grid = get_test_grid();
        assert_eq!(max_increase_parallel_chunked(&grid), 35);
    }

    #[test]
    fn test_empty_grid() {
        let grid: Vec<Vec<i32>> = vec![];
        assert_eq!(max_increase_parallel(&grid), 0);
    }
}
