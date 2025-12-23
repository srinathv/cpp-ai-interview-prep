//! Maximum Increase Keeping Skyline - Rust Implementation
//!
//! This library provides multiple implementations of the skyline problem
//! with varying levels of optimization.

use std::cmp::{max, min};

/// Single-threaded baseline implementation
pub fn max_increase_single_threaded(grid: &[Vec<i32>]) -> i32 {
    if grid.is_empty() || grid[0].is_empty() {
        return 0;
    }

    let n = grid.len();
    
    // Calculate row maximums
    let row_max: Vec<i32> = grid
        .iter()
        .map(|row| *row.iter().max().unwrap_or(&0))
        .collect();
    
    // Calculate column maximums
    let col_max: Vec<i32> = (0..n)
        .map(|j| {
            (0..n)
                .map(|i| grid[i][j])
                .max()
                .unwrap_or(0)
        })
        .collect();
    
    // Calculate total increase
    let mut total = 0;
    for i in 0..n {
        for j in 0..n {
            total += min(row_max[i], col_max[j]) - grid[i][j];
        }
    }
    
    total
}

/// Optimized single-threaded version with iterator chains
pub fn max_increase_optimized(grid: &[Vec<i32>]) -> i32 {
    if grid.is_empty() || grid[0].is_empty() {
        return 0;
    }

    let n = grid.len();
    
    // Calculate row maximums using iterators
    let row_max: Vec<i32> = grid
        .iter()
        .map(|row| row.iter().copied().max().unwrap_or(0))
        .collect();
    
    // Calculate column maximums
    let col_max: Vec<i32> = (0..n)
        .map(|j| (0..n).map(|i| grid[i][j]).max().unwrap_or(0))
        .collect();
    
    // Calculate total increase using functional style
    (0..n)
        .flat_map(|i| (0..n).map(move |j| (i, j)))
        .map(|(i, j)| min(row_max[i], col_max[j]) - grid[i][j])
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
    fn test_single_threaded() {
        let grid = get_test_grid();
        assert_eq!(max_increase_single_threaded(&grid), 35);
    }

    #[test]
    fn test_optimized() {
        let grid = get_test_grid();
        assert_eq!(max_increase_optimized(&grid), 35);
    }

    #[test]
    fn test_empty_grid() {
        let grid: Vec<Vec<i32>> = vec![];
        assert_eq!(max_increase_single_threaded(&grid), 0);
    }

    #[test]
    fn test_single_cell() {
        let grid = vec![vec![5]];
        assert_eq!(max_increase_single_threaded(&grid), 0);
    }
}
