# Algorithms

Essential algorithms for technical interviews and AI/ML development.

## Topics Covered

### 1. Sorting Algorithms
- **sorting.cpp** - Bubble Sort, Quick Sort, Merge Sort, Heap Sort
- Time complexities: O(n²), O(n log n)
- Space complexities and when to use each

### 2. Searching Algorithms
- **searching.cpp** - Linear Search, Binary Search, Search in Rotated Array
- Binary search variants and applications
- Time complexity: O(log n) for binary search

### 3. Dynamic Programming
- **dynamic_programming.cpp** - Fibonacci, LCS, LIS, Knapsack, Coin Change, Edit Distance
- Memoization vs tabulation approaches
- Common DP patterns

## Complexity Quick Reference

### Sorting
| Algorithm | Best | Average | Worst | Space |
|-----------|------|---------|-------|-------|
| Bubble    | O(n) | O(n²)   | O(n²) | O(1)  |
| Quick     | O(n log n) | O(n log n) | O(n²) | O(log n) |
| Merge     | O(n log n) | O(n log n) | O(n log n) | O(n) |
| Heap      | O(n log n) | O(n log n) | O(n log n) | O(1) |

### Searching
| Algorithm | Time | Requirements |
|-----------|------|--------------|
| Linear    | O(n) | None |
| Binary    | O(log n) | Sorted array |

## Common Interview Patterns

1. **Two Pointers** - Used in sorted arrays, palindromes
2. **Sliding Window** - Subarray/substring problems
3. **Dynamic Programming** - Optimization, counting problems
4. **Divide & Conquer** - Merge sort, quick sort, binary search
5. **Greedy** - Local optimal choices leading to global optimal

## Tips for Interviews

- Always clarify input constraints
- Discuss time/space tradeoffs
- Start with brute force, then optimize
- Test with edge cases
- Consider sorted vs unsorted data
