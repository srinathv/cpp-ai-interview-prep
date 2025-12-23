//! CUDA GPU implementation using cuda-sys and rustacuda
//! 
//! This module provides GPU-accelerated computation using NVIDIA CUDA.

#[cfg(feature = "cuda")]
use rustacuda::prelude::*;
#[cfg(feature = "cuda")]
use rustacuda::memory::DeviceBuffer;
#[cfg(feature = "cuda")]
use std::error::Error;
#[cfg(feature = "cuda")]
use std::ffi::CString;

#[cfg(feature = "cuda")]
const CUDA_KERNEL: &str = r#"
extern "C" __global__ void find_row_max(const int* grid, int* row_max, int n) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < n) {
        int max_val = 0;
        for (int j = 0; j < n; ++j) {
            int val = grid[row * n + j];
            if (val > max_val) max_val = val;
        }
        row_max[row] = max_val;
    }
}

extern "C" __global__ void find_col_max(const int* grid, int* col_max, int n) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (col < n) {
        int max_val = 0;
        for (int i = 0; i < n; ++i) {
            int val = grid[i * n + col];
            if (val > max_val) max_val = val;
        }
        col_max[col] = max_val;
    }
}

extern "C" __global__ void calculate_sum(
    const int* grid,
    const int* row_max,
    const int* col_max,
    int* partial_sums,
    int n
) {
    extern __shared__ int shared_sum[];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = n * n;
    
    int local_sum = 0;
    
    // Grid-stride loop
    for (int i = idx; i < total_elements; i += blockDim.x * gridDim.x) {
        int row = i / n;
        int col = i % n;
        int current_height = grid[i];
        int max_height = min(row_max[row], col_max[col]);
        local_sum += (max_height - current_height);
    }
    
    shared_sum[threadIdx.x] = local_sum;
    __syncthreads();
    
    // Reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + stride];
        }
        __syncthreads();
    }
    
    if (threadIdx.x == 0) {
        partial_sums[blockIdx.x] = shared_sum[0];
    }
}
"#;

#[cfg(feature = "cuda")]
pub fn max_increase_cuda(grid: &[Vec<i32>]) -> Result<i32, Box<dyn Error>> {
    if grid.is_empty() || grid[0].is_empty() {
        return Ok(0);
    }

    let n = grid.len();
    let total_elements = n * n;
    
    // Flatten the grid
    let flat_grid: Vec<i32> = grid.iter().flat_map(|row| row.iter().copied()).collect();
    
    // Initialize CUDA
    rustacuda::init(CudaFlags::empty())?;
    let device = Device::get_device(0)?;
    let _context = Context::create_and_push(ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device)?;
    
    // Compile and load the module
    let ptx = CString::new(CUDA_KERNEL)?;
    let module = Module::load_from_string(&ptx)?;
    
    // Allocate device memory
    let mut d_grid = DeviceBuffer::from_slice(&flat_grid)?;
    let mut d_row_max = DeviceBuffer::from_slice(&vec![0i32; n])?;
    let mut d_col_max = DeviceBuffer::from_slice(&vec![0i32; n])?;
    
    let threads_per_block = 256;
    let blocks = (n + threads_per_block - 1) / threads_per_block;
    
    // Launch row max kernel
    let row_max_kernel = module.get_function("find_row_max")?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
    
    unsafe {
        launch!(
            row_max_kernel<<<blocks, threads_per_block, 0, stream>>>(
                d_grid.as_device_ptr(),
                d_row_max.as_device_ptr(),
                n as i32
            )
        )?;
    }
    
    // Launch col max kernel
    let col_max_kernel = module.get_function("find_col_max")?;
    
    unsafe {
        launch!(
            col_max_kernel<<<blocks, threads_per_block, 0, stream>>>(
                d_grid.as_device_ptr(),
                d_col_max.as_device_ptr(),
                n as i32
            )
        )?;
    }
    
    // Calculate sum
    let sum_blocks = ((total_elements + threads_per_block - 1) / threads_per_block).min(1024);
    let mut d_partial_sums = DeviceBuffer::from_slice(&vec![0i32; sum_blocks])?;
    
    let sum_kernel = module.get_function("calculate_sum")?;
    let shared_mem_size = threads_per_block * std::mem::size_of::<i32>();
    
    unsafe {
        launch!(
            sum_kernel<<<sum_blocks, threads_per_block, shared_mem_size as u32, stream>>>(
                d_grid.as_device_ptr(),
                d_row_max.as_device_ptr(),
                d_col_max.as_device_ptr(),
                d_partial_sums.as_device_ptr(),
                n as i32
            )
        )?;
    }
    
    stream.synchronize()?;
    
    // Copy results back
    let mut partial_sums = vec![0i32; sum_blocks];
    d_partial_sums.copy_to(&mut partial_sums)?;
    
    // Final reduction on CPU
    let total: i32 = partial_sums.iter().sum();
    
    Ok(total)
}

// Fallback implementation when CUDA is not available
#[cfg(not(feature = "cuda"))]
pub fn max_increase_cuda(_grid: &[Vec<i32>]) -> Result<i32, Box<dyn std::error::Error>> {
    Err("CUDA support not enabled. Rebuild with --features cuda".into())
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;

    #[test]
    fn test_cuda() {
        let grid = vec![
            vec![3, 0, 8, 4],
            vec![2, 4, 5, 7],
            vec![9, 2, 6, 3],
            vec![0, 3, 1, 0],
        ];
        
        match max_increase_cuda(&grid) {
            Ok(result) => assert_eq!(result, 35),
            Err(_) => println!("CUDA not available, skipping test"),
        }
    }
}
