# Assembly Enhancements for DGEMM on AMD GPUs

This document covers assembly-level optimizations used in high-performance DGEMM implementations on AMD CDNA architectures (MI200/MI300).

## 1. MFMA (Matrix Fused Multiply-Add) Instructions

On AMD CDNA2+ (MI200/MI300), DGEMM uses hardware matrix cores via MFMA instructions.

### V_MFMA_F64_16x16x4F64

The primary MFMA instruction for double precision:
- **Output**: 16x16 matrix
- **Accumulation depth**: K=4
- **Operation**: D = A * B + C, where A is 16x4, B is 4x16, C/D are 16x16

```cpp
// Compiler intrinsic (preferred over inline assembly)
__device__ void mfma_dgemm_tile() {
    // d = A * B + C
    d = __builtin_amdgcn_mfma_f64_16x16x4f64(a, b, c, 0, 0, 0);
}
```

### Key MFMA Characteristics

| Property | Description |
|----------|-------------|
| **Wavefront-wide** | Entire 64-thread wavefront executes one MFMA |
| **Register distribution** | Input/output matrix elements distributed across wavefront lanes |
| **Latency hiding** | Mandatory cycles before using results (compiler handles with intrinsics) |

### MFMA Performance Table

| Instruction | Matrix Size | Cycles | Peak FP64 TFLOP/s (MI250X) |
|-------------|-------------|--------|---------------------------|
| V_MFMA_F64_16x16x4F64 | 16x16 | 64 | ~47.9 |
| V_MFMA_F64_4x4x4F64 | 4x4 (x4 batched) | 64 | ~47.9 |


## 2. Software Pipelining / Double Buffering

Assembly-level optimization implemented via Tensile code generation to overlap memory latency with compute:

```asm
// Pseudo-assembly flow:
// Stage 0: Load tile A[0], B[0] -> shared memory
// Stage 1: Load A[1], B[1] while computing on A[0], B[0]
// Stage 2: Load A[2], B[2] while computing on A[1], B[1]

buffer_load_dwordx4 v[0:3], ...  // Async load tile N+1
s_waitcnt vmcnt(1)                // Wait for tile N (not N+1)
v_mfma_f64_16x16x4f64 ...         // Compute on tile N
```

### Why Software Pipelining Matters
- Overlaps memory latency with compute
- Maintains high MFMA utilization despite memory-bound nature of DGEMM
- Hides the ~300-400 cycle global memory latency


## 3. Register Allocation and Bank Conflict Avoidance

Tensile performs architecture-specific assembly tuning:

```asm
// Source register combinations can cause bank conflicts
// Careful allocation ensures conflict-free access:
v_mfma_f64_16x16x4f64 a[0:15], v[16:17], v[24:25], a[0:15]
//                    ^dst      ^srcA     ^srcB    ^srcC
// Registers chosen to avoid same-bank simultaneous access
```

### Optimization Techniques
- **Register bank conflict elimination**: Different sources from different banks
- **Control code tuning**: Optimized instruction throughput
- **Strategic instruction placement**: Non-MFMA instructions positioned to avoid pipeline bubbles


## 4. Memory Access Optimization

### Global to LDS Transpose During Load

```cpp
// Avoids bank conflicts during MFMA feeding by transposing during load
// Assembly uses MFMA Transpose Load from LDS (CDNA4):
ds_read_tr_b64  // Transposed load directly into MFMA-ready layout
```

### Load Instruction Selection

| Instruction | Use Case |
|-------------|----------|
| `buffer_load` | Large, strided global memory access |
| `ds_read` | LDS with careful bank conflict avoidance |
| `dwordx4` loads | Vectorized for maximum bandwidth |


## 5. Tensile: ML-Guided Kernel Generation

rocBLAS uses Tensile internally for optimal kernel selection:

### How Tensile Works
1. Generates assembly kernels via machine learning for optimal tile sizes
2. Benchmarks thousands of configurations per problem size
3. Outputs YAML with winning kernels for runtime selection

### Example Tensile Configuration

```yaml
# Example Tensile output for DGEMM
- KernelName: Cijk_Alik_Bljk_DB_MT64x128x16
  ThreadTile: [8, 8]
  WorkGroup: [16, 16, 1]
  MacroTile: [64, 128]
  DepthU: 16
  MFMA: v_mfma_f64_16x16x4f64
```

### Tensile Naming Convention
- **Cijk**: C tensor with indices i, j, k
- **Alik**: A tensor layout (transposed)
- **Bljk**: B tensor layout
- **DB**: Double buffering enabled
- **MT64x128x16**: MacroTile dimensions


## 6. Wavefront Scheduling

### Occupancy vs Register Pressure Tradeoff

```
Higher register usage per thread
    -> Fewer concurrent wavefronts
    -> But larger tiles and better data reuse

Balance point for DGEMM:
    - ~128 VGPRs per thread
    - 4 wavefronts per CU
    - Sufficient to hide MFMA latency
```

### Instruction Interleaving

```asm
// Interleave MFMA with memory operations
v_mfma_f64_16x16x4f64 a[0:15], v[0:1], v[2:3], a[0:15]
buffer_load_dwordx4 v[4:7], ...    // Start next load during MFMA
v_mfma_f64_16x16x4f64 a[16:31], v[0:1], v[8:9], a[16:31]
buffer_load_dwordx4 v[10:13], ...  // Continue loading
```


## Summary Table

| Technique | Purpose | Example |
|-----------|---------|---------|
| MFMA intrinsics | Hardware matrix acceleration | `__builtin_amdgcn_mfma_f64_*` |
| Software pipelining | Overlap load/compute | Double-buffered shared memory |
| Register bank optimization | Eliminate conflicts | Tensile assembly tuning |
| Transpose during load | Bank conflict avoidance | `ds_read_tr_b64` |
| ML-guided kernel selection | Optimal tile sizes | Tensile benchmarking |


## Interview Talking Points

1. **Why MFMA over scalar FMA?**: MFMA achieves higher throughput by amortizing instruction fetch/decode overhead across 16x16 output elements

2. **Double buffering necessity**: Without it, the ~400 cycle memory latency would stall the 64-cycle MFMA, dropping utilization below 20%

3. **Register pressure in DGEMM**: Each thread holds part of the output tile in registers; larger tiles = better arithmetic intensity but fewer wavefronts

4. **Tensile vs hand-tuned assembly**: Tensile explores a larger search space than manual tuning and can find non-obvious optima
