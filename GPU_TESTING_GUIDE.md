# GPU Testing Guide

This document outlines strategies for testing CUDA and HIP/ROCm code in this repository.

## Current Testing Status

- **OpenMP (CPU)**: Verified on macOS aarch64 (Apple Silicon) with g++-13
- **CUDA**: Not tested - requires NVIDIA GPU + CUDA toolkit
- **HIP/ROCm**: Not tested - requires AMD GPU + ROCm stack

## Directories Requiring GPU Testing

- `softmax-optimization/cuda/` - NVIDIA CUDA implementations
- `softmax-optimization/hip/` - AMD HIP implementations
- `trig-fusion/cuda/` - NVIDIA CUDA implementations
- `trig-fusion/hip/` - AMD HIP implementations

---

## Options for GPU Testing

### Option 1: Cloud GPU Instances

| Service | GPU Type | Notes |
|---------|----------|-------|
| Google Colab | NVIDIA T4/V100 | Free tier available, `nvcc` pre-installed |
| AWS EC2 | NVIDIA (p3/p4) or AMD | Pay per hour |
| Google Cloud | NVIDIA A100/V100 | Pay per hour |
| Lambda Labs | Various NVIDIA | Cheaper GPU cloud |

**Google Colab** is the easiest for quick CUDA testing - free and ready to use.

### Option 2: University/Work HPC Cluster

Many HPC clusters have NVIDIA or AMD GPU nodes available. Check with your institution.

### Option 3: Local AMD GPU (Linux)

ROCm supports consumer AMD GPUs (RX 6000/7000 series) on Linux. HIP code can also target NVIDIA GPUs via HIP's CUDA backend.

### Option 4: Compiler Syntax Check (Godbolt)

[Compiler Explorer](https://godbolt.org/) can verify CUDA/HIP code compiles, but won't execute it.

---

## Using Claude Code on a GPU System

### Method A: Direct Installation

If the GPU system has internet access and you can install software:

```bash
# Install Claude Code via npm
npm install -g @anthropic-ai/claude-code

# Run it
claude
```

**Requirements:**
- Node.js 18+
- Internet access (for API calls)
- Anthropic API key or Claude Pro/Max subscription

This gives full interactive debugging - Claude can compile, run, see errors, and fix them.

### Method B: SSH from Local Machine

If you can SSH into the GPU system:

```bash
# From your local machine, SSH with a pseudo-terminal
ssh -t user@gpu-server

# Then run claude on the remote system
claude
```

Or use **VS Code Remote SSH** with the Claude Code extension.

### Method C: Manual Testing + Error Reporting

If you cannot install Claude Code on the GPU system:

1. Copy the code to the GPU system (git clone or scp)
2. Compile and run manually:
   ```bash
   # CUDA
   nvcc -O3 -arch=sm_80 naive.cu -o naive && ./naive
   
   # HIP
   hipcc -O3 naive.hip -o naive && ./naive
   ```
3. If errors occur, copy the full error output
4. Paste errors into a Claude Code session on your local machine
5. Apply the suggested fixes manually

This works but involves more copy-paste overhead.

---

## Recommended Workflow

1. **If possible**: Install Claude Code on the GPU system (Method A or B)
   - Interactive debugging is much faster
   - Claude can iterate on fixes immediately

2. **Fallback**: Manual testing with error reporting (Method C)
   - Still effective, just slower iteration

3. **For CI/CD**: Consider GitHub Actions with self-hosted GPU runners
   - Automates testing on every push
   - Requires initial setup effort

---

## Compilation Commands Reference

### CUDA (NVIDIA)

```bash
cd softmax-optimization/cuda
nvcc -O3 -arch=sm_80 naive.cu -o naive && ./naive
nvcc -O3 -arch=sm_80 optimized.cu -o optimized && ./optimized
nvcc -O3 -arch=sm_80 super_optimized.cu -o super_optimized && ./super_optimized

cd trig-fusion/cuda
nvcc -O3 -arch=sm_80 --use_fast_math naive.cu -o naive && ./naive
nvcc -O3 -arch=sm_80 --use_fast_math optimized.cu -o optimized && ./optimized
nvcc -O3 -arch=sm_80 --use_fast_math super_optimized.cu -o super_optimized && ./super_optimized
```

Note: Adjust `-arch=sm_80` to match your GPU architecture:
- sm_70: V100
- sm_75: T4, RTX 20xx
- sm_80: A100, RTX 30xx
- sm_86: RTX 30xx (some models)
- sm_89: RTX 40xx
- sm_90: H100

### HIP (AMD)

```bash
cd softmax-optimization/hip
hipcc -O3 naive.hip -o naive && ./naive
hipcc -O3 optimized.hip -o optimized && ./optimized
hipcc -O3 super_optimized.hip -o super_optimized && ./super_optimized

cd trig-fusion/hip
hipcc -O3 --use_fast_math naive.hip -o naive && ./naive
hipcc -O3 --use_fast_math optimized.hip -o optimized && ./optimized
hipcc -O3 --use_fast_math super_optimized.hip -o super_optimized && ./super_optimized
```

---

## Expected Test Results

### Softmax

- **Sum of outputs**: Should equal 1.0 (probability distribution)
- **Numerical stability**: Optimized/super versions should handle inputs in range [0, 1000] without overflow

### Trig Fusion

- **Identity check**: sin²(x) + cos²(x) should equal 1.0 (error < 1e-6)
- **Sum**: Should match across all implementations for same input
