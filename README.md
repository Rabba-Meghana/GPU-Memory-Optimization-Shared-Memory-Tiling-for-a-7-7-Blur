# GPU Memory Optimization: Naive vs Shared-Memory Tiling (7×7 Blur)

This project explores **when and why shared memory improves GPU performance** by comparing a naive global-memory blur kernel with a tiled shared-memory implementation.

Correctness is validated against a CPU reference, and performance is measured using CUDA events on an **NVIDIA T4 GPU**.

---

## Overview

The goal of this project is to understand:
- GPU memory hierarchy in practice
- When shared memory overhead is worth paying
- How data reuse impacts performance for stencil-like workloads

A 7×7 blur (radius = 3) is used as the primary case study.

---

## Key Concepts Demonstrated

- Global memory vs shared memory access
- Tiling and halo loading
- Thread synchronization
- CPU–GPU correctness verification
- Empirical performance measurement

---

## Results (NVIDIA T4)

- Correctness:
  - `Max diff (CPU vs GPU naive) = 0`
  - `Max diff (CPU vs GPU shared) = 0`

- Performance:
  - Naive kernel: ~1.14 ms  
  - Shared-memory kernel: ~0.45 ms  
  - Speedup: ~**2.3×**

> Note: Exact timings vary by system and runtime conditions.

---

## File Structure

- `blur_shared.cu`  
  CUDA implementation including:
  - CPU reference blur
  - Naive GPU kernel
  - Shared-memory tiled GPU kernel
  - Timing with CUDA events
  - Result logging

- `results.txt`  
  Recorded execution metrics:
  - `naive_ms`
  - `shared_ms`
  - `speedup`

- `input.pgm`  
  Original input image

- `blurred.pgm`  
  Output from shared-memory blur kernel

- `gpu_blur_final.png`  
  Visualization suitable for sharing

---

## Requirements

- NVIDIA GPU (tested on **T4**)
- CUDA Toolkit (`nvcc`)
- Python (for visualization)
  - `numpy`
  - `matplotlib`

---

## Build & Run

Compile for NVIDIA T4:

```bash
nvcc -O3 -std=c++17 -arch=sm_75 blur_shared.cu -o blur_shared
./blur_shared
````

For other GPUs, replace the architecture flag:

```bash
nvcc -O3 -std=c++17 -arch=sm_<your_compute_capability> blur_shared.cu -o blur_shared
```

---

## Notes

* For small kernels (e.g., 3×3), shared-memory overhead can outweigh benefits.
* For larger kernels (e.g., 7×7), increased data reuse makes shared memory effective.
* This project emphasizes **measurement over assumptions**.

---

## Possible Extensions

* Sweep kernel radius and plot speedup vs window size
* Compare with separable blur (two-pass implementation)
* Explore different tile sizes and occupancy trade-offs
* Extend to multi-channel images (RGB)
