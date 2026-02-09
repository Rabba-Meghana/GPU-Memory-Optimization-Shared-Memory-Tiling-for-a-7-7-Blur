# GPU Memory Optimization: Naive vs Shared-Memory Tiling (7×7 Blur)

This project compares a **naive global-memory** CUDA blur kernel with a **tiled shared-memory** version (halo loading + synchronization).  
Correctness is validated against a CPU reference, and performance is measured using CUDA events on an **NVIDIA T4 (sm_75)**.

## What this demonstrates
- GPU memory hierarchy in practice (global vs shared memory)
- Tiling + halo loading for stencil/convolution-like workloads
- Correctness validation (CPU reference vs GPU output)
- Performance measurement with CUDA events (avg kernel time)

## Results (example on NVIDIA T4)
For a 7×7 blur (radius = 3), shared-memory tiling achieves a clear speedup over the naive kernel due to higher data reuse and reduced global memory traffic.

Example output:
- Correctness: `Max diff = 0` for both kernels
- Speedup: ~`2.3×` (varies by run/environment)

## Files
- `blur_shared.cu` — CUDA implementation (CPU reference + naive GPU + shared-memory tiled GPU + timing + results logging)
- `results.txt` — recorded timings from the latest run (`naive_ms`, `shared_ms`, `speedup`)
- `input.pgm`, `blurred.pgm` — saved images (PGM format)
- `gpu_blur_final.png` — visualization suitable for sharing

## Requirements
- NVIDIA GPU (tested on **T4**)
- `nvcc` (CUDA toolkit)
- For visualization: Python + `numpy`, `matplotlib`

## Build & Run (CUDA)
Compile for T4:
```bash
nvcc -O3 -std=c++17 -arch=sm_75 blur_shared.cu -o blur_shared
./blur_shared
