# GPU-Memory-Optimization-Shared-Memory-Tiling-for-a-7-7-Blur
A small GPU experiment to understand when shared memory is beneficial.  Implemented a naive and a shared-memory tiled blur kernel, validated correctness against a CPU reference, and measured performance on an NVIDIA T4.  For larger kernels (7×7), shared memory delivered a meaningful speedup through better data reuse.
