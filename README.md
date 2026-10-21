
# Triton-Based Kernel Auto-Tuning Framework

Searches block size and warp count for Triton GPU kernels, benchmarks each valid
configuration on an NVIDIA GPU, checks correctness against PyTorch, and caches the
best configuration in JSON.

**Status:** See STATUS.md for what is done and what is not.

## Requirements
An NVIDIA GPU (developed on a Colab Tesla T4), PyTorch, Triton, pandas, numpy, matplotlib.
It will not run on a machine without CUDA (it stops with a clear error).

## Run on Google Colab
1. Runtime > Change runtime type > T4 GPU
2. In a cell:
       !git clone https://github.com/<srushti1010-kasurde>/triton-kernel-autotuner /content/triton-kernel-autotuner
3. Then:
       import sys; sys.path.insert(0, "/content/triton-kernel-autotuner/src")
       from environment import print_environment_info, get_environment_info
       print_environment_info(get_environment_info())

## Structure
src/ (modules), results/ (outputs from real runs), notebooks/ (demo notebook), STATUS.md
