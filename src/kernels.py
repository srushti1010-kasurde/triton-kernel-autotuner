"""Kernel Library: Triton kernels, launch wrappers, and a registry (KernelSpec)."""
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


# ================================================================
# Triton kernels
# ================================================================
@triton.jit
def vector_add_kernel(a_ptr, b_ptr, c_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """C[i] = A[i] + B[i]"""
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    tl.store(c_ptr + offsets, a + b, mask=mask)


@triton.jit
def elementwise_mul_kernel(a_ptr, b_ptr, c_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """C[i] = A[i] * B[i]"""
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    tl.store(c_ptr + offsets, a * b, mask=mask)


@triton.jit
def block_sum_kernel(x_ptr, partial_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """partial[pid] = sum of the BLOCK_SIZE elements owned by program `pid`."""
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    # other=0.0 is essential: masked-out lanes must contribute 0 to the sum,
    # otherwise the last (partial) block would add garbage.
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    tl.store(partial_ptr + pid, tl.sum(x, axis=0))


# ================================================================
# Launch wrappers: (inputs tuple, block_size, num_warps, out) -> out
# ================================================================
def _check_cuda_contiguous(*tensors: torch.Tensor) -> None:
    for t in tensors:
        if not t.is_cuda:
            raise ValueError("Inputs must be CUDA tensors.")
        if not t.is_contiguous():
            raise ValueError("Inputs must be contiguous.")
    if len({(t.shape, t.dtype) for t in tensors}) != 1:
        raise ValueError("Inputs must share shape and dtype.")


def launch_vector_add(inputs, block_size, num_warps, out):
    a, b = inputs
    _check_cuda_contiguous(a, b)
    n = a.numel()
    grid = (triton.cdiv(n, block_size),)
    vector_add_kernel[grid](a, b, out, n, BLOCK_SIZE=block_size, num_warps=num_warps)
    return out


def launch_elementwise_mul(inputs, block_size, num_warps, out):
    a, b = inputs
    _check_cuda_contiguous(a, b)
    n = a.numel()
    grid = (triton.cdiv(n, block_size),)
    elementwise_mul_kernel[grid](a, b, out, n, BLOCK_SIZE=block_size, num_warps=num_warps)
    return out


def launch_block_sum(inputs, block_size, num_warps, out):
    (x,) = inputs
    _check_cuda_contiguous(x)
    n = x.numel()
    grid = (triton.cdiv(n, block_size),)
    block_sum_kernel[grid](x, out, n, BLOCK_SIZE=block_size, num_warps=num_warps)
    return out


# Convenience wrapper used in the smoke test
def vector_add(a, b, block_size: int = 1024, num_warps: int = 4, out=None):
    if out is None:
        out = torch.empty_like(a)
    return launch_vector_add((a, b), block_size, num_warps, out)


# ================================================================
# Input generation, output allocation, trusted references
# ================================================================
def _make_inputs_n(num_inputs):
    def make(n: int, dtype=torch.float32, seed: int = 0):
        g = torch.Generator(device="cuda")
        g.manual_seed(seed)                       # reproducible inputs
        return tuple(torch.randn(n, device="cuda", dtype=dtype, generator=g)
                     for _ in range(num_inputs))
    return make


def _alloc_like_input(inputs, block_size):
    return torch.empty_like(inputs[0])


def _alloc_partials(inputs, block_size):
    n_blocks = triton.cdiv(inputs[0].numel(), block_size)
    return torch.empty(n_blocks, device="cuda", dtype=inputs[0].dtype)


def _ref_add(inputs, block_size):
    return inputs[0] + inputs[1]


def _ref_mul(inputs, block_size):
    return inputs[0] * inputs[1]


def _ref_block_sum(inputs, block_size):
    """float64 per-block sums computed by PyTorch (zero-padded last block)."""
    x = inputs[0].double()
    n_blocks = triton.cdiv(x.numel(), block_size)
    padded = F.pad(x, (0, n_blocks * block_size - x.numel()))
    return padded.view(n_blocks, block_size).sum(dim=1)


# ================================================================
# Registry
# ================================================================
@dataclass(frozen=True)
class KernelSpec:
    name: str
    version: str            # bump when the kernel code changes (used by the cache later)
    launch: Callable        # (inputs, block_size, num_warps, out) -> out
    make_inputs: Callable   # (n, dtype, seed) -> tuple of CUDA tensors
    alloc_output: Callable  # (inputs, block_size) -> tensor
    reference: Callable     # (inputs, block_size) -> trusted expected output
    bytes_moved: Callable   # (n, itemsize, block_size) -> bytes read + written
    rtol: float
    atol: float


REGISTRY = {
    "vector_add": KernelSpec(
        name="vector_add", version="1", launch=launch_vector_add,
        make_inputs=_make_inputs_n(2), alloc_output=_alloc_like_input,
        reference=_ref_add, bytes_moved=lambda n, s, b: 3 * n * s,
        rtol=1e-5, atol=1e-6),
    "elementwise_mul": KernelSpec(
        name="elementwise_mul", version="1", launch=launch_elementwise_mul,
        make_inputs=_make_inputs_n(2), alloc_output=_alloc_like_input,
        reference=_ref_mul, bytes_moved=lambda n, s, b: 3 * n * s,
        rtol=1e-5, atol=1e-6),
    "block_sum": KernelSpec(
        name="block_sum", version="1", launch=launch_block_sum,
        make_inputs=_make_inputs_n(1), alloc_output=_alloc_partials,
        reference=_ref_block_sum,
        bytes_moved=lambda n, s, b: n * s + ((n + b - 1) // b) * s,
        rtol=1e-4, atol=1e-3),   # float32 sums are not bit-exact vs float64
}
