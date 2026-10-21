"""Benchmarking: steady-state kernel timing, kept separate from compile time."""
import statistics
import time
from dataclasses import dataclass

import torch
import triton.testing as ttesting


@dataclass(frozen=True)
class BenchSettings:
    warmup_ms: float = 25.0      # do_bench warm-up duration (milliseconds, NOT iterations)
    rep_ms: float = 50.0         # do_bench measurement duration per repeat (milliseconds)
    repeats: int = 5             # independent do_bench runs per configuration (search phase)
    confirm_repeats: int = 15    # runs per config in the final head-to-head re-measurement


def _stats(samples):
    med = statistics.median(samples)
    spread = 100.0 * (max(samples) - min(samples)) / med if med > 0 else 0.0
    return {
        "median_ms": med,
        "min_ms": min(samples),
        "max_ms": max(samples),
        "std_ms": statistics.pstdev(samples) if len(samples) > 1 else 0.0,
        "spread_pct": spread,          # (max-min)/median: a conservative noise estimate
        "samples": list(samples),
    }


def time_kernel_once(spec, inputs, cfg, out, settings: BenchSettings) -> float:
    """One do_bench run -> median kernel time in ms.

    do_bench times with CUDA events and synchronizes internally, so we do not add
    our own synchronize here. Only the kernel launch is inside the timed lambda.
    """
    fn = lambda: spec.launch(inputs, cfg.block_size, cfg.num_warps, out)
    return ttesting.do_bench(fn, warmup=settings.warmup_ms, rep=settings.rep_ms,
                             return_mode="median")


def benchmark_config(spec, inputs, cfg, settings: BenchSettings):
    """Repeat do_bench several times; return robust statistics + bandwidth."""
    out = spec.alloc_output(inputs, cfg.block_size)     # allocated OUTSIDE the timed region
    samples = [time_kernel_once(spec, inputs, cfg, out, settings)
               for _ in range(settings.repeats)]
    st = _stats(samples)
    n = inputs[0].numel()
    nbytes = spec.bytes_moved(n, inputs[0].element_size(), cfg.block_size)
    st["bandwidth_gbs"] = nbytes / (st["median_ms"] * 1e-3) / 1e9
    return st


def head_to_head(spec, inputs, cfg_a, cfg_b, settings: BenchSettings):
    """Re-measure two configs INTERLEAVED (A,B,A,B,...) so slow drift in GPU clocks
    or contention affects both equally. Returns (stats_a, stats_b)."""
    out_a = spec.alloc_output(inputs, cfg_a.block_size)
    out_b = spec.alloc_output(inputs, cfg_b.block_size)
    sa, sb = [], []
    for _ in range(settings.confirm_repeats):
        sa.append(time_kernel_once(spec, inputs, cfg_a, out_a, settings))
        sb.append(time_kernel_once(spec, inputs, cfg_b, out_b, settings))
    return _stats(sa), _stats(sb)


def measure_end_to_end(spec, inputs, cfg, iters: int = 50) -> float:
    """End-to-end latency (ms): allocate output + Python launch + GPU run + synchronize.
    This is what an application experiences, and it is NOT the same as kernel latency."""
    for _ in range(3):                                   # warm-up
        spec.launch(inputs, cfg.block_size, cfg.num_warps,
                    spec.alloc_output(inputs, cfg.block_size))
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        torch.cuda.synchronize()                         # start from an idle GPU
        t0 = time.perf_counter()
        out = spec.alloc_output(inputs, cfg.block_size)
        spec.launch(inputs, cfg.block_size, cfg.num_warps, out)
        torch.cuda.synchronize()                         # wait until the GPU finishes
        samples.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(samples)
