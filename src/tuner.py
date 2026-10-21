"""Auto-Tuning Engine: exhaustive search over launch configurations."""
import datetime
import time
from dataclasses import dataclass
from typing import Optional

import torch

from benchmark import BenchSettings, benchmark_config, head_to_head, measure_end_to_end
from config_generator import Config, grid_size
from environment import require_gpu
from verify import verify_config

DEFAULT_BASELINE = Config(block_size=1024, num_warps=4)


def _now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


@dataclass
class TuneResult:
    kernel: str
    n: int
    gpu: str
    algorithm: str
    records: list                       # one dict per attempted config (incl. failures)
    best: Optional[Config]
    attempted: int
    successful: int
    failed: int
    tuning_time_s: float                # search phase only (includes compile + verify + timing)
    baseline: Config
    confirm_time_s: float = 0.0         # head-to-head re-measurement (reported separately)
    search_best_ms: Optional[float] = None   # best latency seen DURING the search
    baseline_ms: Optional[float] = None      # head-to-head median
    tuned_ms: Optional[float] = None         # head-to-head median
    baseline_spread_pct: Optional[float] = None
    tuned_spread_pct: Optional[float] = None
    improvement_pct: Optional[float] = None
    verdict: str = ""
    e2e_baseline_ms: Optional[float] = None
    e2e_tuned_ms: Optional[float] = None

    def summary_row(self) -> dict:
        return {
            "kernel": self.kernel, "n": self.n, "gpu": self.gpu,
            "algorithm": self.algorithm,
            "best_block": self.best.block_size if self.best else None,
            "best_warps": self.best.num_warps if self.best else None,
            "attempted": self.attempted, "successful": self.successful, "failed": self.failed,
            "search_best_ms": self.search_best_ms,
            "baseline_cfg": self.baseline.label(),
            "baseline_ms": self.baseline_ms, "tuned_ms": self.tuned_ms,
            "improvement_pct": self.improvement_pct,
            "baseline_spread_pct": self.baseline_spread_pct,
            "tuned_spread_pct": self.tuned_spread_pct,
            "verdict": self.verdict,
            "tuning_time_s": self.tuning_time_s, "confirm_time_s": self.confirm_time_s,
            "e2e_baseline_ms": self.e2e_baseline_ms, "e2e_tuned_ms": self.e2e_tuned_ms,
        }


def evaluate_config(spec, inputs, cfg: Config, settings: BenchSettings,
                    gpu: str, algorithm: str) -> dict:
    """Verify one config, then benchmark it if (and only if) it is correct.
    Returns a flat record. Failures are recorded, never dropped."""
    n = inputs[0].numel()
    rec = {
        "kernel": spec.name, "kernel_version": spec.version, "n": n, "gpu": gpu,
        "block_size": cfg.block_size, "num_warps": cfg.num_warps,
        "grid": grid_size(n, cfg), "search_algorithm": algorithm,
        "trials": settings.repeats, "timestamp": _now(),
        "compiled": False, "correct": False, "status": "failed",
        "first_call_ms": None, "latency_ms": None, "latency_min_ms": None,
        "latency_max_ms": None, "latency_std_ms": None, "spread_pct": None,
        "bandwidth_gbs": None, "error_type": "", "error_message": "",
    }
    v = verify_config(spec, inputs, cfg)
    rec.update(compiled=v.compiled, correct=v.correct, first_call_ms=v.first_call_ms,
               error_type=v.error_type, error_message=v.error_message)
    if not (v.compiled and v.correct):
        return rec                                   # recorded as failed, no timing

    try:
        st = benchmark_config(spec, inputs, cfg, settings)
    except Exception as e:                           # timing itself failed
        rec.update(error_type=type(e).__name__, error_message=str(e)[:300])
        return rec

    rec.update(status="ok", latency_ms=st["median_ms"], latency_min_ms=st["min_ms"],
               latency_max_ms=st["max_ms"], latency_std_ms=st["std_ms"],
               spread_pct=st["spread_pct"], bandwidth_gbs=st["bandwidth_gbs"])
    return rec


def _finalize(spec, inputs, n, gpu, algorithm, records, tuning_time_s,
              settings, baseline) -> TuneResult:
    """Rank, pick the best, then compare against the baseline head-to-head."""
    ok = sorted((r for r in records if r["status"] == "ok"), key=lambda r: r["latency_ms"])
    for rank, r in enumerate(ok, start=1):
        r["rank"] = rank
    for r in records:
        r.setdefault("rank", None)

    res = TuneResult(kernel=spec.name, n=n, gpu=gpu, algorithm=algorithm, records=records,
                     best=None, attempted=len(records), successful=len(ok),
                     failed=len(records) - len(ok), tuning_time_s=tuning_time_s,
                     baseline=baseline)
    if not ok:
        res.verdict = "no valid configuration found"
        return res

    best = Config(ok[0]["block_size"], ok[0]["num_warps"])
    res.best = best
    res.search_best_ms = ok[0]["latency_ms"]

    # The baseline must itself be correct, otherwise comparing against it is meaningless.
    if not verify_config(spec, inputs, baseline).correct:
        res.verdict = "baseline config failed verification; no comparison made"
        return res

    t0 = time.perf_counter()
    base_st, best_st = head_to_head(spec, inputs, baseline, best, settings)
    res.confirm_time_s = time.perf_counter() - t0
    res.baseline_ms, res.tuned_ms = base_st["median_ms"], best_st["median_ms"]
    res.baseline_spread_pct, res.tuned_spread_pct = base_st["spread_pct"], best_st["spread_pct"]
    res.improvement_pct = 100.0 * (res.baseline_ms - res.tuned_ms) / res.baseline_ms
    res.e2e_baseline_ms = measure_end_to_end(spec, inputs, baseline)
    res.e2e_tuned_ms = measure_end_to_end(spec, inputs, best)

    noise = max(res.baseline_spread_pct, res.tuned_spread_pct)
    if best == baseline:
        res.verdict = "best config equals baseline (difference shown is pure measurement noise)"
    elif abs(res.improvement_pct) <= noise:
        res.verdict = f"difference is within measurement noise (~{noise:.1f}% spread)"
    elif res.improvement_pct > 0:
        res.verdict = "tuned config measurably faster than baseline"
    else:
        res.verdict = "tuned config measurably SLOWER than baseline"
    return res


def exhaustive_search(spec, n: int, configs, settings: BenchSettings = BenchSettings(),
                      baseline: Config = DEFAULT_BASELINE, dtype=torch.float32,
                      seed: int = 0) -> TuneResult:
    """Test EVERY candidate configuration. The best is the fastest among the valid
    configurations tested for this kernel, size and GPU (not a proven global optimum)."""
    gpu = require_gpu()["gpu_name"]                  # refuses to run without a GPU
    inputs = spec.make_inputs(n, dtype, seed)

    t0 = time.perf_counter()
    records = [evaluate_config(spec, inputs, cfg, settings, gpu, "exhaustive")
               for cfg in configs]
    tuning_time_s = time.perf_counter() - t0

    return _finalize(spec, inputs, n, gpu, "exhaustive", records, tuning_time_s,
                     settings, baseline)
