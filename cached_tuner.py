"""Cache-aware tuning: reuse a stored result when it is safe, otherwise search and store."""
import time
from dataclasses import dataclass, asdict
from typing import Optional

import torch

from benchmark import BenchSettings
from cache import TuningCache, build_key_components, now_iso
from config_generator import Config
from environment import require_gpu
from tuner import DEFAULT_BASELINE, TuneResult, exhaustive_search
from verify import verify_config

# Later phases register more strategies here (e.g. "random", "halving").
SEARCH_FUNCTIONS = {"exhaustive": exhaustive_search}


@dataclass
class CachedTuneOutcome:
    status: str                      # hit | miss | stale_retuned | forced_retune | no_valid_config
    best: Optional[Config]
    entry: Optional[dict]            # the stored record (measured/search metadata)
    result: Optional[TuneResult]     # set ONLY when a new search actually ran
    lookup_ms: float
    total_s: float
    message: str


def _entry_from_result(components, res: TuneResult, settings, env) -> dict:
    return {
        "key_components": components,
        "best_config": res.best.to_dict(),
        "measured": {                        # timings measured at `created_at`
            "search_best_ms": res.search_best_ms,
            "baseline_config": res.baseline.to_dict(),
            "baseline_ms": res.baseline_ms,
            "tuned_ms": res.tuned_ms,
            "improvement_pct": res.improvement_pct,
            "baseline_spread_pct": res.baseline_spread_pct,
            "tuned_spread_pct": res.tuned_spread_pct,
            "verdict": res.verdict,
            "e2e_baseline_ms": res.e2e_baseline_ms,
            "e2e_tuned_ms": res.e2e_tuned_ms,
        },
        "search": {
            "algorithm": res.algorithm, "attempted": res.attempted,
            "successful": res.successful, "failed": res.failed,
            "tuning_time_s": res.tuning_time_s, "confirm_time_s": res.confirm_time_s,
        },
        "bench_settings": asdict(settings),
        "environment": {k: env.get(k) for k in (
            "gpu_name", "gpu_memory_gb", "compute_capability", "torch_version",
            "triton_version", "torch_cuda_build", "python_version")},
        "created_at": now_iso(), "hits": 0, "last_hit_at": None,
    }


def tune_with_cache(cache: TuningCache, spec, n: int, configs,
                    algorithm: str = "exhaustive",
                    settings: BenchSettings = BenchSettings(),
                    baseline: Config = DEFAULT_BASELINE, dtype=torch.float32,
                    verify_on_hit: bool = True,
                    force_retune: bool = False) -> CachedTuneOutcome:
    if algorithm not in SEARCH_FUNCTIONS:
        raise ValueError(f"Unknown algorithm '{algorithm}'. Available: {sorted(SEARCH_FUNCTIONS)}")

    t_start = time.perf_counter()
    env = require_gpu()
    components = build_key_components(spec, n, dtype, env, configs, baseline, algorithm)

    t0 = time.perf_counter()
    key, entry, miss_reason = cache.lookup(components)
    lookup_ms = (time.perf_counter() - t0) * 1e3

    status, reason = "miss", miss_reason
    if force_retune:
        status, reason, entry = "forced_retune", "force_retune=True", None

    if entry is not None and verify_on_hit:
        stored = Config(entry["best_config"]["block_size"], entry["best_config"]["num_warps"])
        v = verify_config(spec, spec.make_inputs(n, dtype, 0), stored)
        if not (v.compiled and v.correct):
            status = "stale_retuned"
            reason = f"stored config failed re-verification ({v.error_type}: {v.error_message})"
            entry = None

    # ---------------- HIT: reuse the recorded configuration ----------------
    if entry is not None:
        cache.record_hit(key)
        best = Config(entry["best_config"]["block_size"], entry["best_config"]["num_warps"])
        msg = (f"CACHE HIT: reusing the configuration recorded on {entry['created_at']}. "
               f"The timings stored in the entry were measured THEN, not now. No new benchmark "
               f"was run" + (" (only a one-launch correctness check)." if verify_on_hit else "."))
        return CachedTuneOutcome("hit", best, entry, None, lookup_ms,
                                 time.perf_counter() - t_start, msg)

    # ---------------- MISS: run the search, then store the result ----------------
    cache.record_miss()
    res = SEARCH_FUNCTIONS[algorithm](spec, n, configs, settings=settings,
                                      baseline=baseline, dtype=dtype)
    if res.best is None:
        return CachedTuneOutcome("no_valid_config", None, None, res, lookup_ms,
                                 time.perf_counter() - t_start,
                                 f"CACHE MISS ({reason}); no valid configuration found, nothing stored.")
    entry = _entry_from_result(components, res, settings, env)
    cache.insert(key, entry)
    msg = (f"CACHE MISS ({reason}): ran a NEW {algorithm} search and stored the result. "
           f"These timings are fresh measurements.")
    return CachedTuneOutcome(status, res.best, entry, res, lookup_ms,
                             time.perf_counter() - t_start, msg)
