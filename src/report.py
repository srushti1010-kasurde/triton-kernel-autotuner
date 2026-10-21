"""Result classification and exports (best_configurations.json, experiment_report.md)."""
import json

import pandas as pd

from cache import now_iso

NOISE_FLOOR_PCT = 1.0   # a spread of 0% from a coarse timer is not real stability


# ------------------------------------------------------------------ helpers
def _py(x):
    """Convert numpy/pandas scalars to plain Python (None for NaN) for JSON."""
    if x is None:
        return None
    try:
        if pd.isna(x):
            return None
    except (TypeError, ValueError):
        pass
    return x.item() if hasattr(x, "item") else x


def _cfg(b, w) -> str:
    return "n/a" if pd.isna(b) else f"B{int(b)}_W{int(w)}"


def _md_table(columns, rows) -> str:
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


# ------------------------------------------------------------ classification
def classify_improvement(improvement_pct, baseline_spread_pct, tuned_spread_pct, same_config):
    """Return (label, noise_pct). noise = max(both spreads, 1% floor)."""
    if improvement_pct is None or pd.isna(improvement_pct):
        return "no valid config", float("nan")
    spreads = [v for v in (baseline_spread_pct, tuned_spread_pct)
               if v is not None and not pd.isna(v)]
    noise = max(spreads + [NOISE_FLOOR_PCT])
    if same_config:
        return "same as baseline", noise
    mag = abs(improvement_pct)
    if mag <= noise:
        return "within noise", noise
    if mag <= 2 * noise:
        return "borderline", noise
    return ("clear improvement" if improvement_pct > 0 else "clear regression"), noise


def add_labels(df_sum):
    """Add `label` and `noise_pct` columns to a summary DataFrame."""
    df = df_sum.copy()
    labels, noises = [], []
    for r in df.itertuples():
        if pd.isna(r.best_block):
            lab, nz = "no valid config", float("nan")
        else:
            same = _cfg(r.best_block, r.best_warps) == r.baseline_cfg
            lab, nz = classify_improvement(r.improvement_pct, r.baseline_spread_pct,
                                           r.tuned_spread_pct, same)
        labels.append(lab)
        noises.append(nz)
    df["label"] = labels
    df["noise_pct"] = noises
    return df


def stability_table(df_all):
    """Per workload: which config won in each run, and whether the winner was stable."""
    rows = []
    for (kernel, n), g in df_all.sort_values("run").groupby(["kernel", "n"], sort=True):
        g = g.dropna(subset=["best_block"])
        if g.empty:
            continue
        rows.append({
            "kernel": kernel, "n": int(n), "runs": len(g),
            "best_per_run": ", ".join(_cfg(b, w) for b, w in zip(g["best_block"], g["best_warps"])),
            "improvement_pct_per_run": ", ".join(f"{x:+.1f}" for x in g["improvement_pct"]),
            "same_winner_every_run": len(g[["best_block", "best_warps"]].drop_duplicates()) == 1,
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------- exports
def build_best_configurations(df_sum, df_rec, stab, env):
    out = []
    for r in df_sum.reset_index(drop=True).itertuples():
        recs = df_rec[(df_rec["kernel"] == r.kernel) & (df_rec["n"] == r.n)
                      & (df_rec["status"] == "ok")].sort_values("latency_ms")
        near = recs[recs["latency_ms"] <= recs["latency_ms"].min() * 1.03]
        s = stab[(stab["kernel"] == r.kernel) & (stab["n"] == r.n)]
        out.append({
            "kernel": r.kernel,
            "input_size": _py(r.n),
            "gpu": env["gpu_name"],
            "algorithm": r.algorithm,
            "best_config": {"block_size": _py(r.best_block), "num_warps": _py(r.best_warps)},
            "meaning": "fastest among the valid configurations tested for this kernel, size and GPU",
            "baseline_config": r.baseline_cfg,
            "baseline_ms": _py(r.baseline_ms),
            "tuned_ms": _py(r.tuned_ms),
            "improvement_pct": _py(r.improvement_pct),
            "noise_pct": _py(r.noise_pct),
            "classification": r.label,
            "configs_within_3pct_of_best_search_latency":
                [_cfg(b, w) for b, w in zip(near["block_size"], near["num_warps"])],
            "same_winner_in_all_runs": bool(s["same_winner_every_run"].iloc[0]) if len(s) else None,
            "configs_attempted": _py(r.attempted),
            "configs_successful": _py(r.successful),
            "configs_failed": _py(r.failed),
            "tuning_time_s": _py(r.tuning_time_s),
        })
    return out


def save_best_configurations(path, entries, env):
    payload = {
        "generated_at": now_iso(),
        "environment": {k: env.get(k) for k in (
            "gpu_name", "compute_capability", "torch_version", "triton_version",
            "torch_cuda_build", "python_version")},
        "workloads": entries,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)


def write_report(path, env, df_rec, df_sum, stab, df_hits, settings, baseline_label):
    df_sum = df_sum.reset_index(drop=True)
    hit_s = {(r.kernel, int(r.n)): r.hit_s for r in df_hits.itertuples()}
    algos = ", ".join(sorted(df_sum["algorithm"].unique()))
    blocks = sorted(int(b) for b in df_rec["block_size"].unique())
    warps = sorted(int(w) for w in df_rec["num_warps"].unique())
    n_runs = int(df_rec["run"].nunique())
    L = []

    L.append("# Triton Kernel Auto-Tuning: Experiment Report\n")
    L.append(f"_Generated {now_iso()}. Every number below was measured on this GPU; "
             "nothing is simulated._\n")

    L.append("## 1. Setup\n")
    L.append(f"- GPU: {env['gpu_name']} ({env['gpu_memory_gb']} GB, compute capability "
             f"{env['compute_capability']})")
    L.append(f"- PyTorch {env['torch_version']}, Triton {env['triton_version']}, "
             f"Python {env['python_version']}")
    L.append(f"- Search algorithm(s): {algos}. Search space: block size in {blocks} "
             f"x num_warps in {warps}")
    L.append(f"- Timing: `triton.testing.do_bench` (CUDA events, L2 flushed), "
             f"{settings.warmup_ms:g} ms warm-up and {settings.rep_ms:g} ms measurement per run; "
             f"{settings.repeats} runs per config in the search (median reported); "
             f"{settings.confirm_repeats} interleaved runs per config in the final "
             "baseline-vs-best comparison")
    L.append(f"- Baseline: {baseline_label}. Improvement % is relative to this choice only.")
    L.append("- Improvement (%) = (baseline latency - tuned latency) / baseline latency x 100, "
             "taken from the interleaved head-to-head re-measurement, not from the search phase "
             "(this avoids the winner's curse).")
    L.append(f"- The full sweep was repeated {n_runs} times; section 4 shows whether the winner "
             "stayed the same.\n")

    L.append("![benchmark results](benchmark_results.png)\n")

    L.append("## 2. Results (run 1)\n")
    rows = [[r.kernel, f"{int(r.n):,}", _cfg(r.best_block, r.best_warps),
             f"{r.baseline_ms:.4f}", f"{r.tuned_ms:.4f}", f"{r.improvement_pct:+.2f}%",
             f"{r.noise_pct:.1f}%", r.label] for r in df_sum.itertuples()]
    L.append(_md_table(["kernel", "n", "best (fastest tested)", "baseline ms", "tuned ms",
                        "improvement", "noise", "classification"], rows))
    L.append("")
    L.append("Classification rule: noise = max(spread of baseline runs, spread of tuned runs, 1%), "
             "where spread = (max - min) / median across repeated runs. "
             "`within noise` means |improvement| <= noise; `borderline` means up to 2x noise; "
             "`clear` means more than 2x noise.\n")
    counts = df_sum["label"].value_counts()
    L.append(f"**Summary:** of {len(df_sum)} workloads, "
             + ", ".join(f"{v} {k}" for k, v in counts.items()) + ".")
    clear = df_sum[df_sum["label"] == "clear improvement"]
    if len(clear):
        L.append("Clear improvements: " + "; ".join(
            f"{r.kernel} n={int(r.n):,} ({r.improvement_pct:+.1f}%)" for r in clear.itertuples()) + ".")
    L.append("Small inputs (about 10K elements) run for only a few microseconds, so launch "
             "overhead and timer resolution dominate and differences there should not be "
             "interpreted as tuning gains.\n")

    L.append("## 3. Near-ties\n")
    L.append("`best_configurations.json` lists, for each workload, every configuration within 3% "
             "of the best search-phase latency. Several configurations are often "
             "indistinguishable, so the defensible claim is that a group of configurations "
             "is fast, not that one specific configuration is uniquely optimal.\n")

    L.append("## 4. Run-to-run stability of the winner\n")
    st_rows = [[r.kernel, f"{r.n:,}", r.runs, r.best_per_run, r.improvement_pct_per_run,
                "yes" if r.same_winner_every_run else "no"] for r in stab.itertuples()]
    L.append(_md_table(["kernel", "n", "runs", "winner per run", "improvement % per run",
                        "same winner every run"], st_rows))
    n_unstable = int((~stab["same_winner_every_run"].astype(bool)).sum())
    L.append(f"\n{n_unstable} of {len(stab)} workloads had a different winner in at least one run, "
             "which is expected when configurations are near-tied.\n")

    L.append("## 5. Tuning cost, cache reuse and amortization\n")
    rows = []
    for r in df_sum.itertuples():
        saving_ms = r.baseline_ms - r.tuned_ms
        cost_s = r.tuning_time_s + r.confirm_time_s
        if r.label == "clear improvement" and saving_ms > 0:
            be = f"{cost_s / (saving_ms * 1e-3):,.0f}"
        else:
            be = "n/a (no reliable saving)"
        hs = hit_s.get((r.kernel, int(r.n)))
        rows.append([r.kernel, f"{int(r.n):,}", f"{r.tuning_time_s:.2f}", f"{r.confirm_time_s:.2f}",
                     "n/a" if hs is None else f"{hs * 1e3:.1f}",
                     f"{saving_ms * 1e3:+.2f}", be])
    L.append(_md_table(["kernel", "n", "search s", "confirmation s",
                        "cache hit total ms", "kernel time saved per launch (us)",
                        "break-even launches"], rows))
    L.append("\nBreak-even = (search + confirmation time) / kernel time saved per launch, shown "
             "only for clear improvements. A cache hit reuses a stored configuration and runs "
             "one correctness launch; it does not produce a new benchmark measurement. Compile "
             "time is only partly included because Triton had cached some kernels earlier in "
             "the session.\n")

    L.append("## 6. Kernel latency vs end-to-end latency\n")
    rows = [[r.kernel, f"{int(r.n):,}", f"{r.baseline_ms:.4f}", f"{r.tuned_ms:.4f}",
             f"{r.e2e_baseline_ms:.4f}", f"{r.e2e_tuned_ms:.4f}",
             f"{r.e2e_baseline_ms / r.baseline_ms:.0f}x"] for r in df_sum.itertuples()]
    L.append(_md_table(["kernel", "n", "kernel baseline ms", "kernel tuned ms",
                        "end-to-end baseline ms", "end-to-end tuned ms",
                        "end-to-end / kernel (baseline)"], rows))
    L.append("\nEnd-to-end latency includes output allocation, Python launch overhead and GPU "
             "synchronization, so it is larger than kernel latency and improves by a smaller "
             "percentage. It is a single median over 50 iterations without a spread estimate.\n")

    L.append("## 7. Limitations\n")
    L.append("- The selected configuration is the fastest among the valid configurations tested "
             "for the given kernel, size and GPU. It is not a proven global optimum.")
    L.append("- Improvement depends on the baseline chosen. A different baseline gives different "
             "percentages.")
    L.append("- Measurements come from one shared Colab T4. Clock changes and contention add "
             "noise, and results do not transfer to other GPUs (the cache key includes the GPU).")
    L.append("- Spread is a rough noise estimate, not a statistical confidence interval.")
    L.append("- Only float32, one-dimensional memory-bound kernels were tuned. The block-sum "
             "kernel tunes the per-block stage only; the final combination of partial sums is "
             "not part of the tuned kernel.")
    L.append("- Only block size and warp count are tuned.")

    text = "\n".join(L) + "\n"
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return text
