"""Plots generated from real experiment data. No numbers are typed in by hand."""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

LABEL_COLORS = {
    "clear improvement": "#2e7d32",
    "borderline": "#f9a825",
    "within noise": "#9e9e9e",
    "same as baseline": "#bdbdbd",
    "clear regression": "#c62828",
    "no valid config": "#000000",
}


def fmt_n(n) -> str:
    n = int(n)
    if n >= 1_000_000:
        return f"{n / 1e6:.0f}M"
    if n >= 1_000:
        return f"{n / 1e3:.0f}K"
    return str(n)


def _workload_labels(df):
    return [f"{k} {fmt_n(n)}" for k, n in zip(df["kernel"], df["n"])]


def plot_config_bars(ax, df_rec, kernel, n, baseline=(1024, 4)):
    """Bar chart: median latency of every successful config for one workload."""
    d = df_rec[(df_rec["kernel"] == kernel) & (df_rec["n"] == n) & (df_rec["status"] == "ok")]
    d = d.sort_values("latency_ms")
    if len(d) == 0:
        ax.text(0.5, 0.5, "no successful configurations", ha="center", transform=ax.transAxes)
        return
    lat = d["latency_ms"].to_numpy(dtype=float)
    lo = lat - d["latency_min_ms"].to_numpy(dtype=float)
    hi = d["latency_max_ms"].to_numpy(dtype=float) - lat
    is_base = ((d["block_size"] == baseline[0]) & (d["num_warps"] == baseline[1])).to_numpy()
    colors = ["#c62828" if b else "#1976d2" for b in is_base]
    x = np.arange(len(d))
    bars = ax.bar(x, lat, yerr=[lo, hi], capsize=3, color=colors)
    ax.bar_label(bars, fmt="%.4f", fontsize=7, padding=3)
    ax.set_xticks(x)
    ax.set_xticklabels([f"B{b}\nW{w}" for b, w in zip(d["block_size"], d["num_warps"])], fontsize=8)
    ax.set_xlabel("configuration (B = block size, W = warps), fastest first")
    ax.set_ylabel("median kernel latency (ms)")
    ax.set_title(f"{kernel}, n={int(n):,}: every configuration\n"
                 "(red = baseline; whiskers = min..max of repeated runs)")


def plot_latency_vs_size(ax, df_sum):
    """Line chart: baseline vs tuned latency across input sizes (log-log)."""
    for kernel, g in df_sum.groupby("kernel"):
        g = g.sort_values("n")
        (line,) = ax.plot(g["n"], g["baseline_ms"], "--o", label=f"{kernel} (baseline)")
        ax.plot(g["n"], g["tuned_ms"], "-s", color=line.get_color(), label=f"{kernel} (tuned)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("input size n (elements)")
    ax.set_ylabel("median kernel latency (ms)")
    ax.set_title("Kernel latency vs input size\n(dashed = baseline, solid = tuned; both axes log)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)


def plot_search_time(ax, df_sum):
    """Stacked bars: how long tuning took for each workload."""
    df = df_sum.reset_index(drop=True)
    x = np.arange(len(df))
    search = df["tuning_time_s"].to_numpy(dtype=float)
    confirm = df["confirm_time_s"].to_numpy(dtype=float)
    ax.bar(x, search, color="#1976d2", label="search (compile + verify + timing)")
    ax.bar(x, confirm, bottom=search, color="#ef6c00", label="baseline-vs-best confirmation")
    ax.set_xticks(x)
    ax.set_xticklabels(_workload_labels(df), rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("seconds")
    ax.set_title("Tuning cost per workload (exhaustive search)")
    ax.legend(fontsize=8)


def plot_improvement(ax, df_sum):
    """Improvement over baseline, coloured by classification, with noise whiskers."""
    df = df_sum.reset_index(drop=True)
    x = np.arange(len(df))
    imp = df["improvement_pct"].to_numpy(dtype=float)
    noise = df["noise_pct"].fillna(0).to_numpy(dtype=float)
    colors = [LABEL_COLORS.get(l, "#9e9e9e") for l in df["label"]]
    ax.bar(x, imp, color=colors)
    ax.errorbar(x, imp, yerr=noise, fmt="none", ecolor="black", capsize=3, linewidth=1)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(_workload_labels(df), rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("latency improvement over baseline (%)")
    ax.set_title("Improvement vs baseline (whiskers = +/- noise estimate)")
    present = [l for l in LABEL_COLORS if l in set(df["label"])]
    ax.legend(handles=[Patch(color=LABEL_COLORS[l], label=l) for l in present], fontsize=8)


def make_dashboard(df_rec, df_sum, headline, path, baseline=(1024, 4)):
    """2x2 figure saved to `path`. `headline` = (kernel, n) for the per-config bar chart."""
    gpu = df_rec["gpu"].iloc[0]
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    plot_config_bars(axes[0, 0], df_rec, headline[0], headline[1], baseline)
    plot_latency_vs_size(axes[0, 1], df_sum)
    plot_search_time(axes[1, 0], df_sum)
    plot_improvement(axes[1, 1], df_sum)
    fig.suptitle(f"Triton kernel auto-tuning on {gpu}: measured results", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=150)
    return fig
