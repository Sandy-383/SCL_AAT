"""
Convergence Plot Generator.

Produces publication-quality plots for:
  1. Objective convergence curves (f1..f4 vs iteration)
  2. Hypervolume indicator over iterations
  3. Speedup bar chart (CUDA-GWO vs baselines)
  4. Pareto front scatter plot (f1 vs f2, f1 vs f3)
  5. Algorithm comparison radar chart
"""

import os
import logging
import numpy as np
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    from matplotlib.gridspec import GridSpec
    _MPL = True
except ImportError:
    _MPL = False
    logger.warning("matplotlib not available — plotting disabled")


# ── Color palette (colorblind-safe) ──────────────────────────────────────────
COLORS = {
    "cuda_gwo" : "#0077BB",
    "cpu_gwo"  : "#EE7733",
    "pso"      : "#009988",
    "ga"       : "#CC3311",
    "aco"      : "#AA3377",
}
LABELS = {
    "cuda_gwo" : "CUDA-GWO (Ours)",
    "cpu_gwo"  : "CPU-GWO",
    "pso"      : "PSO",
    "ga"       : "GA (NSGA-II)",
    "aco"      : "ACO",
}


class ConvergencePlotter:
    def __init__(self, output_dir: str = "outputs"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    # ── 1. Hypervolume Convergence ────────────────────────────────────────────

    def plot_hypervolume(
        self,
        histories    : Dict[str, Dict],   # {"cuda_gwo": history_dict, ...}
        filename     : str = "hypervolume_convergence.png",
        title        : str = "Hypervolume Convergence",
    ) -> str:
        if not _MPL: return ""

        fig, ax = plt.subplots(figsize=(9, 5))

        for algo, hist in histories.items():
            hv   = hist.get("hypervolume", [])
            iters = range(len(hv))
            ax.plot(iters, hv,
                    color=COLORS.get(algo, "#333333"),
                    linewidth=2,
                    label=LABELS.get(algo, algo))

        ax.set_xlabel("Iteration", fontsize=12)
        ax.set_ylabel("Hypervolume Indicator", fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.legend(fontsize=10, loc="lower right")
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.4f"))
        plt.tight_layout()

        path = os.path.join(self.output_dir, filename)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Hypervolume plot saved: %s", path)
        return path

    # ── 2. Objective Convergence (4 objectives) ───────────────────────────────

    def plot_objective_convergence(
        self,
        histories  : Dict[str, Dict],
        filename   : str = "objective_convergence.png",
    ) -> str:
        if not _MPL: return ""

        obj_keys = ["best_f1", "best_f2", "best_f3", "best_f4"]
        obj_labels = [
            "f₁: Avg Travel Time (min)",
            "f₂: Avg Transfers",
            "f₃: Fleet Cost (₹ thousands/day)",
            "f₄: 1 − Coverage",
        ]

        fig, axes = plt.subplots(2, 2, figsize=(13, 8))
        axes = axes.flatten()

        for i, (key, label) in enumerate(zip(obj_keys, obj_labels)):
            ax = axes[i]
            for algo, hist in histories.items():
                vals = hist.get(key, [])
                if vals:
                    ax.plot(range(len(vals)), vals,
                            color=COLORS.get(algo, "#333333"),
                            linewidth=1.8,
                            label=LABELS.get(algo, algo))
            ax.set_title(label, fontsize=11)
            ax.set_xlabel("Iteration", fontsize=9)
            ax.grid(True, alpha=0.3, linestyle="--")
            if i == 0:
                ax.legend(fontsize=8)

        plt.suptitle("Objective Function Convergence", fontsize=13,
                     fontweight="bold", y=1.01)
        plt.tight_layout()

        path = os.path.join(self.output_dir, filename)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Objective convergence plot saved: %s", path)
        return path

    # ── 3. Speedup Bar Chart ──────────────────────────────────────────────────

    def plot_speedup(
        self,
        timings  : Dict[str, float],   # {"cuda_gwo": 12.3, "cpu_gwo": 345.6, ...}
        filename : str = "speedup_comparison.png",
    ) -> str:
        if not _MPL: return ""

        cuda_time = timings.get("cuda_gwo", 1.0)
        algos     = [a for a in timings if a != "cuda_gwo"]
        speedups  = [timings[a] / cuda_time for a in algos]

        if not speedups:
            logger.info("Speedup plot skipped: only one algorithm ran.")
            return ""
        colors    = [COLORS.get(a, "#999999") for a in algos]
        labels    = [LABELS.get(a, a) for a in algos]

        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(labels, speedups, color=colors, edgecolor="black",
                      linewidth=0.8, zorder=3)

        for bar, sp in zip(bars, speedups):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    f"{sp:.1f}×", ha="center", va="bottom",
                    fontsize=11, fontweight="bold")

        ax.axhline(y=1, color="red", linestyle="--", alpha=0.4, linewidth=1.2,
                   label="Baseline (CUDA-GWO)")
        ax.set_ylabel("Speedup over CUDA-GWO", fontsize=12)
        ax.set_title("Execution Time Speedup (CUDA-GWO is baseline)", fontsize=12,
                     fontweight="bold")
        ax.set_ylim(0, max(speedups) * 1.25)
        ax.grid(axis="y", alpha=0.3, linestyle="--", zorder=0)
        ax.legend(fontsize=9)
        plt.tight_layout()

        path = os.path.join(self.output_dir, filename)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Speedup chart saved: %s", path)
        return path

    # ── 4. Pareto Front Scatter ───────────────────────────────────────────────

    def plot_pareto_front(
        self,
        pareto_objectives : Dict[str, np.ndarray],   # {algo: (N, 4) array}
        axes_pair         : tuple = (0, 1),
        filename          : str   = "pareto_front.png",
    ) -> str:
        if not _MPL: return ""

        ax_labels = [
            "f₁: Avg Travel Time (min)",
            "f₂: Avg Transfers",
            "f₃: Fleet Cost",
            "f₄: 1 − Coverage",
        ]
        xi, yi = axes_pair

        fig, ax = plt.subplots(figsize=(8, 6))

        for algo, obj in pareto_objectives.items():
            if len(obj) == 0:
                continue
            ax.scatter(obj[:, xi], obj[:, yi],
                       c=COLORS.get(algo, "#333333"),
                       s=30, alpha=0.75, edgecolors="none",
                       label=LABELS.get(algo, algo))

        ax.set_xlabel(ax_labels[xi], fontsize=11)
        ax.set_ylabel(ax_labels[yi], fontsize=11)
        ax.set_title("Pareto Front Comparison", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, linestyle="--")
        plt.tight_layout()

        path = os.path.join(self.output_dir, filename)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Pareto front plot saved: %s", path)
        return path

    # ── 5. Summary Dashboard (all-in-one) ────────────────────────────────────

    def plot_dashboard(
        self,
        histories         : Dict[str, Dict],
        timings           : Dict[str, float],
        pareto_objectives : Dict[str, np.ndarray],
        filename          : str = "dashboard.png",
    ) -> str:
        if not _MPL: return ""

        fig = plt.figure(figsize=(18, 12))
        gs  = GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

        # 1. Hypervolume (top-left)
        ax1 = fig.add_subplot(gs[0, 0])
        for algo, hist in histories.items():
            hv = hist.get("hypervolume", [])
            ax1.plot(range(len(hv)), hv, color=COLORS.get(algo, "#333"),
                     linewidth=1.8, label=LABELS.get(algo, algo))
        ax1.set_title("Hypervolume Convergence", fontweight="bold")
        ax1.set_xlabel("Iteration"); ax1.set_ylabel("HV")
        ax1.legend(fontsize=7); ax1.grid(True, alpha=0.3, linestyle="--")

        # 2. f1 convergence (top-center)
        ax2 = fig.add_subplot(gs[0, 1])
        for algo, hist in histories.items():
            vals = hist.get("best_f1", [])
            ax2.plot(range(len(vals)), vals, color=COLORS.get(algo, "#333"),
                     linewidth=1.8, label=LABELS.get(algo, algo))
        ax2.set_title("f₁ Convergence (Travel Time)", fontweight="bold")
        ax2.set_xlabel("Iteration"); ax2.set_ylabel("min")
        ax2.legend(fontsize=7); ax2.grid(True, alpha=0.3, linestyle="--")

        # 3. Speedup bar (top-right)
        ax3 = fig.add_subplot(gs[0, 2])
        cuda_t = timings.get("cuda_gwo", 1.0)
        algos  = [a for a in timings if a != "cuda_gwo"]
        sups   = [timings[a] / cuda_t for a in algos]
        cols   = [COLORS.get(a, "#999") for a in algos]
        if algos:
            ax3.bar([LABELS.get(a, a) for a in algos], sups, color=cols,
                    edgecolor="black", linewidth=0.7)
        else:
            ax3.text(0.5, 0.5, "Run with --algo all\nto see speedup comparison",
                     ha="center", va="center", transform=ax3.transAxes,
                     fontsize=10, color="#666666")
        ax3.set_title("Execution Speedup over CUDA-GWO", fontweight="bold")
        ax3.set_ylabel("Speedup x"); ax3.grid(axis="y", alpha=0.3, linestyle="--")

        # 4. Pareto f1 vs f2 (bottom-left)
        ax4 = fig.add_subplot(gs[1, 0])
        for algo, obj in pareto_objectives.items():
            if len(obj) > 0:
                ax4.scatter(obj[:, 0], obj[:, 1], c=COLORS.get(algo, "#333"),
                            s=25, alpha=0.7, label=LABELS.get(algo, algo))
        ax4.set_title("Pareto Front: f₁ vs f₂", fontweight="bold")
        ax4.set_xlabel("f₁: Travel Time"); ax4.set_ylabel("f₂: Transfers")
        ax4.legend(fontsize=7); ax4.grid(True, alpha=0.3, linestyle="--")

        # 5. Pareto f1 vs f3 (bottom-center)
        ax5 = fig.add_subplot(gs[1, 1])
        for algo, obj in pareto_objectives.items():
            if len(obj) > 0:
                ax5.scatter(obj[:, 0], obj[:, 2], c=COLORS.get(algo, "#333"),
                            s=25, alpha=0.7, label=LABELS.get(algo, algo))
        ax5.set_title("Pareto Front: f₁ vs f₃", fontweight="bold")
        ax5.set_xlabel("f₁: Travel Time"); ax5.set_ylabel("f₃: Cost")
        ax5.legend(fontsize=7); ax5.grid(True, alpha=0.3, linestyle="--")

        # 6. Iteration time comparison (bottom-right)
        ax6 = fig.add_subplot(gs[1, 2])
        for algo, hist in histories.items():
            it = hist.get("iter_time_ms", hist.get("iteration_time", []))
            if it:
                it_ms = np.array(it) * (1000 if max(it) < 10 else 1)
                ax6.plot(range(len(it_ms)), it_ms,
                         color=COLORS.get(algo, "#333"),
                         linewidth=1.5, alpha=0.8,
                         label=LABELS.get(algo, algo))
        ax6.set_title("Iteration Time (ms)", fontweight="bold")
        ax6.set_xlabel("Iteration"); ax6.set_ylabel("ms")
        ax6.legend(fontsize=7); ax6.grid(True, alpha=0.3, linestyle="--")

        plt.suptitle("CUDA-GWO Transit Route Optimization — Results Dashboard",
                     fontsize=16, fontweight="bold", y=1.01)

        path = os.path.join(self.output_dir, filename)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Dashboard saved: %s", path)
        return path

    # ── 6. Benchmark Comparison (all algos vs CUDA-GWO) ─────────────────────

    def plot_benchmark_comparison(
        self,
        run_results : list,          # list of result dicts from main.py
        filename    : str = "benchmark_comparison.png",
    ) -> str:
        """
        Comprehensive benchmark comparison image:
          - Convergence curves (f1 over iterations)
          - Hypervolume convergence
          - Final objective values grouped bar chart (f1..f4 normalised)
          - Runtime bar chart
          - Solution quality table (text)
        """
        if not _MPL:
            return ""

        fig = plt.figure(figsize=(20, 14))
        gs  = GridSpec(3, 3, figure=fig, hspace=0.48, wspace=0.35)

        algo_names  = [r["algo"]        for r in run_results]
        runtimes    = [r["runtime_s"]   for r in run_results]
        objectives  = [r["best_obj"]    for r in run_results]   # each (4,)
        histories   = {r["algo"]: r["history"] for r in run_results}

        bar_colors  = [COLORS.get(a, "#999999") for a in algo_names]
        tick_labels = [LABELS.get(a, a)         for a in algo_names]
        x           = np.arange(len(algo_names))

        # ── Panel 1: f1 convergence ───────────────────────────────────────────
        ax1 = fig.add_subplot(gs[0, 0])
        for r in run_results:
            vals = r["history"].get("best_f1", [])
            ax1.plot(range(len(vals)), vals,
                     color=COLORS.get(r["algo"], "#333"),
                     linewidth=2, label=LABELS.get(r["algo"], r["algo"]))
        ax1.set_title("f₁  Travel Time Convergence", fontweight="bold", fontsize=11)
        ax1.set_xlabel("Iteration"); ax1.set_ylabel("Avg Travel Time (min)")
        ax1.legend(fontsize=7.5); ax1.grid(True, alpha=0.3, linestyle="--")

        # ── Panel 2: Hypervolume convergence ──────────────────────────────────
        ax2 = fig.add_subplot(gs[0, 1])
        for r in run_results:
            hv = r["history"].get("hypervolume", [])
            ax2.plot(range(len(hv)), hv,
                     color=COLORS.get(r["algo"], "#333"),
                     linewidth=2, label=LABELS.get(r["algo"], r["algo"]))
        ax2.set_title("Hypervolume Convergence", fontweight="bold", fontsize=11)
        ax2.set_xlabel("Iteration"); ax2.set_ylabel("HV Indicator")
        ax2.legend(fontsize=7.5); ax2.grid(True, alpha=0.3, linestyle="--")

        # ── Panel 3: Runtime bar ──────────────────────────────────────────────
        ax3 = fig.add_subplot(gs[0, 2])
        bars = ax3.bar(x, runtimes, color=bar_colors, edgecolor="black",
                       linewidth=0.8, zorder=3)
        for bar, t in zip(bars, runtimes):
            ax3.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.3,
                     f"{t:.1f}s", ha="center", va="bottom",
                     fontsize=9, fontweight="bold")
        ax3.set_xticks(x); ax3.set_xticklabels(tick_labels, rotation=20, ha="right", fontsize=9)
        ax3.set_title("Runtime Comparison", fontweight="bold", fontsize=11)
        ax3.set_ylabel("Seconds"); ax3.grid(axis="y", alpha=0.3, linestyle="--", zorder=0)

        # ── Panels 4-7: Final value bars for f1..f4 ───────────────────────────
        obj_labels_short = [
            "f₁  Travel Time (min)",
            "f₂  Avg Transfers",
            "f₃  Fleet Cost (kRs)",
            "f₄  1 − Coverage",
        ]
        positions = [(1, 0), (1, 1), (2, 0), (2, 1)]
        for fi, (row_i, col_i) in enumerate(positions):
            ax = fig.add_subplot(gs[row_i, col_i])
            vals = [float(obj[fi]) for obj in objectives]
            bars = ax.bar(x, vals, color=bar_colors, edgecolor="black",
                          linewidth=0.8, zorder=3)
            # Highlight the best (lowest) bar
            best_idx = int(np.argmin(vals))
            bars[best_idx].set_edgecolor("gold")
            bars[best_idx].set_linewidth(2.5)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() * 1.01,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=8)
            ax.set_xticks(x)
            ax.set_xticklabels(tick_labels, rotation=20, ha="right", fontsize=8)
            ax.set_title(obj_labels_short[fi], fontweight="bold", fontsize=10)
            ax.grid(axis="y", alpha=0.3, linestyle="--", zorder=0)
            ax.set_ylabel("Value")

        # ── Panel: Summary text table (bottom-right) ──────────────────────────
        ax_t = fig.add_subplot(gs[2, 2])
        ax_t.axis("off")
        col_headers = ["Algorithm", "f1", "f2", "f3", "f4", "Time(s)"]
        table_data  = [
            [LABELS.get(r["algo"], r["algo"]),
             f"{r['best_obj'][0]:.2f}",
             f"{r['best_obj'][1]:.3f}",
             f"{r['best_obj'][2]:.1f}",
             f"{r['best_obj'][3]:.3f}",
             f"{r['runtime_s']:.1f}"]
            for r in run_results
        ]
        tbl = ax_t.table(
            cellText    = table_data,
            colLabels   = col_headers,
            cellLoc     = "center",
            loc         = "center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8.5)
        tbl.scale(1, 1.6)
        # Header row styling
        for j in range(len(col_headers)):
            tbl[(0, j)].set_facecolor("#222222")
            tbl[(0, j)].set_text_props(color="white", fontweight="bold")
        # Highlight CUDA-GWO row
        for i, r in enumerate(run_results):
            if r["algo"] == "cuda_gwo":
                for j in range(len(col_headers)):
                    tbl[(i + 1, j)].set_facecolor("#D0E8FF")
        ax_t.set_title("Final Results Summary", fontweight="bold", fontsize=10, pad=10)

        plt.suptitle(
            "Benchmark Comparison — CUDA-GWO vs Baselines\n"
            f"(50 stops · 10 routes · 50 wolves · 50 iterations)",
            fontsize=15, fontweight="bold", y=1.01,
        )

        path = os.path.join(self.output_dir, filename)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Benchmark comparison saved: %s", path)
        return path

    # ── 7. Scalability Plot ───────────────────────────────────────────────────

    def plot_scalability(
        self,
        problem_sizes  : List[int],           # e.g. [100, 200, 500, 1000]
        cpu_times      : List[float],         # seconds per problem size
        cuda_times     : List[float],
        filename       : str = "scalability.png",
    ) -> str:
        if not _MPL: return ""

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Left: absolute time
        ax = axes[0]
        ax.plot(problem_sizes, cpu_times,  "o-", color=COLORS["cpu_gwo"],
                linewidth=2, label="CPU-GWO")
        ax.plot(problem_sizes, cuda_times, "s-", color=COLORS["cuda_gwo"],
                linewidth=2, label="CUDA-GWO")
        ax.set_xlabel("Number of Stops", fontsize=11)
        ax.set_ylabel("Time (seconds)", fontsize=11)
        ax.set_title("Scalability: Absolute Time", fontweight="bold")
        ax.legend(); ax.grid(True, alpha=0.3, linestyle="--")
        ax.set_yscale("log")

        # Right: speedup
        ax2 = axes[1]
        speedups = [c / g for c, g in zip(cpu_times, cuda_times)]
        ax2.plot(problem_sizes, speedups, "D-", color="#AA0000",
                 linewidth=2, markersize=7)
        ax2.fill_between(problem_sizes, speedups, alpha=0.15, color="#AA0000")
        ax2.set_xlabel("Number of Stops", fontsize=11)
        ax2.set_ylabel("Speedup (CPU / CUDA)", fontsize=11)
        ax2.set_title("CUDA Speedup Factor", fontweight="bold")
        ax2.grid(True, alpha=0.3, linestyle="--")

        plt.suptitle("Scalability Analysis", fontsize=13, fontweight="bold")
        plt.tight_layout()

        path = os.path.join(self.output_dir, filename)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Scalability plot saved: %s", path)
        return path
