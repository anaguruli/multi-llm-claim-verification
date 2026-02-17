"""
Generates all required plots and saves them to results/plots/.
"""

from __future__ import annotations

import os
from typing import List, Dict, Any, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from matplotlib.gridspec import GridSpec

import config
from evaluation.metrics import EvaluationReport

# ── Style ─────────────────────────────────────────────────────────────────────
PALETTE = {
    "system": "#2E86AB",
    "baseline": "#E84855",
    "TRUE": "#4CAF50",
    "FALSE": "#F44336",
    "PARTIALLY_TRUE": "#FF9800",
    "MISLEADING": "#9C27B0",
}
plt.rcParams.update({
    "figure.dpi": 150,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ── Plot 1: Accuracy Comparison ───────────────────────────────────────────────

def plot_accuracy_comparison(report: EvaluationReport, save_dir: str = config.PLOTS_DIR) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Accuracy Comparison: Full System vs Single-LLM Baseline", fontsize=14, fontweight="bold")

    # Left: Overall bar chart
    ax = axes[0]
    labels = ["Full System\n(RAG + Multi-Agent)", "Single-LLM\nBaseline"]
    values = [report.system_accuracy * 100, report.baseline_accuracy * 100]
    colors = [PALETTE["system"], PALETTE["baseline"]]
    bars = ax.bar(labels, values, color=colors, width=0.45, edgecolor="white", linewidth=1.5)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{val:.1f}%",
            ha="center", va="bottom", fontweight="bold", fontsize=13,
        )
    ax.set_ylim(0, 110)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_title("Overall Accuracy", fontsize=12)
    ax.axhline(y=50, color="gray", linestyle="--", alpha=0.4, linewidth=1)
    ax.text(1.4, 51, "50% baseline", color="gray", fontsize=9)

    # Right: Per-category grouped bars
    ax2 = axes[1]
    cat_stats = report.per_category_accuracy
    categories = list(cat_stats.keys())
    cat_labels = [c.replace("_", "\n") for c in categories]
    x = np.arange(len(categories))
    w = 0.35
    sys_vals = [cat_stats[c]["system_accuracy"] * 100 for c in categories]
    base_vals = [cat_stats[c]["baseline_accuracy"] * 100 for c in categories]

    bars1 = ax2.bar(x - w/2, sys_vals, w, label="Full System", color=PALETTE["system"],
                    edgecolor="white", linewidth=1.2)
    bars2 = ax2.bar(x + w/2, base_vals, w, label="Baseline", color=PALETTE["baseline"],
                    edgecolor="white", linewidth=1.2)

    for bar, val in zip(bars1, sys_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f"{val:.0f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")
    for bar, val in zip(bars2, base_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f"{val:.0f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax2.set_xticks(x)
    ax2.set_xticklabels(cat_labels, fontsize=9)
    ax2.set_ylim(0, 120)
    ax2.set_ylabel("Accuracy (%)", fontsize=11)
    ax2.set_title("Accuracy by Category", fontsize=12)
    ax2.legend(fontsize=9)

    plt.tight_layout()
    path = os.path.join(save_dir, "accuracy_comparison.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved: {path}")
    return path


# ── Plot 2: Confusion Matrices ────────────────────────────────────────────────

def plot_confusion_matrices(report: EvaluationReport, save_dir: str = config.PLOTS_DIR) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Confusion Matrices: Predicted vs Ground Truth", fontsize=14, fontweight="bold")

    sys_matrix, labels = report.confusion_matrix_system
    base_matrix, _ = report.confusion_matrix_baseline

    short_labels = ["TRUE", "FALSE", "PART.\nTRUE", "MISLEAD."]

    for ax, matrix, title in zip(
        axes,
        [sys_matrix, base_matrix],
        ["Full System (RAG + Multi-Agent)", "Single-LLM Baseline"],
    ):
        # Normalise rows for colour, show raw counts as annotations
        row_sums = matrix.sum(axis=1, keepdims=True)
        norm_matrix = np.divide(matrix, row_sums, where=row_sums != 0, out=np.zeros_like(matrix, dtype=float))

        im = ax.imshow(norm_matrix, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(short_labels, fontsize=9)
        ax.set_yticklabels(short_labels, fontsize=9)
        ax.set_xlabel("Predicted Verdict", fontsize=10)
        ax.set_ylabel("True Verdict", fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")

        for i in range(len(labels)):
            for j in range(len(labels)):
                count = matrix[i, j]
                colour = "white" if norm_matrix[i, j] > 0.5 else "black"
                ax.text(j, i, str(count), ha="center", va="center",
                        fontsize=13, fontweight="bold", color=colour)

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Row-normalised rate")

    plt.tight_layout()
    path = os.path.join(save_dir, "confusion_matrices.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved: {path}")
    return path


# ── Plot 3: Deliberation Statistics ──────────────────────────────────────────

def plot_deliberation_stats(report: EvaluationReport, save_dir: str = config.PLOTS_DIR) -> str:
    fig = plt.figure(figsize=(14, 5))
    fig.suptitle("Deliberation Analysis", fontsize=14, fontweight="bold")
    gs = GridSpec(1, 3, figure=fig, wspace=0.4)

    records = report.records

    # ── Left: Disagreement rate pie ──────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    n_disagree = sum(r.had_any_disagreement for r in records)
    n_agree = len(records) - n_disagree
    wedge_colours = [PALETTE["system"], "#CCCCCC"]
    wedges, texts, autotexts = ax1.pie(
        [n_disagree, n_agree],
        labels=[f"Disagreement\n({n_disagree})", f"Full Agreement\n({n_agree})"],
        colors=wedge_colours,
        autopct="%1.0f%%",
        startangle=90,
        textprops={"fontsize": 10},
    )
    for at in autotexts:
        at.set_fontweight("bold")
    ax1.set_title("Verifier Agreement\nAcross Claims", fontsize=11)

    # ── Middle: Mind-change breakdown ────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    n_changed = sum(r.deliberation_changed for r in records)
    n_outcome = sum(r.deliberation_changed_outcome for r in records)
    n_no_change = len(records) - n_changed

    categories_d = ["No Change", "Mind Changed\n(Verdict Same)", "Mind Changed\n(Outcome Changed)"]
    counts_d = [n_no_change, n_changed - n_outcome, n_outcome]
    colours_d = ["#CCCCCC", "#FFC107", PALETTE["system"]]

    bars = ax2.bar(categories_d, counts_d, color=colours_d, edgecolor="white", linewidth=1.2)
    for bar, cnt in zip(bars, counts_d):
        if cnt > 0:
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                     str(cnt), ha="center", va="bottom", fontweight="bold", fontsize=12)
    ax2.set_ylabel("Number of Claims", fontsize=10)
    ax2.set_title("Deliberation\nMind-Change Impact", fontsize=11)
    ax2.set_ylim(0, max(counts_d) + 2)
    plt.setp(ax2.get_xticklabels(), fontsize=8)

    # ── Right: Correctness improvement from deliberation ─────────────────────
    ax3 = fig.add_subplot(gs[2])
    # Claims that changed outcome — were they improvements or regressions?
    outcome_changed = [r for r in records if r.deliberation_changed_outcome]
    if outcome_changed:
        improved = sum(r.system_correct for r in outcome_changed)
        regressed = len(outcome_changed) - improved
        ax3.bar(
            ["Improved\nby Deliberation", "Worsened\nby Deliberation"],
            [improved, regressed],
            color=[PALETTE["TRUE"], PALETTE["FALSE"]],
            edgecolor="white",
            linewidth=1.2,
        )
        for i, cnt in enumerate([improved, regressed]):
            ax3.text(i, cnt + 0.05, str(cnt), ha="center", va="bottom",
                     fontweight="bold", fontsize=13)
        ax3.set_ylabel("Number of Claims", fontsize=10)
        ax3.set_ylim(0, max(improved, regressed) + 2)
    else:
        ax3.text(0.5, 0.5, "No outcome\nchanges", ha="center", va="center",
                 transform=ax3.transAxes, fontsize=12, color="gray")
        ax3.set_xlim(0, 1)
        ax3.set_ylim(0, 1)
    ax3.set_title("Outcome of\nDeliberation Changes", fontsize=11)

    plt.tight_layout()
    path = os.path.join(save_dir, "deliberation_stats.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved: {path}")
    return path


# ── Plot 4: Per-claim verdict heatmap ─────────────────────────────────────────

def plot_verdict_heatmap(
    records_data: List[Dict[str, Any]],
    save_dir: str = config.PLOTS_DIR,
) -> str:
    """
    Heatmap: rows = claims, columns = [Ground Truth, System, Baseline].
    Cells are colour-coded by verdict type.
    """
    verdict_to_int = {"TRUE": 0, "FALSE": 1, "PARTIALLY_TRUE": 2, "MISLEADING": 3}
    cmap = matplotlib.colors.ListedColormap([
        PALETTE["TRUE"], PALETTE["FALSE"], PALETTE["PARTIALLY_TRUE"], PALETTE["MISLEADING"]
    ])
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    n = len(records_data)
    matrix = np.zeros((n, 3), dtype=int)
    y_labels = []

    for i, r in enumerate(records_data):
        matrix[i, 0] = verdict_to_int.get(r["ground_truth"], 1)
        matrix[i, 1] = verdict_to_int.get(r["system_verdict"], 1)
        matrix[i, 2] = verdict_to_int.get(r["baseline_verdict"], 1)
        y_labels.append(r["claim_id"])

    fig, ax = plt.subplots(figsize=(7, max(6, n * 0.55)))
    im = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Ground\nTruth", "Full\nSystem", "Baseline"], fontsize=11, fontweight="bold")
    ax.set_yticks(range(n))
    ax.set_yticklabels(y_labels, fontsize=9)
    ax.set_title("Per-Claim Verdict Comparison", fontsize=13, fontweight="bold", pad=12)

    # Annotate cells with verdict text
    short = {"TRUE": "T", "FALSE": "F", "PARTIALLY_TRUE": "PT", "MISLEADING": "M"}
    col_keys = ["ground_truth", "system_verdict", "baseline_verdict"]
    for i, r in enumerate(records_data):
        for j, key in enumerate(col_keys):
            v = r[key]
            txt_colour = "white" if v in ("FALSE", "MISLEADING") else "black"
            ax.text(j, i, short.get(v, v), ha="center", va="center",
                    fontsize=9, fontweight="bold", color=txt_colour)

    # Legend
    patches = [
        mpatches.Patch(color=PALETTE["TRUE"], label="TRUE"),
        mpatches.Patch(color=PALETTE["FALSE"], label="FALSE"),
        mpatches.Patch(color=PALETTE["PARTIALLY_TRUE"], label="PARTIALLY TRUE"),
        mpatches.Patch(color=PALETTE["MISLEADING"], label="MISLEADING"),
    ]
    ax.legend(handles=patches, loc="upper right", bbox_to_anchor=(1.55, 1.0),
              fontsize=9, title="Verdict", title_fontsize=9)

    plt.tight_layout()
    path = os.path.join(save_dir, "verdict_heatmap.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved: {path}")
    return path


# ── Plot 5: Confidence Distribution ──────────────────────────────────────────

def plot_confidence_distribution(
    records: List[Dict[str, Any]],
    save_dir: str = config.PLOTS_DIR,
) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("System Confidence Analysis", fontsize=13, fontweight="bold")

    correct_conf = [r["system_confidence"] for r in records if r["system_correct"]]
    incorrect_conf = [r["system_confidence"] for r in records if not r["system_correct"]]

    # Left: histogram of confidence for correct vs incorrect
    ax = axes[0]
    bins = np.linspace(0, 1, 11)
    if correct_conf:
        ax.hist(correct_conf, bins=bins, alpha=0.7, color=PALETTE["system"],
                label=f"Correct (n={len(correct_conf)})", edgecolor="white")
    if incorrect_conf:
        ax.hist(incorrect_conf, bins=bins, alpha=0.7, color=PALETTE["baseline"],
                label=f"Incorrect (n={len(incorrect_conf)})", edgecolor="white")
    ax.set_xlabel("Confidence Score", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title("Confidence Distribution\n(Correct vs Incorrect)", fontsize=11)
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1)

    # Right: mean confidence per verdict category
    ax2 = axes[1]
    from collections import defaultdict
    verdict_confs: Dict[str, List[float]] = defaultdict(list)
    for r in records:
        verdict_confs[r["ground_truth"]].append(r["system_confidence"])

    v_labels = [v for v in ["TRUE", "FALSE", "PARTIALLY_TRUE", "MISLEADING"] if v in verdict_confs]
    means = [np.mean(verdict_confs[v]) for v in v_labels]
    stds = [np.std(verdict_confs[v]) if len(verdict_confs[v]) > 1 else 0 for v in v_labels]
    colours = [PALETTE[v] for v in v_labels]
    short_v = [v.replace("_", "\n") for v in v_labels]

    bars = ax2.bar(short_v, means, color=colours, edgecolor="white",
                   linewidth=1.2, yerr=stds, capsize=5, error_kw={"linewidth": 1.5})
    for bar, m in zip(bars, means):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f"{m:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax2.set_ylim(0, 1.15)
    ax2.set_ylabel("Mean Confidence", fontsize=10)
    ax2.set_title("Mean System Confidence\nby Ground-Truth Verdict", fontsize=11)

    plt.tight_layout()
    path = os.path.join(save_dir, "confidence_distribution.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved: {path}")
    return path


# ── Master entry point ────────────────────────────────────────────────────────

def generate_all_plots(
    report: EvaluationReport,
    records_data: List[Dict[str, Any]],
    save_dir: str = config.PLOTS_DIR,
) -> List[str]:
    os.makedirs(save_dir, exist_ok=True)
    paths = [
        plot_accuracy_comparison(report, save_dir),
        plot_confusion_matrices(report, save_dir),
        plot_deliberation_stats(report, save_dir),
        plot_verdict_heatmap(records_data, save_dir),
        plot_confidence_distribution(records_data, save_dir),
    ]
    print(f"\n[Visualizer] All {len(paths)} plots saved to '{save_dir}'")
    return paths