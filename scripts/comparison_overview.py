"""Build a one-page overview graphic comparing Patchcore v1 and v2 from results/comparison.json.

Usage:
    uv run python scripts/comparison_overview.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
IN_JSON = ROOT / "results/comparison.json"
OUT_PNG = ROOT / "results/version_comparison.png"

VERSIONS = [("v1_baseline", "v1\n256px, k=6"), ("v2_384px_k9", "v2\n384px, k=9")]
METRICS = [
    ("image_AUROC", "Image AUROC"),
    ("image_F1Score", "Image F1"),
    ("pixel_AUROC", "Pixel AUROC"),
    ("pixel_F1Score", "Pixel F1"),
]
COLORS = ["#2a78d6", "#eb6834"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"


def style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK2, labelsize=9)


def main() -> None:
    rows = json.loads(IN_JSON.read_text())
    vals = {n: {k: np.array([r[k] for r in rows if r["name"] == n])
                for k in [m for m, _ in METRICS] + ["fit_s", "test_s"]} for n, _ in VERSIONS}
    seeds = sorted({r["seed"] for r in rows})

    fig = plt.figure(figsize=(14, 7.2), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 4, height_ratios=[1.15, 1], hspace=0.55, wspace=0.42)

    # top row: one dot plot per metric (own y range, so differences are visible but dots show spread)
    for col, (key, title) in enumerate(METRICS):
        ax = fig.add_subplot(gs[0, col])
        style(ax)
        means = []
        for i, (name, _) in enumerate(VERSIONS):
            v = vals[name][key]
            jitter = np.linspace(-0.08, 0.08, len(v))
            ax.scatter(np.full(len(v), i) + jitter, v, color=COLORS[i], s=38, zorder=3,
                       edgecolor=SURFACE, linewidth=1)
            ax.hlines(v.mean(), i - 0.25, i + 0.25, color=COLORS[i], linewidth=2.5, zorder=2)
            ax.annotate(f"{v.mean():.3f}", (i, v.max()), xytext=(0, 9), textcoords="offset points",
                        color=INK, fontsize=10, ha="center", fontweight="bold")
            means.append(v.mean())
        allv = np.concatenate([vals[n][key] for n, _ in VERSIONS])
        pad = max((allv.max() - allv.min()) * 0.6, 0.01)
        top = allv.max() + pad * 1.4
        ax.set_ylim(allv.min() - pad, min(top, 1.0) if "AUROC" in key else top)
        ax.set_xlim(-0.6, 1.6)
        ax.set_xticks([0, 1], [lbl for _, lbl in VERSIONS], color=INK2)
        delta = means[1] - means[0]
        ax.set_title(title, loc="left", color=INK, fontsize=11)
        ax.annotate(f"{delta:+.3f} vs v1", (0.98, 0.04), xycoords="axes fraction", ha="right",
                    fontsize=9, color=("#b3321f" if delta < -0.005 else INK2))

    # bottom left: training time (zero-based bars, minutes)
    ax = fig.add_subplot(gs[1, 0:2])
    style(ax)
    fit_min = [vals[n]["fit_s"].mean() / 60 for n, _ in VERSIONS]
    bars = ax.barh([1, 0], fit_min[::-1], color=COLORS[::-1], height=0.55)
    for y, m in zip([1, 0], fit_min[::-1]):
        ax.text(m + 1, y, f"{m:.1f} min", va="center", color=INK, fontsize=10)
    ax.set_yticks([1, 0], [lbl.replace("\n", " - ") for _, lbl in VERSIONS][::-1], color=INK2)
    ax.set_xlim(0, max(fit_min) * 1.18)
    ax.set_xlabel("Training time per run (minutes)", color=INK2)
    ax.set_title("Cost: v2 trains ~%dx slower" % round(fit_min[1] / fit_min[0]), loc="left",
                 color=INK, fontsize=11)
    ax.yaxis.grid(False)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)

    # bottom right: verdict text
    ax = fig.add_subplot(gs[1, 2:])
    ax.axis("off")
    d = {k: vals[VERSIONS[1][0]][k].mean() - vals[VERSIONS[0][0]][k].mean() for k, _ in METRICS}
    lines = [
        ("Verdict", True),
        (f"Image level: essentially unchanged (AUROC {d['image_AUROC']:+.4f}; F1 {d['image_F1Score']:+.3f},", False),
        ("  roughly one image of 100 changes class).", False),
        (f"Pixel level: clearly worse (AUROC {d['pixel_AUROC']:+.3f}, F1 {d['pixel_F1Score']:+.3f}).", False),
        (f"Cost: ~{fit_min[1] / fit_min[0]:.0f}x training time and GPU memory near the 8 GB limit.", False),
        ("", False),
        ("Conclusion: larger images + more neighbours is NOT an improvement.", True),
        ("Keep v1 as the baseline.", False),
        ("", False),
        (f"Mean over seeds {seeds}. Seed spread is tiny (see dots), so differences are real.", False),
    ]
    y = 0.98
    for text, bold in lines:
        ax.text(0.0, y, text, transform=ax.transAxes, va="top", fontsize=10.5 if not bold else 11.5,
                color=INK if bold else INK2, fontweight="bold" if bold else "normal")
        y -= 0.105

    fig.suptitle("Patchcore v1 vs v2 on MVTec AD transistor (100-image test set)", x=0.01, y=0.99,
                 ha="left", color=INK, fontsize=14, fontweight="bold")
    fig.savefig(OUT_PNG, dpi=130, bbox_inches="tight", facecolor=SURFACE)
    print(f"Saved {OUT_PNG}")


if __name__ == "__main__":
    main()
