"""Build a one-page overview graphic comparing all Patchcore versions in results/comparison.json.

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

# (name in JSON, short tag, axis label)
VERSIONS = [
    ("v1_baseline", "v1", "v1\n256px, k=6"),
    ("v2_384px_k9", "v2", "v2\n384px, k=9"),
    ("v3_aug2x", "v3", "v3\n256px, k=6\n2x data"),
]
METRICS = [
    ("image_AUROC", "Image AUROC"),
    ("image_F1Score", "Image F1"),
    ("pixel_AUROC", "Pixel AUROC"),
    ("pixel_F1Score", "Pixel F1"),
]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]  # reference palette slots 1-3
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
WORSE = "#b3321f"


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
    keys = [m for m, _ in METRICS] + ["fit_s", "test_s"]
    vals = {n: {k: np.array([r[k] for r in rows if r["name"] == n]) for k in keys}
            for n, _, _ in VERSIONS}
    seeds = sorted({r["seed"] for r in rows})
    base = VERSIONS[0][0]

    def delta(name: str, key: str) -> float:
        return vals[name][key].mean() - vals[base][key].mean()

    fig = plt.figure(figsize=(15, 7.6), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 4, height_ratios=[1.15, 1], hspace=0.62, wspace=0.42)

    # top row: one dot plot per metric
    for col, (key, title) in enumerate(METRICS):
        ax = fig.add_subplot(gs[0, col])
        style(ax)
        for i, (name, _, _) in enumerate(VERSIONS):
            v = vals[name][key]
            jitter = np.linspace(-0.08, 0.08, len(v))
            ax.scatter(np.full(len(v), i) + jitter, v, color=COLORS[i], s=34, zorder=3,
                       edgecolor=SURFACE, linewidth=1)
            ax.hlines(v.mean(), i - 0.27, i + 0.27, color=COLORS[i], linewidth=2.5, zorder=2)
            ax.annotate(f"{v.mean():.3f}", (i, v.max()), xytext=(0, 9), textcoords="offset points",
                        color=INK, fontsize=9.5, ha="center", fontweight="bold")
        allv = np.concatenate([vals[n][key] for n, _, _ in VERSIONS])
        pad = max((allv.max() - allv.min()) * 0.6, 0.01)
        top = allv.max() + pad * 1.4
        ax.set_ylim(allv.min() - pad, min(top, 1.0) if "AUROC" in key else top)
        ax.set_xlim(-0.6, len(VERSIONS) - 0.4)
        ax.set_xticks(range(len(VERSIONS)), [lbl for _, _, lbl in VERSIONS], color=INK2, fontsize=8)
        ax.set_title(title, loc="left", color=INK, fontsize=11)
        for j, (name, tag, _) in enumerate(VERSIONS[1:]):
            d = delta(name, key)
            ax.annotate(f"{tag} {d:+.3f}", (0.98, 0.05 + 0.09 * (len(VERSIONS) - 2 - j)),
                        xycoords="axes fraction", ha="right", fontsize=8.5,
                        color=(WORSE if d < -0.005 else INK2))

    # bottom left: training time (zero-based)
    ax = fig.add_subplot(gs[1, 0:2])
    style(ax)
    fit_min = [vals[n]["fit_s"].mean() / 60 for n, _, _ in VERSIONS]
    ys = np.arange(len(VERSIONS))[::-1]
    ax.barh(ys, fit_min, color=COLORS, height=0.55)
    for y, m in zip(ys, fit_min):
        ax.text(m + 1, y, f"{m:.1f} min", va="center", color=INK, fontsize=10)
    ax.set_yticks(ys, [lbl.replace("\n", " - ") for _, _, lbl in VERSIONS], color=INK2, fontsize=8.5)
    ax.set_xlim(0, max(fit_min) * 1.18)
    ax.set_xlabel("Training time per run (minutes)", color=INK2)
    ax.set_title("Cost: training time", loc="left", color=INK, fontsize=11)
    ax.yaxis.grid(False)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)

    # bottom right: findings
    ax = fig.add_subplot(gs[1, 2:])
    ax.axis("off")
    v2, v3 = VERSIONS[1][0], VERSIONS[2][0]
    lines = [
        ("Findings (change vs v1, mean over seeds %s)" % seeds, True),
        ("v2 (384px, k=9): image level unchanged; pixel AUROC %+.3f, pixel F1 %+.3f;"
         % (delta(v2, "pixel_AUROC"), delta(v2, "pixel_F1Score")), False),
        ("   ~%dx slower to train. Worse overall." % round(fit_min[1] / fit_min[0]), False),
        ("v3 (2x augmented data): image AUROC %+.4f, image F1 %+.4f;"
         % (delta(v3, "image_AUROC"), delta(v3, "image_F1Score")), False),
        ("   pixel AUROC %+.4f, pixel F1 %+.4f (no real change). ~%.1fx slower."
         % (delta(v3, "pixel_AUROC"), delta(v3, "pixel_F1Score"), fit_min[2] / fit_min[0]), False),
        ("", False),
        ("Neither change is a clear improvement over v1; v3 is a small image-level", True),
        ("gain at best, within test-set noise (100 images).", True),
    ]
    y = 0.98
    for text, bold in lines:
        ax.text(0.0, y, text, transform=ax.transAxes, va="top", fontsize=10.5,
                color=INK if bold else INK2, fontweight="bold" if bold else "normal")
        y -= 0.105

    fig.suptitle("Patchcore versions on MVTec AD transistor (100-image test set)", x=0.01, y=0.99,
                 ha="left", color=INK, fontsize=14, fontweight="bold")
    fig.savefig(OUT_PNG, dpi=130, bbox_inches="tight", facecolor=SURFACE)
    print(f"Saved {OUT_PNG}")


if __name__ == "__main__":
    main()
