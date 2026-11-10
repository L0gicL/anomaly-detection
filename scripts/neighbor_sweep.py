"""Sweep Patchcore's num_neighbors on the v3 (augmented 2x, 256px) training data.

Each seed is trained once and then scored with every k on the same memory bank.
Results go to results/neighbor_sweep.json (saved after every seed), then
results/neighbor_sweep.md and results/neighbor_sweep.png.

Usage:
    uv run python scripts/neighbor_sweep.py
    uv run python scripts/neighbor_sweep.py --report-only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from anomaly_detection.experiment import Config, run_neighbor_sweep

CONFIG = Config(name="v3_aug2x", image_size=256, num_neighbors=6, data_root="data/MVTecAD_aug")
KS = [1, 3, 6, 9, 15, 25]
SEEDS = [0, 1, 2]
METRICS = [("image_AUROC", "Image AUROC"), ("image_F1Score", "Image F1"),
           ("pixel_AUROC", "Pixel AUROC"), ("pixel_F1Score", "Pixel F1")]
OUT_JSON = ROOT / "results/neighbor_sweep.json"
OUT_MD = ROOT / "results/neighbor_sweep.md"
OUT_PNG = ROOT / "results/neighbor_sweep.png"
BLUE = "#2a78d6"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"


def train_all() -> None:
    results = json.loads(OUT_JSON.read_text()) if OUT_JSON.exists() else []
    done = {r["seed"] for r in results}
    for seed in SEEDS:
        if seed in done:
            print(f"[skip] seed {seed}")
            continue
        print(f"[run] seed {seed}", flush=True)
        per_k = run_neighbor_sweep(CONFIG, seed, KS)
        for k, metrics in per_k.items():
            results.append({"seed": seed, "k": k, **metrics})
        OUT_JSON.write_text(json.dumps(results, indent=2))
        print(f"[done] seed {seed}", flush=True)


def report() -> None:
    rows = json.loads(OUT_JSON.read_text())
    ks = sorted({r["k"] for r in rows})

    def vals(k: int, key: str) -> np.ndarray:
        return np.array([r[key] for r in rows if r["k"] == k])

    md = ["# v3 neighbour sweep", "",
          f"Mean ± std over seeds {SEEDS}; each seed trained once, scored with every k.", "",
          "| k | " + " | ".join(t for _, t in METRICS) + " |", "|---|" + "---|" * len(METRICS)]
    for k in ks:
        md.append(f"| {k} | " + " | ".join(f"{vals(k, m).mean():.4f} ± {vals(k, m).std():.4f}"
                                           for m, _ in METRICS) + " |")
    md += ["", "![sweep](neighbor_sweep.png)"]
    OUT_MD.write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6), facecolor=SURFACE)
    for ax, (key, title) in zip(axes, METRICS):
        ax.set_facecolor(SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.yaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(colors=INK2, labelsize=9)
        xs = np.arange(len(ks))
        means = np.array([vals(k, key).mean() for k in ks])
        for i, k in enumerate(ks):
            ax.scatter(np.full(len(vals(k, key)), i), vals(k, key), color=INK2, s=14, zorder=3)
        ax.plot(xs, means, color=BLUE, linewidth=2, marker="o", markersize=6, zorder=4)
        best = int(np.argmax(means))
        ax.annotate(f"{means[best]:.3f}", (xs[best], means[best]), xytext=(0, 9),
                    textcoords="offset points", ha="center", fontsize=9, color=INK,
                    fontweight="bold")
        pad = max(np.ptp(means) * 0.8, 0.004)
        ax.set_ylim(means.min() - pad, means.max() + pad * 1.6)
        ax.set_xticks(xs, [str(k) for k in ks])
        ax.set_xlabel("num_neighbors (k)", color=INK2, fontsize=9)
        ax.set_title(title, loc="left", color=INK, fontsize=11)
    fig.suptitle("v3 (256px, 2x augmented data): effect of num_neighbors "
                 "(line = mean, dots = seeds; y-axis truncated)", x=0.01, ha="left", color=INK,
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=130, bbox_inches="tight", facecolor=SURFACE)
    print(f"\nSaved {OUT_MD} and {OUT_PNG}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    if not args.report_only:
        train_all()
    report()
