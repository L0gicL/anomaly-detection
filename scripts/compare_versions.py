"""Train several Patchcore versions with multiple seeds and compare their test metrics.

Results are appended to results/comparison.json after every run (so a crash loses nothing),
then summarised in results/comparison.md and results/comparison.png.

Usage:
    uv run python scripts/compare_versions.py                 # train + report
    uv run python scripts/compare_versions.py --report-only   # rebuild report from the JSON
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

from anomaly_detection.experiment import Config, run

# v1 = Anomalib defaults as committed in the baseline; v2 = larger images + more neighbours.
CONFIGS = [
    Config(name="v1_baseline", image_size=256, num_neighbors=6),
    Config(name="v2_384px_k9", image_size=384, num_neighbors=9),
    # v3 = baseline settings, but trained on 2x data (originals + 1 augmented copy each)
    Config(name="v3_aug2x", image_size=256, num_neighbors=6, data_root="data/MVTecAD_aug"),
]
SEEDS = [0, 1, 2]
METRICS = ["image_AUROC", "image_F1Score", "pixel_AUROC", "pixel_F1Score"]
OUT_JSON = ROOT / "results/comparison.json"
OUT_MD = ROOT / "results/comparison.md"
OUT_PNG = ROOT / "results/comparison.png"

COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]  # reference palette slots 1-3 (blue, orange, aqua)
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"


def train_all() -> None:
    results = json.loads(OUT_JSON.read_text()) if OUT_JSON.exists() else []
    done = {(r["name"], r["seed"]) for r in results}
    for config in CONFIGS:
        for seed in SEEDS:
            if (config.name, seed) in done:
                print(f"[skip] {config.name} seed {seed}")
                continue
            print(f"[run] {config.name} seed {seed}", flush=True)
            metrics = run(config, seed, export=(seed == SEEDS[0]))
            results.append({"name": config.name, "seed": seed, **metrics})
            OUT_JSON.parent.mkdir(exist_ok=True)
            OUT_JSON.write_text(json.dumps(results, indent=2))
            print(f"[done] {metrics}", flush=True)


def report() -> None:
    results = json.loads(OUT_JSON.read_text())
    names = [c.name for c in CONFIGS]
    by_name = {n: [r for r in results if r["name"] == n] for n in names}

    def stats(name: str, key: str) -> tuple[float, float]:
        vals = np.array([r[key] for r in by_name[name]])
        return float(vals.mean()), float(vals.std())

    cfg = {c.name: c for c in CONFIGS}
    md = [
        "# Patchcore version comparison",
        "",
        f"Mean ± std over seeds {SEEDS} on the 100-image transistor test set.",
        "",
        "| Version | Image size | Neighbours | Training data |",
        "|---|---|---|---|",
        *[f"| {n} | {cfg[n].image_size}px | {cfg[n].num_neighbors} | {cfg[n].data_root} |"
          for n in names],
        "",
        "| Metric | " + " | ".join(names) + " | " + " | ".join(f"{n.split('_')[0]} - v1" for n in names[1:]) + " |",
        "|---|" + "---|" * (len(names) + len(names) - 1),
    ]
    for key in METRICS + ["fit_s", "test_s"]:
        cells = [f"{stats(n, key)[0]:.4f} ± {stats(n, key)[1]:.4f}" for n in names]
        deltas = [f"{stats(n, key)[0] - stats(names[0], key)[0]:+.4f}" for n in names[1:]]
        md.append(f"| {key} | " + " | ".join(cells) + " | " + " | ".join(deltas) + " |")
    md += ["", "Per-seed values:", "", "| Version | Seed | " + " | ".join(METRICS) + " |",
           "|---|---|" + "---|" * len(METRICS)]
    for n in names:
        for r in by_name[n]:
            md.append(f"| {n} | {r['seed']} | " + " | ".join(f"{r[k]:.4f}" for k in METRICS) + " |")
    md += ["", "![comparison](comparison.png)"]
    OUT_MD.write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))

    # figure: one panel per metric, per-seed dots with the mean as a bar
    fig, axes = plt.subplots(1, len(METRICS), figsize=(14, 3.8), facecolor=SURFACE)
    for ax, key in zip(axes, METRICS):
        ax.set_facecolor(SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.yaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(colors=INK2, labelsize=9)
        for i, n in enumerate(names):
            vals = np.array([r[key] for r in by_name[n]])
            ax.bar(i, vals.mean(), width=0.55, color=COLORS[i], alpha=0.85)
            ax.scatter(np.full(len(vals), i), vals, color=INK, s=14, zorder=3)
            ax.text(i, vals.max() + 0.004, f"{vals.mean():.3f}", ha="center", fontsize=9, color=INK)
        lo = min(r[key] for r in results)
        ax.set_ylim(max(0, lo - 0.08), 1.0 if "AUROC" in key else min(1.0, max(r[key] for r in results) + 0.08))
        ax.set_xticks(range(len(names)), [n.split("_")[0] for n in names], color=INK2)
        ax.set_title(key, loc="left", color=INK, fontsize=11)
    fig.suptitle("Patchcore versions (bars = mean, dots = individual seeds; y-axis truncated)",
                 x=0.01, ha="left", color=INK, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=130, bbox_inches="tight", facecolor=SURFACE)
    print(f"\nSaved {OUT_MD} and {OUT_PNG}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--train-only", action="store_true", help="skip the report at the end")
    args = parser.parse_args()
    if not args.report_only:
        train_all()
    if not args.train_only:
        report()
