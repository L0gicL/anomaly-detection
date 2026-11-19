"""Summarise the latest trained Patchcore model: metrics, per-category results, charts.

Loads the newest checkpoint from results/Patchcore/MVTecAD/transistor/latest (no retraining)
and writes results/overview.png and results/overview.md.

Usage:
    uv run python scripts/results_overview.py [--ckpt path/to/model.ckpt]
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import Patchcore

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CKPT = ROOT / "results/Patchcore/MVTecAD/transistor/latest/weights/lightning/model.ckpt"
OUT_PNG = ROOT / "results/overview.png"
OUT_MD = ROOT / "results/overview.md"
CATEGORIES = ["good", "bent_lead", "cut_lead", "damaged_case", "misplaced"]

# Reference palette (dataviz skill): normal = blue, anomalous = orange; text in ink tokens.
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"


def collect(engine: Engine, model: Patchcore, datamodule: MVTecAD) -> dict[str, np.ndarray]:
    """Run inference over the test set and gather per-image results."""
    preds = engine.predict(model=model, datamodule=datamodule, ckpt_path=str(ARGS.ckpt))
    rows: dict[str, list] = defaultdict(list)
    for batch in preds:
        for i in range(len(batch.image)):
            rows["category"].append(Path(batch.image_path[i]).parent.name)
            rows["score"].append(float(batch.pred_score[i]))
            rows["pred"].append(bool(batch.pred_label[i]))
            rows["truth"].append(bool(batch.gt_label[i]))
    return {k: np.array(v) for k, v in rows.items()}


def style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")

    datamodule = MVTecAD(root=ROOT / "data/MVTecAD", category="transistor",
                         eval_batch_size=8, num_workers=0)
    engine = Engine(accelerator="gpu", devices=1)
    model = Patchcore(num_neighbors=6)

    metrics = engine.test(model=model, datamodule=datamodule, ckpt_path=str(ARGS.ckpt),
                          verbose=False)[0]
    r = collect(engine, model, datamodule)

    truth, pred, score, cat = r["truth"], r["pred"], r["score"], r["category"]
    tp = int((truth & pred).sum())
    tn = int((~truth & ~pred).sum())
    fp = int((~truth & pred).sum())
    fn = int((truth & ~pred).sum())
    acc = (tp + tn) / len(truth)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    # per-category: share of images classified correctly
    per_cat = {}
    for c in CATEGORIES:
        m = cat == c
        correct = (pred[m] == truth[m]).sum()
        per_cat[c] = (int(correct), int(m.sum()), float(score[m].mean()), float(score[m].max()))

    # ---- figure -------------------------------------------------------------------------
    fig = plt.figure(figsize=(14, 4.6), facecolor=SURFACE)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.25, 0.8], wspace=0.4)

    # 1) score distribution by true class
    ax = fig.add_subplot(gs[0])
    style(ax)
    bins = np.linspace(score.min(), score.max(), 25)
    ax.hist(score[~truth], bins=bins, color=BLUE, alpha=0.85, label=f"Normal (n={(~truth).sum()})")
    ax.hist(score[truth], bins=bins, color=ORANGE, alpha=0.85, label=f"Anomalous (n={truth.sum()})")
    top = ax.get_ylim()[1] * 1.4
    ax.set_ylim(0, top)  # headroom so legend and threshold label clear the bars
    ax.axvline(0.5, color=INK2, linestyle="--", linewidth=1)
    ax.text(0.51, top * 0.75, "decision threshold (0.5)", color=INK2, fontsize=8, va="top")
    ax.set_xlabel("Anomaly score", color=INK2)
    ax.set_ylabel("Images", color=INK2)
    ax.set_title("Scores separate the two classes", color=INK, loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="upper right")

    # 2) per-category correct rate
    ax = fig.add_subplot(gs[1])
    style(ax)
    labels = [c.replace("_", "\n") for c in CATEGORIES]
    rates = [per_cat[c][0] / per_cat[c][1] * 100 for c in CATEGORIES]
    colors = [BLUE if c == "good" else ORANGE for c in CATEGORIES]
    bars = ax.bar(labels, rates, color=colors, width=0.6)
    for b, c in zip(bars, CATEGORIES):
        k, n = per_cat[c][:2]
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5, f"{k}/{n}", ha="center",
                color=INK, fontsize=9)
    ax.set_ylim(0, 112)
    ax.set_ylabel("Classified correctly (%)", color=INK2)
    ax.set_title("Correct classification by category", color=INK, loc="left", fontsize=11)
    ax.tick_params(axis="x", labelsize=8)

    # 3) confusion matrix (counts)
    ax = fig.add_subplot(gs[2])
    ax.set_facecolor(SURFACE)
    cm = np.array([[tn, fp], [fn, tp]])
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max() * 1.15)
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center", fontsize=16,
                color="white" if v > cm.max() * 0.5 else INK)
    ax.set_xticks([0, 1], ["Normal", "Anomalous"], color=INK2, fontsize=9)
    ax.set_yticks([0, 1], ["Normal", "Anomalous"], color=INK2, fontsize=9)
    ax.set_xlabel("Predicted", color=INK2)
    ax.set_ylabel("Actual", color=INK2)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Confusion matrix", color=INK, loc="left", fontsize=11)

    fig.suptitle("Patchcore on MVTec AD transistor - test set results", color=INK, x=0.01,
                 ha="left", fontsize=13, fontweight="bold")
    OUT_PNG.parent.mkdir(exist_ok=True)
    fig.savefig(OUT_PNG, dpi=130, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)

    # ---- markdown -----------------------------------------------------------------------
    md = [
        "# Patchcore - transistor results",
        "",
        f"Checkpoint: `{ARGS.ckpt.relative_to(ROOT) if ARGS.ckpt.is_relative_to(ROOT) else ARGS.ckpt}`",
        f"Test set: {len(truth)} images ({(~truth).sum()} normal, {truth.sum()} anomalous)",
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "|---|---|",
        *[f"| {k} | {v:.4f} |" for k, v in metrics.items()],
        f"| image accuracy | {acc:.4f} |",
        f"| image precision | {precision:.4f} |",
        f"| image recall | {recall:.4f} |",
        "",
        "## Confusion matrix",
        "",
        "| | Predicted normal | Predicted anomalous |",
        "|---|---|---|",
        f"| **Actual normal** | {tn} | {fp} |",
        f"| **Actual anomalous** | {fn} | {tp} |",
        "",
        "## Per category",
        "",
        "| Category | Correct | Mean score | Max score |",
        "|---|---|---|---|",
        *[f"| {c} | {k}/{n} | {m:.3f} | {mx:.3f} |" for c, (k, n, m, mx) in per_cat.items()],
        "",
        "![overview](overview.png)",
    ]
    OUT_MD.write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    print(f"\nSaved {OUT_PNG} and {OUT_MD}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    ARGS = parser.parse_args()
    main()
