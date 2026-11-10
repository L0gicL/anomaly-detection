"""Create data/MVTecAD_aug: the transistor dataset with extra augmented normal training images.

Only train/good grows; test/ and ground_truth/ are copied unchanged so results stay comparable
with models trained on the original data. The original data/ folder is never modified.

Augmentations are deliberately mild and orientation-preserving (a rotated or flipped transistor
looks like the 'misplaced' defect, so no flips and only a few degrees of rotation):
  - brightness/contrast jitter, slight hue/saturation shift
  - small translation (+-2%) and rotation (+-3 deg), with reflected borders instead of black edges
  - light gaussian blur

Usage:
    uv run python scripts/make_augmented_data.py [--copies 1] [--seed 0]
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import v2

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data/MVTecAD/transistor"
DST = ROOT / "data/MVTecAD_aug/transistor"
PAD = 80  # reflect-pad before the affine so rotation/shift never exposes black corners


def link_or_copy(src: Path, dst: Path) -> None:
    try:
        os.link(src, dst)  # no extra disk space
    except OSError:
        shutil.copy2(src, dst)


def build_transform() -> v2.Compose:
    return v2.Compose(
        [
            v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02),
            v2.Pad(PAD, padding_mode="reflect"),
            v2.RandomAffine(degrees=3, translate=(0.02, 0.02)),
            v2.CenterCrop(1024),
            v2.GaussianBlur(kernel_size=3, sigma=(0.1, 0.8)),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--copies", type=int, default=1, help="augmented copies per training image")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not SRC.is_dir():
        raise SystemExit(f"Original data not found: {SRC} (run scripts/prepare_data.py first)")
    if DST.parent.exists():
        shutil.rmtree(DST.parent)  # rebuild from scratch so reruns are deterministic

    # test + ground truth unchanged
    for sub in ("test", "ground_truth"):
        for f in (SRC / sub).rglob("*"):
            if f.is_file():
                target = DST / f.relative_to(SRC)
                target.parent.mkdir(parents=True, exist_ok=True)
                link_or_copy(f, target)

    # training images: originals + augmented copies
    train_out = DST / "train" / "good"
    train_out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    transform = build_transform()
    originals = sorted((SRC / "train" / "good").glob("*.png"))
    for f in originals:
        link_or_copy(f, train_out / f.name)
        image = Image.open(f).convert("RGB")
        for c in range(args.copies):
            transform(image).save(train_out / f"{f.stem}_aug{c}.png")

    n = len(list(train_out.glob("*.png")))
    print(f"train/good: {len(originals)} originals + {len(originals) * args.copies} augmented = {n}")
    print(f"Wrote {DST}")


if __name__ == "__main__":
    main()
