"""Prepare the MVTec AD transistor category for Anomalib.

Steps:
  1. Extract ``transistor.tar.xz`` into ``data/MVTecAD`` (skipped if already extracted).
  2. Validate the folder layout and image/mask counts.
  3. Build the Anomalib ``MVTecAD`` datamodule and run a smoke test on one batch.

Usage:
    uv run python scripts/prepare_data.py
    uv run python scripts/prepare_data.py --archive path/to/transistor.tar.xz --root data/MVTecAD
"""

from __future__ import annotations

import argparse
import sys
import tarfile
from pathlib import Path

from anomalib.data import MVTecAD

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CATEGORY = "transistor"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}
DEFECT_TYPES = ["bent_lead", "cut_lead", "damaged_case", "misplaced"]
EXPECTED_TRAIN_GOOD = 213
EXPECTED_TEST_GOOD = 60


def extract_archive(archive: Path, root: Path) -> None:
    """Extract the archive into ``root`` unless the category folder already exists."""
    if (root / CATEGORY).is_dir():
        print(f"[skip] {root / CATEGORY} already exists")
        return
    if not archive.is_file():
        sys.exit(f"Archive not found: {archive}\nPlace transistor.tar.xz in the project root.")

    root.mkdir(parents=True, exist_ok=True)
    print(f"[extract] {archive} -> {root}")
    with tarfile.open(archive, "r:xz") as tar:
        tar.extractall(root, filter="data")  # "data" filter blocks path traversal


def count_images(folder: Path) -> int:
    return sum(1 for p in folder.glob("*") if p.suffix.lower() in IMAGE_EXTS)


def validate_layout(root: Path) -> None:
    """Check the folder structure and that every anomalous image has a mask."""
    base = root / CATEGORY
    problems: list[str] = []

    train_good = count_images(base / "train" / "good")
    test_good = count_images(base / "test" / "good")
    print(f"train/good: {train_good} images (expected {EXPECTED_TRAIN_GOOD})")
    print(f"test/good:  {test_good} images (expected {EXPECTED_TEST_GOOD})")
    if train_good != EXPECTED_TRAIN_GOOD:
        problems.append(f"train/good has {train_good} images")
    if test_good != EXPECTED_TEST_GOOD:
        problems.append(f"test/good has {test_good} images")

    for defect in DEFECT_TYPES:
        images = sorted((base / "test" / defect).glob("*"))
        images = [p for p in images if p.suffix.lower() in IMAGE_EXTS]
        masks = {p.stem for p in (base / "ground_truth" / defect).glob("*")}
        # MVTec masks are named "<id>_mask.png" for test image "<id>.png"
        missing = [p.name for p in images if f"{p.stem}_mask" not in masks]
        print(f"test/{defect}: {len(images)} images, {len(masks)} masks")
        if not images:
            problems.append(f"test/{defect} is empty or missing")
        if missing:
            problems.append(f"test/{defect}: {len(missing)} images without masks")

    if problems:
        sys.exit("Layout validation failed:\n  - " + "\n  - ".join(problems))
    print("[ok] layout validated")


def build_datamodule(root: Path, batch_size: int, num_workers: int) -> MVTecAD:
    """Create the Anomalib datamodule and print split sizes plus one batch's shapes."""
    datamodule = MVTecAD(
        root=root,
        category=CATEGORY,
        train_batch_size=batch_size,
        eval_batch_size=batch_size,
        num_workers=num_workers,
    )
    datamodule.setup()
    print(f"[datamodule] train={len(datamodule.train_data)} test={len(datamodule.test_data)}")

    batch = next(iter(datamodule.train_dataloader()))
    print(f"[datamodule] train batch image shape: {tuple(batch.image.shape)}")
    return datamodule


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--archive", type=Path, default=PROJECT_ROOT / "transistor.tar.xz")
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT / "data" / "MVTecAD")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0, help="0 is safest on Windows")
    args = parser.parse_args()

    extract_archive(args.archive, args.root)
    validate_layout(args.root)
    build_datamodule(args.root, args.batch_size, args.num_workers)


if __name__ == "__main__":
    main()
