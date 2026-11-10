"""Train + evaluate one Patchcore configuration on MVTec AD transistor."""

from __future__ import annotations

import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from anomalib.data import MVTecAD
from anomalib.deploy import ExportType
from anomalib.engine import Engine
from anomalib.models import Patchcore
from lightning.pytorch import seed_everything

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Config:
    name: str
    image_size: int = 256  # Anomalib's Patchcore default
    num_neighbors: int = 6
    coreset_sampling_ratio: float = 0.1
    batch_size: int = 8
    data_root: str = "data/MVTecAD"  # use data/MVTecAD_aug for extra augmented training images


def run_neighbor_sweep(config: Config, seed: int, ks: list[int]) -> dict[int, dict[str, float]]:
    """Train once, then score the test set with several neighbour counts.

    ``num_neighbors`` is only used at scoring time (not when the memory bank is built), so every
    k is evaluated on exactly the same trained memory bank.
    """
    seed_everything(seed, workers=True)
    scratch = PROJECT_ROOT / "results" / "_scratch" / f"{config.name}_sweep_seed{seed}"
    datamodule = MVTecAD(
        root=PROJECT_ROOT / config.data_root,
        category="transistor",
        train_batch_size=config.batch_size,
        eval_batch_size=config.batch_size,
        num_workers=0,
    )
    size = (config.image_size, config.image_size)
    model = Patchcore(
        num_neighbors=config.num_neighbors,
        coreset_sampling_ratio=config.coreset_sampling_ratio,
        pre_processor=Patchcore.configure_pre_processor(image_size=size),
    )
    engine = Engine(accelerator="gpu", devices=1, default_root_dir=scratch)
    engine.fit(datamodule=datamodule, model=model)

    out: dict[int, dict[str, float]] = {}
    for k in ks:
        model.model.num_neighbors = k
        metrics = engine.test(datamodule=datamodule, model=model, verbose=False)[0]
        out[k] = {key: float(v) for key, v in metrics.items()}
    shutil.rmtree(scratch, ignore_errors=True)
    return out


def run(config: Config, seed: int, export: bool = False) -> dict[str, float]:
    """Train and test one configuration; returns metrics plus fit/test times in seconds.

    With ``export=True`` the model is exported to ``models/<config.name>/``.
    Training outputs go to a scratch folder that is deleted afterwards.
    """
    seed_everything(seed, workers=True)
    scratch = PROJECT_ROOT / "results" / "_scratch" / f"{config.name}_seed{seed}"

    datamodule = MVTecAD(
        root=PROJECT_ROOT / config.data_root,
        category="transistor",
        train_batch_size=config.batch_size,
        eval_batch_size=config.batch_size,
        num_workers=0,  # multi-worker loading crashed silently on Windows
    )
    size = (config.image_size, config.image_size)
    model = Patchcore(
        num_neighbors=config.num_neighbors,
        coreset_sampling_ratio=config.coreset_sampling_ratio,
        pre_processor=Patchcore.configure_pre_processor(image_size=size),
    )
    engine = Engine(accelerator="gpu", devices=1, default_root_dir=scratch)

    t0 = time.perf_counter()
    engine.fit(datamodule=datamodule, model=model)
    fit_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    metrics = engine.test(datamodule=datamodule, model=model, verbose=False)[0]
    test_s = time.perf_counter() - t0

    if export:
        engine.export(
            model=model,
            export_type=ExportType.TORCH,
            export_root=PROJECT_ROOT / "models" / config.name,
        )
    shutil.rmtree(scratch, ignore_errors=True)

    return {**{k: float(v) for k, v in metrics.items()}, "fit_s": fit_s, "test_s": test_s}
