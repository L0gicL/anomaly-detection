from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save figures to disk, no GUI window
import matplotlib.pyplot as plt
import numpy as np

from anomalib.data import MVTecAD
from anomalib.deploy import ExportType
from anomalib.engine import Engine
from anomalib.models import Patchcore

EXPORT_ROOT = Path("models/Patchcore/transistor")
FIGURE_DIR = Path("results/figures")
N_PER_CLASS = 3  # figures saved per test category
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def save_prediction_figures(predictions) -> None:
    """Save image | anomaly map | overlay figures, up to N_PER_CLASS per test category."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for old in FIGURE_DIR.glob("prediction_*.png"):
        old.unlink()  # remove figures from earlier runs
    saved: Counter = Counter()
    for batch in predictions:
        for i in range(len(batch.image)):
            image_path = Path(batch.image_path[i])
            category = image_path.parent.name  # good, bent_lead, cut_lead, ...
            if saved[category] >= N_PER_CLASS:
                continue
            image = batch.image[i].permute(1, 2, 0).cpu().numpy()
            image = (image * IMAGENET_STD + IMAGENET_MEAN).clip(0, 1)  # undo input normalization
            amap = batch.anomaly_map[i].squeeze().cpu().numpy()
            score = float(batch.pred_score[i])
            pred = "anomalous" if bool(batch.pred_label[i]) else "normal"
            truth = "anomalous" if bool(batch.gt_label[i]) else "normal"

            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(image)
            axes[0].set_title(f"{category} (truth: {truth})")
            axes[1].imshow(amap, cmap="jet")
            axes[1].set_title("Anomaly map")
            axes[2].imshow(image)
            axes[2].imshow(amap, cmap="jet", alpha=0.45)
            axes[2].set_title(f"Predicted: {pred} (score {score:.2f})")
            for ax in axes:
                ax.axis("off")
            fig.tight_layout()
            fig.savefig(FIGURE_DIR / f"prediction_{category}_{image_path.stem}.png", dpi=100)
            plt.close(fig)
            saved[category] += 1


def main() -> None:
    datamodule = MVTecAD(
        root="./data/MVTecAD",
        category="transistor",
        train_batch_size=8,
        eval_batch_size=8,
        num_workers=0,  # multi-worker loading crashed silently on Windows
    )

    model = Patchcore(num_neighbors=6)

    engine = Engine(accelerator="gpu", devices=1, max_epochs=3)

    engine.fit(datamodule=datamodule, model=model)

    test_results = engine.test(datamodule=datamodule, model=model)
    print(test_results)

    # Export the trained model as a PyTorch file into models/ (git-ignored).
    path = engine.export(model=model, export_type=ExportType.TORCH, export_root=EXPORT_ROOT)
    print(f"Exported torch model: {path}")

    # Run inference on the test set and save visual results.
    predictions = engine.predict(model=model, datamodule=datamodule)
    save_prediction_figures(predictions)
    print(f"Saved figures to {FIGURE_DIR.resolve()}")


if __name__ == "__main__":
    main()
