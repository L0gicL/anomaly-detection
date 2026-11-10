# Anomaly Detection

Visual anomaly detection on the **MVTec AD – transistor** category using [Anomalib](https://github.com/open-edge-platform/anomalib).

## Dataset

The dataset is not included in this repo (the MVTec AD license is CC BY-NC-SA 4.0). Place `transistor.tar.xz` in the project root and extract it:

```bash
mkdir -p data/MVTecAD && tar -xJf transistor.tar.xz -C data/MVTecAD
```

Expected layout:

```
data/MVTecAD/transistor/
├── train/good/            # 213 normal images
├── test/good/             # 60 normal images
├── test/{bent_lead,cut_lead,damaged_case,misplaced}/   # 40 anomalous images
└── ground_truth/{bent_lead,cut_lead,damaged_case,misplaced}/  # pixel masks
```

## Setup

Requires [uv](https://docs.astral.sh/uv/). This creates a Python 3.12 virtual environment with PyTorch (CUDA 13.0 build) and Anomalib:

```bash
uv sync
```

## Project structure

```
configs/                 # Anomalib model / data configs
src/anomaly_detection/   # Project source code
scripts/                 # Training / evaluation / export entry points
notebooks/               # Exploration notebooks
tests/                   # Tests
results/                 # Training outputs (gitignored)
models/                  # Exported models (gitignored)
data/                    # Dataset (gitignored)
```
