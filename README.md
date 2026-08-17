# Colony Segmentation with Classical Vision and SAM2

[![CI](https://github.com/zhuhw0916/colony-segmentation-sam2/actions/workflows/ci.yml/badge.svg)](https://github.com/zhuhw0916/colony-segmentation-sam2/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repository provides reproducible baselines and SAM2 inference utilities
for bacterial colony segmentation and counting. It contains source code and
tests only: image datasets, ground-truth files, model checkpoints, SAM2
source, and generated results are intentionally kept outside the repository.

The code is designed to be used as a component in a larger image-analysis or
multimodal project. It does not perform species classification or downstream
multimodal analysis.

## Contents

- `traditional_benchmark.py`: Otsu, adaptive thresholding, watershed, DoG,
  LoG, morphology, and HSV connected-component/blob counting.
- `sam2_benchmark.py`: SAM2 Tiny trained/base automatic-mask inference with
  the tuned colony postprocessing chain and optional preprocessing variants.
- `sam2_pipeline.py`: saves filtered masks as `.json.gz`, a count CSV, and
  optional mask overlays.
- `sam2_common.py`: shared parameters, preprocessing, postprocessing, and
  evaluation helpers.

Repository maintenance files include `pyproject.toml`, `LICENSE`, tests,
continuous integration, contribution guidelines, a security policy, and
third-party notices.

Images, ground-truth CSVs, SAM2 source code, checkpoints, and generated
outputs are deliberately not included.

## Installation

Python 3.10 or newer is supported. For local development:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

The benchmark environment can also be managed with conda. SAM2 is not
installed by this repository; use an existing SAM2 checkout and pass its path
with `--sam2-repo`.

## Data format

The benchmark expects a label CSV with the columns `filename,count_after`.
Images are PNG files named `<filename>.png`. Both paths are configurable from
the command line.

For a labelled image named `sample_001.png`, the CSV row is:

```csv
filename,count_after
sample_001,24
```

## Traditional image processing

From the project root:

```bash
python traditional_benchmark.py \
  --image-dir all_pic \
  --gt-csv data/merged.csv \
  --output results/benchmark_traditional.csv
```

## SAM2 benchmark

SAM2 is loaded from an existing checkout; neither SAM2 nor its weights are
vendored here. The default postprocessing values are:

| Parameter | Value |
| --- | ---: |
| points per side | 32 |
| predicted IoU threshold | 0.5 |
| stability threshold | 0.30 |
| minimum mask area | 20 |
| compactness lower bound | 0.1 |
| brightness factor | 1.245 |
| colony radius divisor | 3.5 |
| mask NMS IoU threshold | 0.3 |

Example using both the trained and base checkpoints:

```bash
python sam2_benchmark.py \
  --image-dir all_pic \
  --gt-csv data/merged.csv \
  --sam2-repo /path/to/sam2 \
  --trained-checkpoint /path/to/sam2_hiera_tiny_prep.pt \
  --base-checkpoint /path/to/sam2_hiera_tiny.pt \
  --model both \
  --gpu 0 \
  --preprocess none \
  --output results/benchmark_sam2.csv
```

For a quick smoke test, add `--max-images 2`. To run only one checkpoint,
use `--model trained` or `--model base` and provide the corresponding path.

## SAM2 mask export

```bash
python sam2_pipeline.py \
  --image-dir all_pic \
  --sam2-repo /path/to/sam2 \
  --checkpoint /path/to/sam2_hiera_tiny.pt \
  --gt-csv data/merged.csv \
  --output-dir sam2_output \
  --gpu 0 \
  --visualize
```

The pipeline writes one compressed mask file per image under `masks/`, a
`colony_counts.csv`, and optional PNG overlays under `overlays/`.

## Testing and formatting

The CPU-only tests do not require a SAM2 checkpoint:

```bash
python -m compileall -q .
python -m pytest -q
black --check .
```

The SAM2 commands require a compatible PyTorch/CUDA environment, a SAM2
checkout, and a checkpoint. See `CONTRIBUTING.md` before changing filtering
parameters or output formats.

## Citation

If you use this repository in academic work, please cite the associated paper
when it is published. Until the paper metadata is available, the repository's
`CITATION.cff` file provides the software citation record.

## License and third-party software

The original code in this repository is released under the MIT License. This
license does not relicense SAM2, PyTorch, NumPy, OpenCV, scikit-image, model
checkpoints, or datasets. See `THIRD_PARTY_NOTICES.md` for the dependency
licenses and attribution links.
