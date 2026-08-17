# High-Throughput Microbial Colony Image Analysis

[![CI](https://github.com/zhuhw0916/colony-segmentation-sam2/actions/workflows/ci.yml/badge.svg)](https://github.com/zhuhw0916/colony-segmentation-sam2/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repository provides the image-analysis module of a reproducible SOP for
high-throughput microbial colony imaging. It includes tools for image
preprocessing, colony segmentation, colony enumeration, and quantitative
benchmarking of traditional computer-vision and SAM2-based methods. The
workflow converts microbial colony images into quantitative segmentation and
counting results for downstream microbiological analysis.

The repository contains source code and tests only: image datasets,
ground-truth files, model checkpoints, SAM2 source, and generated results are
intentionally kept outside the repository.

## Scope

This repository covers the image-analysis stage of the SOP, including image
preprocessing, segmentation, enumeration, and method evaluation. Wet-lab
experimental procedures, species identification, and downstream statistical
or multimodal analyses are outside the scope of this repository.

## Workflow position

```text
High-throughput images
        ↓
Image preprocessing
        ↓
Colony segmentation
        ↓
Colony enumeration and method evaluation
        ↓
Counts, masks, overlays, and benchmark metrics
```

## Repository layout

- `src/colony_segmentation/`: reusable image-analysis modules and command-line
  entry points.
- `tests/`: CPU-only unit tests for preprocessing, postprocessing, metrics, and
  traditional methods.
- `.github/`: continuous integration and contribution templates.
- Root-level documentation and metadata: the SOP context, citation record,
  license, dependency notices, and development guidance.

## Contents

- `src/colony_segmentation/traditional_benchmark.py`: Otsu, adaptive
  thresholding, watershed, DoG,
  LoG, morphology, and HSV connected-component/blob counting.
- `src/colony_segmentation/sam2_benchmark.py`: SAM2 Tiny trained/base
  automatic-mask inference with
  the tuned colony postprocessing chain and optional preprocessing variants.
- `src/colony_segmentation/sam2_pipeline.py`: saves filtered masks as `.json.gz`,
  a count CSV, and
  optional mask overlays.
- `src/colony_segmentation/sam2_common.py`: shared parameters, preprocessing,
  postprocessing, and
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
with `--sam2-repo`. The SAM2 source reference, observed dependency versions,
and license boundary are recorded in `THIRD_PARTY_NOTICES.md`.

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

From the project root, after installation:

```bash
python -m colony_segmentation.traditional_benchmark \
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
python -m colony_segmentation.sam2_benchmark \
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
python -m colony_segmentation.sam2_pipeline \
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
python -m compileall -q src tests
python -m pytest -q
black --check src tests
```

The SAM2 commands require a compatible PyTorch/CUDA environment, a SAM2
checkout, and a checkpoint. See `CONTRIBUTING.md` before changing filtering
parameters or output formats.

## Citation

If you use this repository in academic work, please cite the associated paper
when it is published. Until the paper metadata is available, the repository's
`CITATION.cff` file provides the software citation record.

If you use the SAM2-based methods, also cite the SAM 2 paper:

```bibtex
@article{ravi2024sam2,
  title={SAM 2: Segment Anything in Images and Videos},
  author={Ravi, Nikhila and Gabeur, Valentin and Hu, Yuan-Ting and Hu, Ronghang and Ryali, Chaitanya and Ma, Tengyu and Khedr, Haitham and R{\"a}dle, Roman and Rolland, Chloe and Gustafson, Laura and Mintun, Eric and Pan, Junting and Alwala, Kalyan Vasudev and Carion, Nicolas and Wu, Chao-Yuan and Girshick, Ross and Doll{\'a}r, Piotr and Feichtenhofer, Christoph},
  journal={arXiv preprint arXiv:2408.00714},
  url={https://arxiv.org/abs/2408.00714},
  year={2024}
}
```

## License and third-party software

The original code in this repository is released under the MIT License. This
license does not relicense SAM2, PyTorch, NumPy, OpenCV, scikit-image, model
checkpoints, or datasets. See `THIRD_PARTY_NOTICES.md` for the dependency
licenses and attribution links.
