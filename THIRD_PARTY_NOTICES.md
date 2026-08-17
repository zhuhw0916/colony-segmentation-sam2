# Third-party notices

This repository contains original project code and imports the following
external packages at runtime. The dependencies are not vendored into this
repository; their own licenses continue to apply to their respective code and
distributions.

## Runtime dependencies and attribution

The versions below are the versions observed in the `seg_benchmark` environment
when this record was prepared. The version constraints in `pyproject.toml` and
`requirements.txt` remain the installation interface; this table is a
reproducibility record rather than a lock file.

| Dependency | Version or source reference | Use here | License / attribution |
| --- | --- | --- | --- |
| [SAM2](https://github.com/facebookresearch/sam2) | Local checkout HEAD: `9e380bb046129177052c59223e38d122a48627fc`; upstream package metadata: `1.0` | Automatic mask generation and SAM2 inference | [Apache License 2.0](https://github.com/facebookresearch/sam2/blob/main/LICENSE) |
| [PyTorch](https://github.com/pytorch/pytorch) | `2.7.1+cu128` | Model execution and inference mode | [BSD-style license](https://github.com/pytorch/pytorch/blob/main/LICENSE) |
| [Torchvision](https://github.com/pytorch/vision) | `0.22.1+cu128` | SAM2/PyTorch model support | [BSD-style license](https://github.com/pytorch/vision/blob/main/LICENSE) |
| [NumPy](https://github.com/numpy/numpy) | `2.2.6` | Array and metric operations | [BSD-3-Clause](https://github.com/numpy/numpy/blob/main/LICENSE.txt) and bundled notices |
| [OpenCV](https://github.com/opencv/opencv-python) | `4.13.0.92` | Image I/O and image processing | [Apache License 2.0](https://github.com/opencv/opencv-python/blob/master/LICENSE.txt) |
| [scikit-image](https://github.com/scikit-image/scikit-image) | `0.25.2` | DoG and LoG blob detection | [BSD-3-Clause](https://github.com/scikit-image/scikit-image/blob/main/LICENSE.txt) |
| [SciPy](https://github.com/scipy/scipy) | `1.15.3` | Numerical dependency used by scikit-image | [BSD-3-Clause](https://github.com/scipy/scipy/blob/main/LICENSE.txt) |

The external SAM2 installation has additional dependencies, including
`tqdm`, `hydra-core`, `iopath`, and `Pillow`. Their versions and license
notices are managed by the SAM2 installation and are not bundled or
re-licensed by this repository.

## SAM2 source checkout note

The local SAM2 checkout inspected for this project was in detached-`HEAD`
state at `9e380bb046129177052c59223e38d122a48627fc`. It also contained
uncommitted and untracked project-specific changes. The commit identifies the
checked-out base revision, but it is not a complete snapshot of those local
changes. This repository does not vendor SAM2 source, those local changes,
SAM2 checkpoints, or datasets. For exact reproduction of historical runs,
preserve the corresponding SAM2 checkout and its local patch set.

## License boundary

The MIT license in `LICENSE` applies to this repository's original code only;
it does not relicense external dependencies, SAM2 checkpoints, or datasets.
When redistributing an environment or binary package, follow the notices and
license terms supplied by the relevant dependency distributors.
