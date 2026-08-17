# Third-party notices

This repository contains original project code and imports the following
external packages at runtime. The dependencies are not vendored into this
repository; their own licenses continue to apply to their respective code and
distributions.

| Dependency | Use here | License / notice |
| --- | --- | --- |
| [SAM2](https://github.com/facebookresearch/sam2) | Automatic mask generation | Apache License 2.0 |
| [PyTorch](https://github.com/pytorch/pytorch) | Model execution and inference mode | BSD-style license |
| [NumPy](https://github.com/numpy/numpy) | Array and metric operations | BSD-3-Clause and bundled notices |
| [OpenCV](https://github.com/opencv/opencv-python) | Image I/O and image processing | Apache License 2.0 |
| [scikit-image](https://github.com/scikit-image/scikit-image) | DoG and LoG blob detection | BSD-family licenses and bundled notices |

The MIT license in `LICENSE` applies to this repository's original code only;
it does not relicense external dependencies, SAM2 checkpoints, or datasets.
When redistributing an environment or binary package, follow the notices and
license terms supplied by the relevant dependency distributors.
