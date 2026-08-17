"""Traditional image-processing methods for microbial colony counting."""

from __future__ import annotations

import argparse
import csv
import time
from collections import OrderedDict
from pathlib import Path
from typing import Callable

import cv2
import numpy as np


def load_ground_truth(csv_path: str | Path) -> dict[str, int]:
    with Path(csv_path).open(newline="") as handle:
        return {
            row["filename"]: int(row["count_after"]) for row in csv.DictReader(handle)
        }


def find_images(image_dir: str | Path, ground_truth: dict[str, int]) -> dict[str, str]:
    image_dir = Path(image_dir)
    return {
        filename: str(image_dir / f"{filename}.png")
        for filename in sorted(ground_truth)
        if (image_dir / f"{filename}.png").is_file()
    }


def _count_components(binary: np.ndarray, min_area: int = 8) -> int:
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    return sum(
        int(stats[index, cv2.CC_STAT_AREA] >= min_area)
        for index in range(1, num_labels)
    )


def method_otsu(image: np.ndarray, min_area: int = 8) -> int:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return _count_components(binary, min_area)


def method_adaptive(
    image: np.ndarray,
    min_area: int = 8,
    block_size: int = 31,
    c_value: int = 5,
) -> int:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size,
        c_value,
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    return _count_components(binary, min_area)


def method_watershed(image: np.ndarray, min_area: int = 8) -> int:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

    distance = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    _, seeds = cv2.threshold(distance, 0.3 * distance.max(), 255, cv2.THRESH_BINARY)
    _, markers = cv2.connectedComponents(np.uint8(seeds))
    markers = markers + 1
    markers[binary == 0] = 0
    markers = cv2.watershed(
        cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), markers.astype(np.int32)
    )
    labels = set(markers.ravel()) - {-1, 0, 1}
    return len(labels)


def method_dog(
    image: np.ndarray,
    min_sigma: float = 1,
    max_sigma: float = 10,
    threshold: float = 0.1,
) -> int:
    from skimage.feature import blob_dog

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0
    return len(
        blob_dog(
            gray,
            min_sigma=min_sigma,
            max_sigma=max_sigma,
            threshold=threshold,
        )
    )


def method_log(
    image: np.ndarray,
    min_sigma: float = 1,
    max_sigma: float = 10,
    num_sigma: int = 10,
    threshold: float = 0.05,
) -> int:
    from skimage.feature import blob_log

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0
    return len(
        blob_log(
            gray,
            min_sigma=min_sigma,
            max_sigma=max_sigma,
            num_sigma=num_sigma,
            threshold=threshold,
        )
    )


def method_morphology(image: np.ndarray, min_area: int = 8) -> int:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, open_kernel, iterations=2)
    return _count_components(binary, min_area)


def method_hsv(image: np.ndarray, min_area: int = 8) -> int:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    _, binary_value = cv2.threshold(
        value, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    _, binary_saturation = cv2.threshold(
        saturation, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    binary = cv2.bitwise_or(binary_value, binary_saturation)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    return _count_components(binary, min_area)


METHODS: OrderedDict[str, Callable[[np.ndarray], int]] = OrderedDict(
    [
        ("Otsu+CC", method_otsu),
        ("AdaptiveGauss", method_adaptive),
        ("Watershed", method_watershed),
        ("DOG_Blob", method_dog),
        ("LoG_Blob", method_log),
        ("Morphology", method_morphology),
        ("HSV_Color", method_hsv),
    ]
)


def compute_metrics(y_true: list[int], y_pred: list[int]) -> OrderedDict:
    true = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(y_pred, dtype=np.float64)
    if true.size == 0 or true.size != pred.size:
        raise ValueError("No valid image predictions were produced")
    errors = pred - true
    absolute_errors = np.abs(errors)
    ss_total = np.sum((true - true.mean()) ** 2)
    pearson = (
        float(np.corrcoef(true, pred)[0, 1])
        if np.std(true) > 0 and np.std(pred) > 0
        else 0.0
    )
    return OrderedDict(
        [
            ("MAE", float(np.mean(absolute_errors))),
            ("RMSE", float(np.sqrt(np.mean(errors**2)))),
            ("R²", float(1 - np.sum(errors**2) / ss_total) if ss_total else 0.0),
            ("Pearson_r", pearson),
            ("MedAE", float(np.median(absolute_errors))),
        ]
    )


def run_benchmark(
    image_dir: str | Path,
    gt_csv: str | Path,
    output_csv: str | Path | None = None,
    max_images: int | None = None,
) -> list[dict[str, float | str]]:
    ground_truth = load_ground_truth(gt_csv)
    images = find_images(image_dir, ground_truth)
    filenames = sorted(images)
    if max_images is not None:
        filenames = filenames[:max_images]

    predictions: dict[str, list[int]] = {name: [] for name in METHODS}
    processed_names: list[str] = []
    timings = {name: 0.0 for name in METHODS}

    print(f"Ground-truth entries: {len(ground_truth)}")
    print(f"Images found: {len(images)}; running on: {len(filenames)}")
    for index, filename in enumerate(filenames, start=1):
        image = cv2.imread(images[filename])
        if image is None:
            print(f"[WARN] Cannot read {images[filename]}")
            continue
        processed_names.append(filename)
        for name, method in METHODS.items():
            started = time.perf_counter()
            try:
                prediction = int(method(image))
            except Exception as exc:
                print(f"[WARN] {name} failed on {filename}: {exc}")
                prediction = 0
            timings[name] += time.perf_counter() - started
            predictions[name].append(prediction)
        if index % 200 == 0 or index == len(filenames):
            print(f"Processed {index}/{len(filenames)} images")

    true_values = [ground_truth[name] for name in processed_names]
    results: list[dict[str, float | str]] = []
    for name in METHODS:
        metrics = compute_metrics(true_values, predictions[name])
        result: dict[str, float | str] = {"Method": name, **metrics}
        result["Time(s)"] = timings[name]
        results.append(result)
        print(
            f"{name:16s} MAE={metrics['MAE']:.2f} "
            f"RMSE={metrics['RMSE']:.2f} R²={metrics['R²']:.4f}"
        )

    if output_csv is not None:
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["filename", "ground_truth", *METHODS])
            for index, filename in enumerate(processed_names):
                writer.writerow(
                    [
                        filename,
                        ground_truth[filename],
                        *(predictions[name][index] for name in METHODS),
                    ]
                )
        print(f"Detailed results saved to {output_path}")
    return results


def _default_paths() -> tuple[Path, Path, Path]:
    project_root = Path.cwd()
    return (
        project_root / "all_pic",
        project_root / "data" / "merged.csv",
        project_root / "results" / "benchmark_traditional.csv",
    )


def main() -> None:
    default_image_dir, default_gt_csv, default_output = _default_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", default=default_image_dir)
    parser.add_argument("--gt-csv", default=default_gt_csv)
    parser.add_argument("--output", default=default_output)
    parser.add_argument("--max-images", type=int, default=None)
    args = parser.parse_args()
    run_benchmark(args.image_dir, args.gt_csv, args.output, args.max_images)


if __name__ == "__main__":
    main()
