"""Shared preprocessing, postprocessing, and evaluation helpers for SAM2."""

from __future__ import annotations

import csv
import math
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import cv2
import numpy as np


@dataclass(frozen=True)
class Sam2PostprocessConfig:
    """Parameters used after SAM2 automatic mask generation.

    The defaults are the tuned values used by the benchmark. Keep this object
    as the single source of truth when changing the filtering chain.
    """

    points_per_side: int = 32
    pred_iou_thresh: float = 0.5
    stability_score_thresh: float = 0.30
    box_nms_thresh: float = 0.2
    min_mask_region_area: int = 20
    area_min: int = 20
    compactness_min: float = 0.1
    brightness_factor: float = 1.245
    colony_radius_div: float = 3.5
    colony_radius_bonus: float = 1.15
    nms_iou_thresh: float = 0.3
    nms_containment_thresh: float = 0.5
    max_area_fraction: float = 0.25


DEFAULT_SAM2_CONFIG = Sam2PostprocessConfig()


def pp_none(image_rgb: np.ndarray) -> np.ndarray:
    return image_rgb


def pp_gaussian(image_rgb: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(image_rgb, (5, 5), 0)


def pp_clahe(image_rgb: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    lightness, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lightness = clahe.apply(lightness)
    return cv2.cvtColor(cv2.merge((lightness, a_channel, b_channel)), cv2.COLOR_LAB2RGB)


def pp_sharpen(image_rgb: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(image_rgb, (0, 0), sigmaX=1.0)
    return cv2.addWeighted(image_rgb, 2.0, blur, -1.0, 0)


def pp_enhanced(image_rgb: np.ndarray) -> np.ndarray:
    return pp_sharpen(pp_clahe(pp_gaussian(image_rgb)))


def pp_clahe_sharpen(image_rgb: np.ndarray) -> np.ndarray:
    return pp_sharpen(pp_clahe(image_rgb))


PREPROCESS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "none": pp_none,
    "gaussian": pp_gaussian,
    "clahe": pp_clahe,
    "sharpen": pp_sharpen,
    "enhanced": pp_enhanced,
    "clahe_sharpen": pp_clahe_sharpen,
}


def load_ground_truth(csv_path: str | Path) -> dict[str, int]:
    """Load ``filename,count_after`` labels from a CSV file."""

    with Path(csv_path).open(newline="") as handle:
        return {
            row["filename"]: int(row["count_after"]) for row in csv.DictReader(handle)
        }


def find_images(
    image_dir: str | Path, ground_truth: Mapping[str, int]
) -> dict[str, str]:
    """Return image paths whose extension-less names occur in the labels."""

    image_dir = Path(image_dir)
    found: dict[str, str] = {}
    for filename in sorted(ground_truth):
        image_path = image_dir / f"{filename}.png"
        if image_path.is_file():
            found[filename] = str(image_path)
    return found


def _mask_array(mask: Mapping[str, Any]) -> np.ndarray:
    return np.asarray(mask["segmentation"], dtype=bool)


def postprocess_sam2_masks(
    masks: Iterable[Mapping[str, Any]],
    image_rgb: np.ndarray,
    config: Sam2PostprocessConfig = DEFAULT_SAM2_CONFIG,
) -> list[Mapping[str, Any]]:
    """Filter raw SAM2 masks into one mask per visible colony candidate.

    Filtering follows the benchmark chain: area and predicted-IoU gates,
    brightness and compactness filters, a circular-region prior, and mask NMS
    using both IoU and containment. The returned dictionaries retain SAM2's
    metadata and segmentation arrays for visualization or serialization.
    """

    height, width = image_rgb.shape[:2]
    max_area = int(height * width * config.max_area_fraction)
    quality_masks: list[Mapping[str, Any]] = []
    for mask in masks:
        area = int(mask["area"])
        if area < config.area_min or area > max_area:
            continue
        if float(mask.get("predicted_iou", 1.0)) < config.pred_iou_thresh:
            continue
        quality_masks.append(mask)

    if not quality_masks:
        return []

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    border = max(1, int(min(width, height) * 0.30))
    corners = np.concatenate(
        (
            gray[:border, :border].ravel(),
            gray[:border, -border:].ravel(),
            gray[-border:, :border].ravel(),
            gray[-border:, -border:].ravel(),
        )
    )
    bright_threshold = float(corners.mean()) * config.brightness_factor

    total_weight = 0.0
    weighted_x = 0.0
    weighted_y = 0.0
    for mask in quality_masks:
        segmentation = _mask_array(mask)
        ys, xs = np.where(segmentation)
        if len(xs) == 0:
            continue
        weight = float(mask.get("predicted_iou", 1.0)) * int(mask["area"])
        weighted_x += float(xs.mean()) * weight
        weighted_y += float(ys.mean()) * weight
        total_weight += weight

    image_cx = weighted_x / total_weight if total_weight else width / 2
    image_cy = weighted_y / total_weight if total_weight else height / 2
    colony_radius = (
        math.sqrt(width * height / (config.colony_radius_div * math.pi))
        * config.colony_radius_bonus
    )

    candidates: list[tuple[Mapping[str, Any], np.ndarray]] = []
    for mask in quality_masks:
        x, y, box_width, box_height = mask["bbox"]
        center_x = float(x) + float(box_width) / 2
        center_y = float(y) + float(box_height) / 2
        distance = math.hypot(center_x - image_cx, center_y - image_cy)
        if distance > colony_radius:
            continue

        segmentation = _mask_array(mask)
        area = int(mask["area"])
        contours, _ = cv2.findContours(
            (segmentation.astype(np.uint8) * 255),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        if contours and area > 0:
            perimeter = cv2.arcLength(contours[0], True)
            if perimeter > 0:
                compactness = 4 * math.pi * area / (perimeter * perimeter)
                if compactness < config.compactness_min:
                    continue

        mask_pixels = gray[segmentation]
        if mask_pixels.size and float(mask_pixels.mean()) < bright_threshold:
            continue
        candidates.append((mask, segmentation))

    candidates.sort(key=lambda item: int(item[1].sum()), reverse=True)
    kept: list[Mapping[str, Any]] = []
    kept_segmentations: list[np.ndarray] = []
    for mask, segmentation in candidates:
        segmentation_area = int(segmentation.sum())
        duplicate = False
        for kept_segmentation in kept_segmentations:
            intersection = int(np.logical_and(segmentation, kept_segmentation).sum())
            if intersection == 0:
                continue
            union = int(np.logical_or(segmentation, kept_segmentation).sum())
            iou = intersection / union if union else 0.0
            containment = intersection / segmentation_area if segmentation_area else 0.0
            if (
                iou > config.nms_iou_thresh
                or containment > config.nms_containment_thresh
            ):
                duplicate = True
                break
        if not duplicate:
            kept.append(mask)
            kept_segmentations.append(segmentation)

    return kept


def compute_metrics(y_true: Iterable[float], y_pred: Iterable[float]) -> OrderedDict:
    """Compute the metrics used by the benchmark."""

    true = np.asarray(list(y_true), dtype=np.float64)
    pred = np.asarray(list(y_pred), dtype=np.float64)
    if true.size == 0 or pred.size == 0 or true.size != pred.size:
        raise ValueError("y_true and y_pred must be non-empty and equally sized")

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
