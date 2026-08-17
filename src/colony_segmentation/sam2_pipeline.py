"""Generate filtered SAM2 masks, counts, and optional overlays for images."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

try:
    from .sam2_benchmark import _autocast, _load_sam2
    from .sam2_common import (
        DEFAULT_SAM2_CONFIG,
        PREPROCESS,
        find_images,
        load_ground_truth,
        postprocess_sam2_masks,
    )
except ImportError:  # pragma: no cover - used by direct script execution.
    from sam2_benchmark import _autocast, _load_sam2  # type: ignore[no-redef]
    from sam2_common import (  # type: ignore[no-redef]
        DEFAULT_SAM2_CONFIG,
        PREPROCESS,
        find_images,
        load_ground_truth,
        postprocess_sam2_masks,
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _save_masks(path: Path, masks: list[dict[str, Any]]) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(_jsonable(masks), handle, separators=(",", ":"))


def _save_overlay(
    image_rgb: np.ndarray, masks: list[dict[str, Any]], path: Path
) -> None:
    overlay = image_rgb.copy()
    rng = np.random.default_rng(42)
    for mask in masks:
        segmentation = np.asarray(mask["segmentation"], dtype=bool)
        color = rng.integers(50, 255, size=3, dtype=np.uint8)
        color_layer = np.zeros_like(overlay)
        color_layer[segmentation] = color
        overlay = cv2.addWeighted(overlay, 0.7, color_layer, 0.3, 0)
        x, y, width, height = (int(v) for v in mask["bbox"])
        cv2.rectangle(
            overlay, (x, y), (x + width, y + height), tuple(int(v) for v in color), 1
        )
    cv2.putText(
        overlay,
        f"colonies: {len(masks)}",
        (5, max(18, overlay.shape[0] - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        1,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))


def run_pipeline(
    image_dir: str | Path,
    output_dir: str | Path,
    checkpoint: str | Path,
    *,
    sam2_repo: str | Path | None,
    config_name: str,
    device: str,
    preprocess_name: str = "none",
    max_images: int | None = None,
    visualize: bool = False,
    gt_csv: str | Path | None = None,
) -> None:
    """Run SAM2 on all PNGs in ``image_dir`` and save compressed masks."""

    build_sam2, automatic_mask_generator = _load_sam2(sam2_repo)
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {device}")

    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    mask_dir = output_dir / "masks"
    overlay_dir = output_dir / "overlays"
    mask_dir.mkdir(parents=True, exist_ok=True)
    if visualize:
        overlay_dir.mkdir(parents=True, exist_ok=True)

    ground_truth = None
    if gt_csv:
        ground_truth = load_ground_truth(gt_csv)
        images = find_images(image_dir, ground_truth)
    else:
        supported = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
        images = {
            path.stem: str(path)
            for path in sorted(image_dir.iterdir())
            if path.is_file() and path.suffix.lower() in supported
        }
    filenames = sorted(images)
    if max_images is not None:
        filenames = filenames[:max_images]
    if not filenames:
        raise ValueError(f"No images found in {image_dir}")

    print(f"Loading SAM2 from {checkpoint} on {torch_device}")
    model = build_sam2(
        config_name,
        str(checkpoint),
        device=torch_device,
        apply_postprocessing=False,
    )
    generator = automatic_mask_generator(
        model=model,
        points_per_side=DEFAULT_SAM2_CONFIG.points_per_side,
        points_per_batch=128,
        pred_iou_thresh=DEFAULT_SAM2_CONFIG.pred_iou_thresh,
        stability_score_thresh=DEFAULT_SAM2_CONFIG.stability_score_thresh,
        box_nms_thresh=DEFAULT_SAM2_CONFIG.box_nms_thresh,
        min_mask_region_area=DEFAULT_SAM2_CONFIG.min_mask_region_area,
    )

    rows: list[tuple[str, int, int]] = []
    started = time.perf_counter()
    try:
        with torch.inference_mode():
            for index, filename in enumerate(filenames, start=1):
                image_bgr = cv2.imread(images[filename])
                if image_bgr is None:
                    print(f"[WARN] Cannot read {images[filename]}; skipping")
                    continue
                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                image_rgb = PREPROCESS[preprocess_name](image_rgb)
                with _autocast(torch_device):
                    raw_masks = generator.generate(image_rgb)
                masks = postprocess_sam2_masks(
                    raw_masks, image_rgb, DEFAULT_SAM2_CONFIG
                )
                serializable = [dict(mask) for mask in masks]
                _save_masks(mask_dir / f"{filename}.json.gz", serializable)
                if visualize:
                    _save_overlay(
                        image_rgb, serializable, overlay_dir / f"{filename}.png"
                    )
                rows.append(
                    (
                        filename,
                        len(masks),
                        int(sum(int(mask["area"]) for mask in masks)),
                    )
                )
                if index % 50 == 0 or index == len(filenames):
                    elapsed = time.perf_counter() - started
                    print(f"Processed {index}/{len(filenames)} ({elapsed:.1f}s)")
                del image_bgr, image_rgb, raw_masks, masks
    finally:
        del generator, model
        if torch_device.type == "cuda":
            torch.cuda.empty_cache()

    counts_path = output_dir / "colony_counts.csv"
    with counts_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        header = ["filename", "colony_count", "colony_area"]
        if ground_truth is not None:
            header.append("ground_truth")
        writer.writerow(header)
        for filename, count, area in rows:
            row: list[Any] = [filename, count, area]
            if ground_truth is not None:
                row.append(ground_truth.get(filename, ""))
            writer.writerow(row)
    print(f"Masks saved to {mask_dir}; counts saved to {counts_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--output-dir", default=Path.cwd() / "sam2_output")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--gt-csv", default=None)
    parser.add_argument("--sam2-repo", default=os.environ.get("SAM2_REPO"))
    parser.add_argument("--config", default="sam2_hiera_t.yaml")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--preprocess", choices=sorted(PREPROCESS), default="none")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()

    run_pipeline(
        args.image_dir,
        args.output_dir,
        args.checkpoint,
        sam2_repo=args.sam2_repo,
        config_name=args.config,
        device=args.device or f"cuda:{args.gpu}",
        preprocess_name=args.preprocess,
        max_images=args.max_images,
        visualize=args.visualize,
        gt_csv=args.gt_csv,
    )


if __name__ == "__main__":
    main()
