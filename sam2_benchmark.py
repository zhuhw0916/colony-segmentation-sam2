"""SAM2 automatic-mask benchmark for microbial colony counting."""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import cv2
import torch

try:  # Supports both ``python file.py`` and ``python -m package.module``.
    from .sam2_common import (
        DEFAULT_SAM2_CONFIG,
        PREPROCESS,
        Sam2PostprocessConfig,
        compute_metrics,
        find_images,
        load_ground_truth,
        postprocess_sam2_masks,
    )
except ImportError:  # pragma: no cover - used by direct script execution.
    from sam2_common import (  # type: ignore[no-redef]
        DEFAULT_SAM2_CONFIG,
        PREPROCESS,
        Sam2PostprocessConfig,
        compute_metrics,
        find_images,
        load_ground_truth,
        postprocess_sam2_masks,
    )


def _load_sam2(sam2_repo: str | Path | None):
    """Import SAM2 lazily so traditional methods do not require torch/SAM2."""

    if sam2_repo:
        repo = str(Path(sam2_repo).expanduser().resolve())
        if repo not in sys.path:
            sys.path.insert(0, repo)
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.build_sam import build_sam2

    return build_sam2, SAM2AutomaticMaskGenerator


def _autocast(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def run_sam2(
    checkpoint: str | Path,
    model_name: str,
    images: dict[str, str],
    ground_truth: dict[str, int],
    *,
    sam2_repo: str | Path | None,
    config_name: str,
    device: str,
    max_images: int | None = None,
    preprocess_name: str = "none",
    postprocess_config: Sam2PostprocessConfig = DEFAULT_SAM2_CONFIG,
) -> tuple[list[str], list[int], list[int]]:
    """Run one SAM2 checkpoint and return names, labels, and predictions."""

    build_sam2, automatic_mask_generator = _load_sam2(sam2_repo)
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {device}")

    print(f"Loading {model_name} from {checkpoint} on {torch_device}...")
    model = build_sam2(
        config_name,
        str(checkpoint),
        device=torch_device,
        apply_postprocessing=False,
    )
    mask_generator = automatic_mask_generator(
        model=model,
        points_per_side=postprocess_config.points_per_side,
        points_per_batch=128,
        pred_iou_thresh=postprocess_config.pred_iou_thresh,
        stability_score_thresh=postprocess_config.stability_score_thresh,
        box_nms_thresh=postprocess_config.box_nms_thresh,
        min_mask_region_area=postprocess_config.min_mask_region_area,
    )

    filenames = sorted(images)
    if max_images is not None:
        filenames = filenames[:max_images]
    preprocess = PREPROCESS[preprocess_name]
    processed_names: list[str] = []
    labels: list[int] = []
    predictions: list[int] = []
    started = time.perf_counter()

    try:
        with torch.inference_mode():
            for index, filename in enumerate(filenames, start=1):
                image_bgr = cv2.imread(images[filename])
                if image_bgr is None:
                    print(f"[WARN] Cannot read {images[filename]}; skipping")
                    continue
                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                image_rgb = preprocess(image_rgb)
                with _autocast(torch_device):
                    raw_masks = mask_generator.generate(image_rgb)
                filtered_masks = postprocess_sam2_masks(
                    raw_masks, image_rgb, postprocess_config
                )
                processed_names.append(filename)
                labels.append(ground_truth[filename])
                predictions.append(len(filtered_masks))

                if index % 100 == 0 or index == len(filenames):
                    elapsed = time.perf_counter() - started
                    rate = index / elapsed if elapsed else 0.0
                    print(
                        f"{model_name}: {index}/{len(filenames)} "
                        f"({rate:.2f} images/s)"
                    )
                del image_bgr, image_rgb, raw_masks, filtered_masks
    finally:
        del mask_generator, model
        if torch_device.type == "cuda":
            torch.cuda.empty_cache()

    return processed_names, labels, predictions


def _checkpoint_for_model(args: argparse.Namespace, model: str) -> str:
    checkpoint = {
        "trained": args.trained_checkpoint,
        "base": args.base_checkpoint,
    }[model]
    if not checkpoint:
        option = "--trained-checkpoint" if model == "trained" else "--base-checkpoint"
        raise ValueError(f"{option} is required for --model {model}")
    return checkpoint


def _write_results(
    output_csv: str | Path,
    filenames: list[str],
    labels: list[int],
    result_columns: dict[str, dict[str, int]],
) -> None:
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    label_by_name = dict(zip(filenames, labels))
    with output_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        names = list(result_columns)
        writer.writerow(["filename", "ground_truth", *names])
        for filename in filenames:
            writer.writerow(
                [
                    filename,
                    label_by_name[filename],
                    *(result_columns[name].get(filename, 0) for name in names),
                ]
            )


def _default_paths() -> tuple[Path, Path, Path]:
    project_root = Path(__file__).resolve().parents[1]
    return (
        project_root / "all_pic",
        project_root / "data" / "merged.csv",
        project_root / "results" / "benchmark_sam2.csv",
    )


def main() -> None:
    default_image_dir, default_gt_csv, default_output = _default_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", default=default_image_dir)
    parser.add_argument("--gt-csv", default=default_gt_csv)
    parser.add_argument("--output", default=default_output)
    parser.add_argument("--sam2-repo", default=os.environ.get("SAM2_REPO"))
    parser.add_argument(
        "--config",
        default=os.environ.get("SAM2_CONFIG", "sam2_hiera_t.yaml"),
        help="SAM2 Hydra config name, for example sam2_hiera_t.yaml",
    )
    parser.add_argument(
        "--trained-checkpoint", default=os.environ.get("SAM2_TRAINED_CHECKPOINT")
    )
    parser.add_argument(
        "--base-checkpoint", default=os.environ.get("SAM2_BASE_CHECKPOINT")
    )
    parser.add_argument("--model", choices=("trained", "base", "both"), default="both")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--device", default=None, help="Overrides --gpu, e.g. cpu or cuda:1"
    )
    parser.add_argument("--preprocess", choices=sorted(PREPROCESS), default="none")
    parser.add_argument("--max-images", type=int, default=None)
    args = parser.parse_args()

    device = args.device or f"cuda:{args.gpu}"
    ground_truth = load_ground_truth(args.gt_csv)
    images = find_images(args.image_dir, ground_truth)
    if not images:
        raise SystemExit("No labelled PNG images were found")
    print(f"Ground-truth entries: {len(ground_truth)}; images found: {len(images)}")

    selected_models = ("trained", "base") if args.model == "both" else (args.model,)
    columns: dict[str, dict[str, int]] = {}
    common_filenames: list[str] | None = None
    common_labels: list[int] | None = None
    for model in selected_models:
        checkpoint = _checkpoint_for_model(args, model)
        names, labels, predictions = run_sam2(
            checkpoint,
            f"SAM2_{model.title()}",
            images,
            ground_truth,
            sam2_repo=args.sam2_repo,
            config_name=args.config,
            device=device,
            max_images=args.max_images,
            preprocess_name=args.preprocess,
        )
        metrics = compute_metrics(labels, predictions)
        column_name = f"SAM2_{model.title()}"
        columns[column_name] = dict(zip(names, predictions))
        if common_filenames is None:
            common_filenames, common_labels = names, labels
        print(
            f"{column_name}: MAE={metrics['MAE']:.2f}, "
            f"RMSE={metrics['RMSE']:.2f}, R²={metrics['R²']:.4f}, "
            f"Pearson={metrics['Pearson_r']:.4f}"
        )

    assert common_filenames is not None and common_labels is not None
    _write_results(args.output, common_filenames, common_labels, columns)
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
