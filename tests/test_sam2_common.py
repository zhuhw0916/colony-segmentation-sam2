import cv2
import numpy as np

from colony_segmentation.sam2_common import (
    PREPROCESS,
    compute_metrics,
    postprocess_sam2_masks,
)


def test_preprocessing_variants_preserve_image_shape():
    image = np.full((32, 48, 3), 120, dtype=np.uint8)
    for preprocess in PREPROCESS.values():
        output = preprocess(image)
        assert output.shape == image.shape
        assert output.dtype == image.dtype


def test_postprocess_keeps_a_valid_colony_mask():
    image = np.full((128, 128, 3), 100, dtype=np.uint8)
    cv2.circle(image, (64, 64), 12, (220, 220, 220), -1)
    segmentation = np.zeros((128, 128), dtype=bool)
    cv2.circle(segmentation.view(np.uint8), (64, 64), 10, 1, -1)
    masks = [
        {
            "segmentation": segmentation,
            "area": int(segmentation.sum()),
            "predicted_iou": 0.9,
            "bbox": [54, 54, 20, 20],
        }
    ]

    filtered = postprocess_sam2_masks(masks, cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    assert len(filtered) == 1


def test_compute_metrics_returns_expected_values():
    metrics = compute_metrics([1, 2, 3], [1, 3, 2])

    assert metrics["MAE"] == 2 / 3
    assert metrics["RMSE"] == np.sqrt(2 / 3)
    assert metrics["MedAE"] == 1
