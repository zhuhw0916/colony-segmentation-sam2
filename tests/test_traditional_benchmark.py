import cv2
import numpy as np

from colony_segmentation.traditional_benchmark import METHODS, compute_metrics


def test_traditional_method_registry_is_complete():
    expected = {
        "Otsu+CC",
        "AdaptiveGauss",
        "Watershed",
        "DOG_Blob",
        "LoG_Blob",
        "Morphology",
        "HSV_Color",
    }

    assert set(METHODS) == expected


def test_traditional_methods_return_counts():
    image = np.full((128, 128, 3), 220, dtype=np.uint8)
    cv2.circle(image, (40, 64), 10, (80, 80, 80), -1)
    cv2.circle(image, (88, 64), 10, (80, 80, 80), -1)

    predictions = [method(image) for method in METHODS.values()]

    assert all(isinstance(prediction, int) for prediction in predictions)
    assert all(prediction >= 0 for prediction in predictions)


def test_traditional_metrics_are_finite():
    metrics = compute_metrics([0, 10, 20], [1, 11, 19])

    assert all(np.isfinite(value) for value in metrics.values())
