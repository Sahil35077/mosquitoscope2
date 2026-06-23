"""Load UI metadata from bundled JSON without loading PyTorch."""

import json
from functools import lru_cache

from webapp.config import TEST_METRICS_PATH


@lru_cache(maxsize=1)
def get_ui_context() -> dict:
    with open(TEST_METRICS_PATH) as f:
        metrics = json.load(f)

    per_class = metrics.get("metrics", {}).get("per_class", {})
    return {
        "classes": sorted(per_class.keys()),
        "model_metrics": {
            "test_accuracy": metrics.get("test_accuracy"),
            "test_f1_macro": metrics.get("metrics", {}).get("f1_macro"),
            "test_f1_weighted": metrics.get("metrics", {}).get("f1_weighted"),
            "cv_mean_accuracy": metrics.get("cv_mean_accuracy"),
            "model_name": metrics.get("model_name", "convnextv2_tiny"),
            "image_size": 288,
            "train_samples": metrics.get("train_samples"),
            "test_samples": metrics.get("test_samples"),
            "class_weight_method": metrics.get("class_weight_method"),
            "training_recipe": metrics.get("training_recipe"),
        },
    }
