import json
import os
from io import BytesIO

import timm
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from webapp.config import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    TEST_METRICS_PATH,
    TOP_K,
)
from webapp.download_assets import ensure_model
from webapp.memory_utils import configure_torch, is_low_memory_mode, release_memory
from webapp.metrics_loader import get_ui_context


class MosquitoClassifier:
    def __init__(self):
        configure_torch()
        force_cpu = os.environ.get("FORCE_CPU", "").lower() in ("1", "true", "yes")
        self.device = "cpu" if force_cpu else ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_tta = not is_low_memory_mode() and os.environ.get("DISABLE_TTA", "").lower() not in (
            "1",
            "true",
            "yes",
        )
        self.model = None
        self.classes = []
        self.label_to_idx = {}
        self.idx_to_label = {}
        self.image_size = 224
        self.transform = None
        self.model_metrics = {}
        self._per_class_metrics = None
        self._load()

    def _load(self):
        model_path = ensure_model()
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.classes = checkpoint["classes"]
        self.label_to_idx = checkpoint["label_to_idx"]
        self.idx_to_label = {int(k): v for k, v in checkpoint["idx_to_label"].items()}
        cfg = checkpoint.get("config", {})
        self.image_size = int(
            cfg.get("image_size") or checkpoint.get("image_size") or 224
        )

        model_name = checkpoint["model_name"]
        state_dict = checkpoint["model_state_dict"]
        del checkpoint
        release_memory()

        self.model = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=len(self.classes),
        )
        self.model.load_state_dict(state_dict)
        del state_dict
        release_memory()

        self.model.to(self.device)
        self.model.eval()

        resize_side = int(self.image_size * 1.14)
        self.transform = transforms.Compose([
            transforms.Resize(resize_side),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

        self.model_metrics = get_ui_context()["model_metrics"]

    def _get_per_class_metrics(self) -> dict:
        if self._per_class_metrics is None and TEST_METRICS_PATH.exists():
            with open(TEST_METRICS_PATH) as f:
                metrics = json.load(f)
            self._per_class_metrics = metrics.get("metrics", {}).get("per_class", {})
        return self._per_class_metrics or {}

    def predict(self, image_bytes: bytes, top_k: int = TOP_K) -> dict:
        with Image.open(BytesIO(image_bytes)) as image:
            image_rgb = image.convert("RGB")
            image_size = list(image_rgb.size)
            tensor = self.transform(image_rgb).unsqueeze(0)

        try:
            with torch.inference_mode():
                tensor = tensor.to(self.device)
                if self.use_tta:
                    flipped = torch.flip(tensor, dims=[3])
                    logits = (self.model(tensor) + self.model(flipped)) / 2.0
                    del flipped
                else:
                    logits = self.model(tensor)
                probs = F.softmax(logits, dim=1).cpu().numpy()[0]
                del tensor, logits
        finally:
            release_memory()

        order = probs.argsort()[::-1][:top_k]
        predictions = [
            {
                "class": self.idx_to_label[int(i)],
                "confidence": float(probs[i]),
                "confidence_pct": round(float(probs[i]) * 100, 2),
            }
            for i in order
        ]
        del probs

        return {
            "top_class": predictions[0]["class"],
            "top_confidence_pct": predictions[0]["confidence_pct"],
            "predictions": predictions,
            "image_size": image_size,
            "tta": self.use_tta,
        }

    def evaluate_ground_truth(self, predicted: str, ground_truth: str) -> dict:
        per_class = self._get_per_class_metrics()
        correct = predicted == ground_truth
        gt_metrics = per_class.get(ground_truth, {})
        pred_metrics = per_class.get(predicted, {})
        return {
            "ground_truth": ground_truth,
            "predicted": predicted,
            "correct": correct,
            "result": "Correct" if correct else "Incorrect",
            "ground_truth_test_metrics": {
                "precision": gt_metrics.get("precision"),
                "recall": gt_metrics.get("recall"),
                "f1": gt_metrics.get("f1"),
                "support": gt_metrics.get("support"),
            },
            "predicted_class_test_metrics": {
                "precision": pred_metrics.get("precision"),
                "recall": pred_metrics.get("recall"),
                "f1": pred_metrics.get("f1"),
                "support": pred_metrics.get("support"),
            },
        }


_classifier = None


def get_classifier() -> MosquitoClassifier:
    global _classifier
    if _classifier is None:
        _classifier = MosquitoClassifier()
    return _classifier
