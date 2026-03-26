from typing import Final, cast

import torch
from torch import nn
from torchmetrics import Metric, MetricCollection


class DeltaAccuracy(Metric):
    """Threshold error accuracy: percent of samples with max(ŷ/y, y/ŷ) < δ."""

    full_state_update: bool = False

    def __init__(self, delta=1.25, eps=1e-6):
        """
        Initialize Delta Accuracy.

        Args:
            delta: The threshold ratio.
            eps: Small constant for numerical stability.
        """
        super().__init__()
        self.delta: Final[float] = delta
        self.eps: Final[float] = eps

        self.add_state("correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

        self.correct: torch.Tensor
        self.total: torch.Tensor

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """
        Update the running state with batch results.

        Args:
            preds: Predicted values tensor of shape (N,).
            target: Ground truth values tensor of shape (N,).
        """
        preds = preds + self.eps
        target = target + self.eps

        ratio = torch.maximum(preds / target, target / preds)
        correct_mask = (ratio < self.delta).float()

        self.correct += correct_mask.sum()
        self.total += target.numel()

    def compute(self) -> torch.Tensor:
        """Compute the final delta accuracy."""
        return self.correct / self.total


class ClasswiseRegressionWrapper(nn.Module):
    """Wraps a MetricCollection to calculate the same metrics separately for each class."""

    def __init__(self, base_collection: MetricCollection, category_dict: dict[int, str]) -> None:
        """
        Initialize the class-wise wrapper.

        Args:
            base_collection: The collection of regression metrics to clone.
            category_dict: Mapping of class IDs to class names.
        """
        super().__init__()
        self.category_dict: Final[dict[int, str]] = category_dict

        self.class_metrics = nn.ModuleDict(
            {str(class_id): base_collection.clone() for class_id in category_dict.keys()}
        )

    def update(self, preds: torch.Tensor, target: torch.Tensor, labels: torch.Tensor) -> None:
        """
        Update metrics for each class found in labels.

        Args:
            preds: Model predictions of shape (N,).
            target: Ground truth of shape (N,).
            labels: Class IDs for each sample of shape (N,).
        """
        for class_id in torch.unique(labels):
            cid_str = str(int(class_id.item()))

            if cid_str not in self.class_metrics:
                continue

            metric_collection = cast(MetricCollection, self.class_metrics[cid_str])

            mask = labels == class_id
            metric_collection.update(preds[mask], target[mask])

    def compute(self) -> dict[str, torch.Tensor]:
        """
        Compute metrics for all classes and return with descriptive keys.

        Returns:
            A dictionary mapping formatted metric names to their calculated
            tensors. Keys follow the pattern:
            '{metric_name}_per_class_{class_name}'.
        """
        results: dict[str, torch.Tensor] = {}
        for class_id_str, metrics in self.class_metrics.items():
            class_name = self.category_dict[int(class_id_str)]
            metric_collection = cast(MetricCollection, metrics)
            class_results = metric_collection.compute()

            for metric_name, value in class_results.items():
                new_key = f"{metric_name}_per_class_{class_name}"
                results[new_key] = value

        return results

    def reset(self) -> None:
        """Reset all internal metric states."""
        for metrics in self.class_metrics.values():
            metric_collection = cast(MetricCollection, metrics)
            metric_collection.reset()
