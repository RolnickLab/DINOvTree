import json
from pathlib import Path
from typing import Dict, Final, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig


class ClassBalancedFocalLoss(nn.Module):
    """
    Class-Balanced Focal Loss for single-label multi-class classification.

    This loss addresses class imbalance by weighting the Focal Loss by the inverse
    of the "effective number" of samples per class, based on the theory that
    as the number of samples increases, the additional benefit of a new sample
    diminishes exponentially.

    Reference:
        Cui et al., "Class-Balanced Loss Based on Effective Number of Samples,"
        CVPR 2019.
    """

    weights: torch.Tensor

    def __init__(
        self,
        num_classes: int,
        stats_path: Path,
        beta: float = 0.9999,
        gamma: float = 2.0,
    ) -> None:
        """
        Initializes the Class-Balanced Focal Loss.

        Args:
            num_classes: Total number of categories.
            stats_path: Path to a JSON file containing class counts in 'train' key.
            beta: Hyperparameter for calculating effective number of samples.
            gamma: Focusing parameter for Focal Loss to down-weight easy examples.
        """
        super().__init__()
        self.num_classes: Final[int] = num_classes
        self.beta: Final[float] = beta
        self.gamma: Final[float] = gamma
        stats_path = Path(stats_path)

        stats_path = Path(stats_path)
        if not stats_path.exists():
            raise FileNotFoundError(f"Statistics file not found at: {stats_path}")

        with open(stats_path, "r") as f:
            stats = json.load(f)

        if "train" not in stats:
            raise ValueError("The stats JSON must contain a 'train' key with class counts.")

        counts_dict = stats["train"]

        samples_per_class = torch.tensor(
            [counts_dict.get(str(i + 1), 0) for i in range(num_classes)], dtype=torch.float
        )

        effective_num = 1.0 - torch.pow(self.beta, samples_per_class)
        weights = (1.0 - self.beta) / (effective_num + 1e-12)

        weights = weights / weights.sum() * num_classes

        self.register_buffer("weights", weights)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Computes the weighted focal loss.

        Args:
            logits: Predicted raw outputs of shape [batch_size, num_classes].
            target: Ground truth 0-indexed labels of shape [batch_size].

        Returns:
            A scalar tensor representing the balanced focal loss.
        """
        target_one_hot = F.one_hot(target, num_classes=self.num_classes).float()

        cb_weights = self.weights.gather(0, target).view(-1, 1)

        focal_weights = torch.exp(
            -self.gamma * target_one_hot * logits - self.gamma * torch.log(1 + torch.exp(-logits))
        )

        bce_loss = F.binary_cross_entropy_with_logits(logits, target_one_hot, reduction="none")

        loss = cb_weights * focal_weights * bce_loss

        return loss.sum() / target_one_hot.sum()


class MaskedSmoothL1Loss(nn.Module):
    """Smooth L1 Loss (Huber Loss) that ignores NaN values in the target."""

    def __init__(self) -> None:
        """Initializes the Masked Smooth L1 Loss."""
        super().__init__()
        self.loss_fn = nn.SmoothL1Loss()

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Computes the loss only on valid (non-NaN) target entries.

        Args:
            preds: Predicted values of shape (N, ...).
            target: Target values of same shape, potentially containing NaNs.

        Returns:
            The computed loss scalar or tensor.
        """
        mask = ~torch.isnan(target)

        if not mask.any():
            return torch.zeros((), device=preds.device, dtype=preds.dtype)

        return self.loss_fn(preds[mask], target[mask])


class MultiTaskLoss(nn.Module):
    """
    Automated Multi-Task Weighting Loss Module.

    Supports three strategies for balancing task-specific losses:
    1. Static: Fixed weights defined at initialization.
    2. Uncertainty: Learnable parameters based on homoscedastic uncertainty (Kendall et al., 2018).
    3. DWA: Dynamic Weight Averaging based on loss change rates (Liu et al., 2019).
    """

    def __init__(
        self,
        cls_loss: nn.Module | None,
        height_loss: nn.Module | None,
        weighting_strategy: str = "static",
        dwa_temp: float = 2.0,
        cls_weight: float = 1.0,
        height_weight: float = 1.0,
    ) -> None:
        """
        Initializes the multi-task loss container.

        Args:
            cls_loss: Loss module for classification task.
            height_loss: Loss module for height regression task.
            weighting_strategy: One of 'static', 'uncertainty', or 'dwa'.
            dwa_temp: Temperature coefficient for DWA (higher = softer weights).
            cls_weight: Starting weight for classification.
            height_weight: Starting weight for height regression.
        """
        super().__init__()
        self.losses = nn.ModuleDict()
        if cls_loss:
            self.losses["cls"] = cls_loss
        if height_loss:
            self.losses["height"] = height_loss

        self.task_names: Final[List[str]] = list(self.losses.keys())
        self.weighting_strategy: Final[str] = weighting_strategy

        self.static_weights: Final[Dict[str, float]] = {"cls": cls_weight, "height": height_weight}

        if self.weighting_strategy == "uncertainty":
            self.log_vars = nn.ParameterDict({task: nn.Parameter(torch.zeros(1)) for task in self.task_names})

        self.dwa_temp: Final[float] = dwa_temp
        if self.weighting_strategy == "dwa":
            self.dwa_weights: Dict[str, float] = {t: 1.0 for t in self.task_names}
            self.epoch_loss_history: Dict[str, List[float]] = {t: [] for t in self.task_names}
            self.current_epoch_accumulator: Dict[str, float] = {t: 0.0 for t in self.task_names}
            self.num_batches_tracked: int = 0

    def forward(
        self,
        preds: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Computes the weighted sum of all active task losses.

        Args:
            preds: Dictionary mapping task names to model outputs.
            targets: Dictionary mapping task names to ground truth labels.

        Returns:
            A scalar tensor representing the total multi-task loss.
        """
        device = next(iter(preds.values())).device
        total_loss = torch.tensor(0.0, device=device)

        for task_name in self.task_names:
            if task_name not in preds or task_name not in targets:
                continue

            raw_loss = self.losses[task_name](preds[task_name], targets[task_name])

            if self.weighting_strategy == "static":
                weighted_loss = self.static_weights[task_name] * raw_loss

            elif self.weighting_strategy == "uncertainty":
                s = self.log_vars[task_name]
                factor = 0.5 if task_name == "height" else 1.0
                weighted_loss = factor * torch.exp(-s) * raw_loss + 0.5 * s

            elif self.weighting_strategy == "dwa":
                if self.training:
                    self.current_epoch_accumulator[task_name] += raw_loss.detach().item()
                weight = self.dwa_weights[task_name]
                weighted_loss = weight * raw_loss

            else:
                raise ValueError(f"Unknown weighting strategy: {self.weighting_strategy}")

            total_loss += weighted_loss

        if self.weighting_strategy == "dwa" and self.training:
            self.num_batches_tracked += 1

        return total_loss

    def step_epoch(self) -> None:
        """
        Updates DWA weights based on loss change rates from previous epochs.

        Should be called at the end of every training epoch.
        """
        if self.weighting_strategy != "dwa" or self.num_batches_tracked == 0:
            return

        for task_name in self.task_names:
            avg_loss = self.current_epoch_accumulator[task_name] / self.num_batches_tracked
            self.epoch_loss_history[task_name].append(avg_loss)
            if len(self.epoch_loss_history[task_name]) > 2:
                self.epoch_loss_history[task_name].pop(0)

            self.current_epoch_accumulator[task_name] = 0.0

        if all(len(self.epoch_loss_history[task_name]) >= 2 for task_name in self.task_names):
            w_t = {}
            for task_name in self.task_names:
                l_t_minus_1 = self.epoch_loss_history[task_name][-1]
                l_t_minus_2 = self.epoch_loss_history[task_name][-2]
                w_t[task_name] = l_t_minus_1 / (l_t_minus_2 + 1e-12)

            exp_w = {
                task_name: torch.exp(torch.tensor(w_t[task_name] / self.dwa_temp)) for task_name in self.task_names
            }
            sum_exp_w = sum(exp_w.values())

            for task_name in self.task_names:
                self.dwa_weights[task_name] = (len(self.task_names) * exp_w[task_name] / sum_exp_w).item()

        self.num_batches_tracked = 0


def build_cls_loss(cfg: DictConfig) -> nn.Module:
    """
    Factory function to build classification loss functions.

    Args:
        cfg: Configuration object containing 'cls_loss_name' and
            task-specific hyperparameters (beta, gamma, label_smoothing).

    Returns:
        A PyTorch Module for classification loss.

    Raises:
        ValueError: If the specified loss name is not supported.
    """
    if cfg.cls_loss_name == "cross_entropy":
        return nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

    elif cfg.cls_loss_name == "cb_focal":
        return ClassBalancedFocalLoss(cfg.n_classes, cfg.stats_id_path, beta=cfg.beta, gamma=cfg.gamma)

    else:
        raise ValueError(f"Unsupported classification loss: {cfg.cls_loss_name}")


def build_height_loss(cfg: DictConfig) -> nn.Module:
    """
    Factory function to build regression loss for height estimation.

    Args:
        cfg: Configuration object containing 'height_loss_name'.

    Returns:
        A PyTorch Module for regression loss.
    """
    if cfg.height_loss_name == "smooth_l1":
        return MaskedSmoothL1Loss()

    else:
        raise ValueError(f"Unsupported height loss: {cfg.height_loss_name}")


def build_loss(cfg: DictConfig) -> MultiTaskLoss:
    """
    Factory function to assemble the total MultiTaskLoss.

    This function determines which tasks are active, builds their individual
    losses, and wraps them in a MultiTaskLoss container with the chosen
    weighting strategy (static, uncertainty, or DWA).

    Args:
        cfg: Global configuration object containing 'tasks' list and weights.

    Returns:
        An initialized MultiTaskLoss module.
    """
    cls_loss: Optional[nn.Module] = None
    height_loss: Optional[nn.Module] = None

    cls_weight = cfg.get("cls_weight", 1.0)
    height_weight = cfg.get("height_weight", 1.0)

    if "classification" in cfg.tasks:
        cls_loss = build_cls_loss(cfg)
    else:
        cls_weight = 0.0

    if "height" in cfg.tasks:
        height_loss = build_height_loss(cfg)
    else:
        height_weight = 0.0

    total_w = cls_weight + height_weight
    if total_w > 0:
        cls_weight /= total_w
        height_weight /= total_w

    return MultiTaskLoss(
        cls_loss=cls_loss,
        height_loss=height_loss,
        weighting_strategy=cfg.weighting_strategy,
        cls_weight=cls_weight,
        height_weight=height_weight,
    )
