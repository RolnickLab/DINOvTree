from typing import Callable, cast

import torch
import torch.nn as nn
from omegaconf import DictConfig

from src.models.dinovtree_model.core.task_heads import TaskHeads


class DINOvTree(nn.Module):
    """
    Core architecture for DINOvTree.

    Integrates a Vision Transformer backbone (e.g., DINOv3) with multi-task heads.
    """

    def __init__(self, cfg: DictConfig) -> None:
        """
        Initialize the DINOvTree model.

        Args:
            cfg: Configuration object containing backbone and head parameters.
        """
        super().__init__()
        self.backbone = self._build_backbone(cfg)
        self.task_heads = self._build_task_heads(cfg)

    def _build_backbone(self, cfg: DictConfig) -> nn.Module:
        """
        Logic for selecting and configuring the backbone.

        Args:
            cfg: Configuration object.

        Returns:
            The instantiated backbone module.

        Raises:
            ValueError: If the backbone name is not supported.
        """
        if cfg.backbone_name == "dinov3":
            from src.models.dinov3_model.core.dinov3 import DINOv3

            return DINOv3(
                return_features=True,
                version=cfg.backbone_version,
                freeze_backbone=False,
            )

        else:
            raise ValueError(f"Unsupported backbone: {cfg.backbone_name}")

    def _build_task_heads(self, cfg: DictConfig) -> nn.Module:
        """
        Initialize the multi-task heads based on backbone output dimensions.

        Args:
            cfg: Configuration object.

        Returns:
            An instantiated TaskHeads module.
        """
        attr = getattr(self.backbone, "get_output_dim", None)
        if not callable(attr):
            raise AttributeError(
                f"Backbone of type {type(self.backbone).__name__} " "must implement 'get_output_dim() -> int'"
            )
        get_dim_fn = cast(Callable[[], int], attr)
        backbone_dim: int = get_dim_fn()

        return TaskHeads(
            cfg,
            backbone_dim,
        )

    def forward(
        self, img: torch.Tensor, meta: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """
        Perform a forward pass through the backbone and task heads.

        Args:
            img: Input image tensor of shape (B, C, H, W).
            meta: Dictionary containing metadata tensors for the task heads.

        Returns:
            A tuple containing:
                - class_logits: Predicted class probabilities/logits (B, num_classes) or None.
                - height_preds: Predicted height values (B, 1) or (B, H, W) or None.
        """
        features: torch.Tensor = self.backbone(img)

        class_logits, height_preds = self.task_heads(features)

        return class_logits, height_preds
