from typing import Any, cast

import torch
from omegaconf import DictConfig

from src.models.base_model.base_model import BaseModel
from src.models.maskrcnn_model.core.maskrcnn import (
    MaskRCNN,
)


class MaskRCNNModel(BaseModel):
    """A PyTorch Lightning-compatible wrapper for the Mask R-CNN architecture."""

    def __init__(self, cfg: DictConfig) -> None:
        """
        Initializes the model wrapper.

        Args:
            cfg: Configuration object containing hyperparameters like
                'lr_head', 'lr_reduction_backbone', and 'weight_decay'.
        """
        super().__init__(cfg)

    def build_model(self) -> None:
        """Instantiates the core MaskRCNN module."""
        self.model = MaskRCNN(cfg=self.cfg)

    def configure_optimizers(self) -> tuple[list[torch.optim.Optimizer], list[dict[str, Any]]]:
        """
        Configure the AdamW optimizer and cosine learning rate scheduler.

        Returns:
            A tuple containing a list of optimizers and a list of scheduler configs.
        """
        param_groups = self._get_parameter_groups()

        optimizer = torch.optim.AdamW(
            param_groups,
            betas=(0.9, 0.999),
            weight_decay=self.cfg.weight_decay,
        )

        scheduler = self.build_cosine_scheduler(
            optimizer=optimizer,
        )

        return [optimizer], [scheduler]

    def _get_parameter_groups(self) -> list[dict[str, list[torch.Tensor] | float]]:
        """
        Categorize parameters into groups for differential learning rates.

        Returns:
            A list of parameter groups for the optimizer.
        """
        param_groups: list[dict[str, list[torch.Tensor] | float]] = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            if name.startswith("model.backbone"):
                lr = self.cfg.lr_head / self.cfg.lr_reduction_backbone
            elif name.startswith("model.rpn") or name.startswith("model.roi_heads"):
                lr = self.cfg.lr_head
            else:
                raise ValueError(f"Unknown parameter name: {name}")

            param_groups.append({"params": [param], "lr": lr})

        return param_groups

    def forward(
        self, batch: tuple[list[Any], torch.Tensor, torch.Tensor | None, dict[str, torch.Tensor]]
    ) -> tuple[dict[str, torch.Tensor | None], torch.Tensor]:
        """
        Performs the forward pass, extracting tree predictions and calculating the aggregated multi-task loss during
        training.

        Args:
            batch: A tuple containing:
                - meta: List of dictionaries containing metadata.
                - image: RGB image tensor of shape (B, 3, H, W).
                - dsm: Optional Digital Surface Model tensor of shape (B, 1, H, W).
                - targets: Dict containing 'class', 'segmentation', and 'height' labels.

        Returns:
            A tuple containing:
                - predictions: A dict mapping task names to prediction tensors.
                - loss: A scalar tensor representing the total multi-task loss.
        """
        _, image, dsm, targets = batch
        losses, class_preds, mask_preds, height_preds = self.model(image, dsm, targets, return_loss_list=self.training)

        predictions: dict[str, torch.Tensor | None] = {}

        if self.classification:
            predictions["class"] = class_preds
        if self.segmentation:
            predictions["segmentation"] = mask_preds
        if self.height:
            predictions["height"] = height_preds

        total_loss = torch.tensor(0.0, device=image.device)
        if self.training:
            total_loss = sum(loss for loss in losses.values())
            total_loss = cast(torch.Tensor, total_loss)

        return predictions, total_loss
