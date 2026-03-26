from typing import Any

import torch
from omegaconf import DictConfig

from src.models.base_model.base_model import BaseModel
from src.models.dinov3_model.core.dinov3 import DINOv3
from src.utils.losses import build_loss


class DINOv3Model(BaseModel):
    """Implementation for DINOv3."""

    def __init__(self, cfg: DictConfig) -> None:
        """
        Initialize the model and the loss criterion.

        Args:
            cfg: Configuration object containing model and training hyperparameters.
        """
        super().__init__(cfg)
        self.criterion = build_loss(cfg)

    def build_model(self) -> None:
        """Instantiate the DINOv3 core architecture."""
        self.model = DINOv3(
            return_features=False,
            version=self.cfg.backbone_version,
            freeze_backbone=self.cfg.freeze_backbone,
            classification=self.classification,
            height=self.height,
            num_classes=self.cfg.n_classes,
            avg_pool=self.cfg.avg_pool,
            use_sigmoid_for_height=self.cfg.use_sigmoid_for_height,
            max_height=self.cfg.max_height,
        )

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

            if name.startswith("backbone"):
                lr = self.cfg.lr_head / self.cfg.lr_reduction_backbone
            elif name.startswith("head"):
                lr = self.cfg.lr_head
            else:
                raise ValueError(f"Unknown parameter name: {name}")

            param_groups.append({"params": [param], "lr": lr})

        return param_groups

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

    def forward(
        self, batch: tuple[list[Any], torch.Tensor, torch.Tensor | None, dict[str, torch.Tensor]]
    ) -> tuple[dict[str, torch.Tensor | None], torch.Tensor]:
        """
        Perform the forward pass and compute the loss.

        Args:
            batch: A tuple containing (meta, image, dsm, targets).

        Returns:
            A tuple containing a dictionary of predictions and the scalar loss.
        """
        meta, image, _, targets = batch

        class_logits, height_preds = self.model(image)

        predictions: dict[str, torch.Tensor | None] = {}
        preds_dict: dict[str, torch.Tensor | None] = {}
        targets_dict: dict[str, torch.Tensor] = {}

        if self.classification:
            predictions["class"] = class_logits.argmax(dim=1)
            preds_dict["cls"] = class_logits
            targets_dict["cls"] = targets["class"]

        if self.height:
            predictions["height"] = height_preds
            preds_dict["height"] = height_preds
            targets_dict["height"] = targets["height"]

        loss = self.criterion(preds_dict, targets_dict)

        return predictions, loss
