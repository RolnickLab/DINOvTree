from typing import Any

import torch
from omegaconf import DictConfig

from src.models.base_model.base_model import BaseModel
from src.models.dinovtree_model.core.dinovtree import DINOvTree
from src.utils.losses import build_loss


class DINOvTreeModel(BaseModel):
    """
    Multi-task model implementation for DINOvTree.

    Handles hierarchical feature extraction for classification and height estimation tasks. It employs specific learning
    rate scaling for the backbone versus the task heads.
    """

    def __init__(self, cfg: DictConfig) -> None:
        """
        Initialize the model and the loss criterion.

        Args:
            cfg: Configuration object containing model and training hyperparameters.
        """
        super().__init__(cfg)
        self.criterion = build_loss(cfg)

    def build_model(self) -> None:
        """Instantiate the DINOvTree core architecture."""
        self.model = DINOvTree(cfg=self.cfg)

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
            elif name.startswith("task_heads"):
                lr = self.cfg.lr_head
            else:
                raise ValueError(f"Unknown parameter name: {name}")

            param_groups.append({"params": [param], "lr": lr})

        for param in self.criterion.parameters():
            if param.requires_grad:
                param_groups.append({"params": [param], "lr": self.cfg.lr_head})

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
        Perform the forward pass and compute the multi-task loss.

        Args:
            batch: A tuple containing (meta, image, dsm, targets).

        Returns:
            A tuple containing a dictionary of predictions and the scalar loss.
        """
        meta, image, _, targets = batch

        class_logits, height_preds = self.model(image, meta)

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
