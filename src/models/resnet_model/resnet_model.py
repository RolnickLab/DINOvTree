from typing import Any

import torch
from omegaconf import DictConfig

from src.models.base_model.base_model import BaseModel
from src.models.resnet_model.core.resnet import ResNet
from src.utils.losses import build_loss


class ResNetModel(BaseModel):
    """A Lightning-based wrapper for ResNet architectures."""

    def __init__(self, cfg: DictConfig) -> None:
        """
        Initializes the model wrapper and the criterion.

        Args:
            cfg: Configuration object containing 'n_classes', 'backbone',
                'lr_head', and 'weight_decay'.
        """
        super().__init__(cfg)
        self.criterion = build_loss(cfg)

    def build_model(self) -> None:
        """Instantiates the core ResNet architecture with task-specific flags."""
        self.model = ResNet(
            num_classes=self.cfg.n_classes,
            backbone=self.cfg.backbone,
            classification=self.classification,
            height=self.height,
            use_dsm=("dsm" in self.modalities),
            use_sigmoid_for_height=self.cfg.use_sigmoid_for_height,
            max_height=self.cfg.max_height,
        )

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

            if name.startswith("backbone"):
                lr = self.cfg.lr_head / self.cfg.lr_reduction_backbone
            elif name.startswith("head"):
                lr = self.cfg.lr_head
            else:
                raise ValueError(f"Unknown parameter name: {name}")

            param_groups.append({"params": [param], "lr": lr})

        return param_groups

    def forward(
        self, batch: tuple[list[Any], torch.Tensor, torch.Tensor | None, dict[str, torch.Tensor]]
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """
        Performs the forward pass, processes logits into predictions, and computes multi-task losses.

        Args:
            batch: A tuple containing:
                - meta: List of spatial metadata.
                - image: Input RGB tensor (B, 3, H, W).
                - dsm: Optional DSM tensor (B, 1, H, W).
                - targets: Dict containing ground truth 'class' and 'height'.

        Returns:
            A tuple of (predictions_dict, scalar_loss).
        """
        _, image, dsm, targets = batch

        class_logits, height_preds = self.model(image, dsm)

        predictions: dict[str, torch.Tensor] = {}
        preds_dict: dict[str, torch.Tensor] = {}
        targets_dict: dict[str, torch.Tensor] = {}

        tasks = [
            (self.classification, class_logits, "cls", "class", "class", lambda x: x.argmax(dim=1)),
            (self.height, height_preds, "height", "height", "height", lambda x: x),
        ]

        for active, raw_output, loss_key, tgt_key, pred_key, process_fn in tasks:
            if active:
                predictions[pred_key] = process_fn(raw_output)
                preds_dict[loss_key] = raw_output
                target_tensor = targets.get(tgt_key)
                if target_tensor is None:
                    raise ValueError(f"Task '{pred_key}' is active, but '{tgt_key}' is missing from batch targets.")
                targets_dict[loss_key] = target_tensor

        loss = self.criterion(preds_dict, targets_dict)

        return predictions, loss
