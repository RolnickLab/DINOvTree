from typing import Any

import torch
from omegaconf import DictConfig

from src.models.allometric_model.core.allometric import Allometric
from src.models.base_model.base_model import BaseModel


class AllometricModel(BaseModel):
    """Allometric model wrapper that integrates the Allometric core logic into the project's BaseModel framework."""

    def __init__(self, cfg: DictConfig) -> None:
        """
        Initializes the model wrapper.

        Args:
            cfg: Hydra/OmegaConf configuration object containing model
                parameters such as 'radius_determination' and 'allometry_path'.
        """
        super().__init__(cfg)

    def build_model(self) -> None:
        """Instantiates the underlying Allometric core module using configuration parameters."""
        self.model = Allometric(
            radius_determination=self.cfg.radius_determination,
            allometry_path=self.cfg.allometry_path,
            dataset_name=self.cfg.dataset_name,
            category_dict=self.category_dict,
        )

    def configure_optimizers(self):
        pass

    def forward(
        self, batch: tuple[list[Any], torch.Tensor, torch.Tensor | None, dict[str, torch.Tensor]]
    ) -> tuple[dict[str, torch.Tensor | None], torch.Tensor]:
        """
        Performs the forward pass by extracting heights from tree crown geometry.

        Args:
            batch: A tuple containing four elements:
                1. meta: A list of dictionaries, each providing spatial 'resolution'.
                2. image: The input RGB image tensor (unused here).
                3. dsm: An optional Digital Surface Model tensor (unused here).
                4. targets: A dictionary containing 'class' IDs, 'segmentation'
                   masks, and 'height' labels.

        Returns:
            A tuple containing:
                - predictions: A dictionary where 'height' maps to the
                  calculated height tensor.
                - loss: A scalar tensor (0.0) as this model is
                  non-trainable/deterministic.
        """
        meta, _, _, targets = batch
        if not getattr(self, "height", True):
            raise ValueError("Height prediction is not enabled in the configuration.")
        height_preds = self.model(targets["class"], targets["segmentation"], meta)

        predictions: dict[str, torch.Tensor | None] = {}

        predictions["height"] = height_preds
        predictions["class"] = targets["class"]
        predictions["segmentation"] = targets["segmentation"]

        device = height_preds.device
        loss = torch.tensor(0.0, device=device)
        return predictions, loss
