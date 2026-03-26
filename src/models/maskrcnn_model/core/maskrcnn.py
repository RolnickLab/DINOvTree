from typing import cast

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torchvision.models.detection import (
    MaskRCNN_ResNet50_FPN_Weights,
    maskrcnn_resnet50_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.ops import masks_to_boxes


class MaskRCNN(nn.Module):
    """
    A Mask R-CNN model adapted for tree species classification or height estimation.

    Supports 4-channel input (RGB + DSM) and specialized post-processing to select the central object in an image tile.
    """

    def __init__(self, cfg: DictConfig) -> None:
        """
        Initializes the MaskRCNN architecture with custom ROI heads and backbone.

        Args:
            cfg: The OmegaConf configuration object containing model
                parameters (tasks, modalities, n_classes, max_height, etc.).
        """
        super().__init__()
        self.cfg = cfg
        self.use_dsm = "dsm" in self.cfg.modalities
        self.classification = "classification" in cfg.tasks
        self.height = "height" in cfg.tasks
        if self.classification:
            num_classes = cfg.n_classes + 1
        elif self.height:
            self.height_bin_size = cfg.height_bin_size
            num_classes = int(cfg.max_height / self.height_bin_size + 1e-5) + 1
        else:
            raise ValueError("One task (classification or height) must be specified.")

        self.num_classes = num_classes
        weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
        self.model = maskrcnn_resnet50_fpn(weights=weights, progress=False)

        box_predictor = self.model.roi_heads.box_predictor
        if not isinstance(box_predictor, FastRCNNPredictor):
            raise ValueError("Unexpected box predictor architecture; expected FastRCNNPredictor.")
        in_features = box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        mask_predictor = self.model.roi_heads.mask_predictor
        if not isinstance(mask_predictor, MaskRCNNPredictor):
            raise ValueError("Unexpected mask predictor architecture; expected MaskRCNNPredictor.")
        mask_conv_layer = mask_predictor.conv5_mask
        if not isinstance(mask_conv_layer, (nn.Conv2d, nn.ConvTranspose2d)):
            raise ValueError("Unexpected mask predictor conv layer; expected Conv2d or ConvTranspose2d.")
        in_features_mask = mask_conv_layer.in_channels
        self.model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)

        if self.use_dsm:
            self._expand_backbone_for_dsm()

    def _expand_backbone_for_dsm(self) -> None:
        """Modifies the ResNet backbone to accept 4-channel input (RGB+DSM)."""
        backbone_body = self.model.backbone.body
        maybe_conv = getattr(backbone_body, "conv1", None)
        if not isinstance(maybe_conv, nn.Conv2d):
            raise ValueError("Unexpected backbone architecture; expected 'conv1' attribute.")
        old_conv: nn.Conv2d = maybe_conv
        k_size = cast(tuple[int, int], old_conv.kernel_size)
        s_size = cast(tuple[int, int], old_conv.stride)
        p_size = cast(tuple[int, int], old_conv.padding)
        new_conv = nn.Conv2d(
            in_channels=4,
            out_channels=old_conv.out_channels,
            kernel_size=k_size,
            stride=s_size,
            padding=p_size,
            bias=(old_conv.bias is not None),
        )
        with torch.no_grad():
            new_conv.weight[:, :3] = old_conv.weight
            new_conv.weight[:, 3] = old_conv.weight.mean(dim=1)
            if old_conv.bias is not None:
                new_conv.bias = old_conv.bias
        setattr(backbone_body, "conv1", new_conv)

        transform = self.model.transform
        image_mean = getattr(transform, "image_mean")
        image_std = getattr(transform, "image_std")
        if not isinstance(image_mean, list) or not isinstance(image_std, list):
            raise ValueError("Unexpected transform architecture; expected lists for mean and std.")
        new_mean = image_mean + [0.0]
        new_std = image_std + [1.0]
        setattr(transform, "image_mean", new_mean)
        setattr(transform, "image_std", new_std)

    def forward(
        self, img: torch.Tensor, dsm: torch.Tensor | None, targets: dict[str, torch.Tensor], return_loss_list: bool
    ) -> tuple[list[torch.Tensor], torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """
        Executes the forward pass for training or inference.

        Args:
            img: The input RGB image tensor of shape (B, 3, H, W).
            dsm: The input Digital Surface Model tensor of shape (B, 1, H, W).
            targets: Dictionary containing ground truth 'class' or 'height'
                and 'segmentation' masks.
            return_loss_list: If True, returns the internal R-CNN loss dictionary.

        Returns:
            A tuple of (losses, class_preds, mask_preds, height_preds).
        """
        x = torch.cat((img, dsm), dim=1) if self.use_dsm and dsm is not None else img

        adapted_targets = self.adapt_targets(targets) if targets is not None else None

        losses = []

        if return_loss_list:
            self.model.train()
            losses = self.model(x, targets=adapted_targets)
            self.model.eval()

        with torch.no_grad():
            raw_predictions = self.model(x)
            filtered_predictions = [self.select_center_object(pred) for pred in raw_predictions]

        labels = torch.stack([p["labels"][0] for p in filtered_predictions])
        mask_preds = torch.stack([p["masks"][0] for p in filtered_predictions])

        class_preds, height_preds = None, None
        if self.classification:
            class_preds = labels - 1
        if self.height:
            height_preds = ((labels - 1).float() * self.height_bin_size) + self.height_bin_size / 2

        return losses, class_preds, mask_preds, height_preds

    def adapt_targets(self, targets: dict[str, torch.Tensor]) -> list[dict[str, torch.Tensor]]:
        """
        Reformats dataset targets into Torchvision-compatible dictionaries.

        Args:
            targets: Dictionary containing 'class'/'height' and 'segmentation' keys.

        Returns:
            A list of dictionaries, each containing 'boxes', 'labels', and 'masks'.
        """
        if self.classification:
            classes = targets["class"] + 1
        else:
            max_bin = int(self.cfg.max_height / self.height_bin_size + 1e-5) - 1
            classes = torch.clamp((targets["height"] / self.height_bin_size).long(), min=0, max=max_bin) + 1

        masks = targets["segmentation"].squeeze(1)
        new_targets = []
        for i in range(classes.shape[0]):
            current_mask = (masks[i].unsqueeze(0) > 0).to(torch.uint8)
            new_targets.append(
                {
                    "boxes": masks_to_boxes(current_mask),
                    "labels": classes[i].unsqueeze(0).to(torch.int64),
                    "masks": current_mask,
                }
            )
        return new_targets

    def select_center_object(self, predictions: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Filters predictions to return only the detection closest to the tile center.

        Args:
            predictions: Dictionary of R-CNN predictions (boxes, scores, masks, labels).

        Returns:
            Dictionary containing the single most central detection.
        """
        boxes = predictions["boxes"]
        if len(boxes) == 0:
            spatial_dim = cast(tuple[int, int], tuple(predictions["masks"].shape[-2:]))
            return self._generate_empty_prediction(boxes.device, spatial_dim)

        center = self.cfg.image_size / 2
        dist = ((boxes[:, 0] + boxes[:, 2]) / 2 - center) ** 2 + ((boxes[:, 1] + boxes[:, 3]) / 2 - center) ** 2
        idx = torch.argmin(dist)
        return {k: v[idx].unsqueeze(0) for k, v in predictions.items()}

    def _generate_empty_prediction(self, device: torch.device, spatial_dim: tuple[int, int]) -> dict[str, torch.Tensor]:
        """
        Creates a placeholder prediction for empty detections.

        Args:
            device: The torch device to place tensors on.
            spatial_dim: The (H, W) dimensions for the empty mask.

        Returns:
            A dictionary of zeroed/placeholder tensors.
        """
        h, w = spatial_dim
        if self.classification:
            label = torch.randint(1, self.num_classes, (1,), device=device, dtype=torch.int64)
        else:
            label = torch.tensor([int(self.num_classes / 2)], device=device, dtype=torch.int64)
        return {
            "boxes": torch.zeros((1, 4), device=device),
            "scores": torch.tensor([0.0], device=device),
            "masks": torch.zeros((1, 1, h, w), device=device),
            "labels": label,
        }
