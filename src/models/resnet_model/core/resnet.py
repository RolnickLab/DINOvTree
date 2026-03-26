from typing import cast

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T

from src.utils.head import LinearHead


class ResNet(nn.Module):
    """A ResNet-based encoder with support for multi-modal (RGB + DSM) input and a custom multi-task head for tree
    species classification or height estimation."""

    def __init__(
        self,
        num_classes: int,
        backbone: str,
        classification: bool,
        height: bool,
        use_dsm: bool,
        use_sigmoid_for_height: bool,
        max_height: float,
    ) -> None:
        """
        Initializes the ResNet architecture and adapts the input layer for DSM if required.

        Args:
            num_classes: Number of target species categories.
            backbone: String name of the ResNet variant (e.g., 'resnet18', 'resnet50').
            classification: Boolean flag to enable the classification output.
            height: Boolean flag to enable the height regression output.
            use_dsm: If True, the model expects a 4-channel input (RGB + DSM).
            use_sigmoid_for_height: If True, applies sigmoid activation to height predictions.
            max_height: Maximum height value for normalization if using sigmoid.
        """
        super().__init__()
        self.use_dsm = use_dsm

        if backbone == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT
            self.backbone = models.resnet18(weights=weights)
        elif backbone == "resnet34":
            weights = models.ResNet34_Weights.DEFAULT
            self.backbone = models.resnet34(weights=weights)
        elif backbone == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT
            self.backbone = models.resnet50(weights=weights)
        elif backbone == "resnet101":
            weights = models.ResNet101_Weights.DEFAULT
            self.backbone = models.resnet101(weights=weights)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        norm_mean = weights.transforms().mean
        norm_std = weights.transforms().std
        self.normalize = T.Normalize(mean=norm_mean, std=norm_std)

        if self.use_dsm:
            self._expand_for_dsm()

        self.feat_dim: int = self.backbone.fc.in_features
        setattr(self.backbone, "fc", nn.Identity())
        self.head = LinearHead(
            in_dim=self.feat_dim,
            classification=classification,
            height=height,
            num_classes=num_classes,
            use_sigmoid_for_height=use_sigmoid_for_height,
            max_height=max_height,
        )

    def _expand_for_dsm(self) -> None:
        """Internal helper to modify the first convolution to accept 4 channels."""
        old_conv = self.backbone.conv1

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
            new_conv.weight[:, 3] = old_conv.weight[:, :1].mean(dim=1)

            if old_conv.bias is not None:
                new_conv.bias = old_conv.bias

        self.backbone.conv1 = new_conv

    def forward(self, img: torch.Tensor, dsm: torch.Tensor | None) -> torch.Tensor:
        """
        Args:
            img: Input image tensor of shape (B, 3, H, W).
            dsm: Optional Digital Surface Model tensor of shape (B, 1, H, W).

        Returns:
            The output tensor from the LinearHead.
        """
        img = self.normalize(img)
        if self.use_dsm:
            if dsm is None:
                raise ValueError("DSM input is required but not provided for this model configuration.")
            x = torch.cat((img, dsm), dim=1)
        else:
            x = img

        features = self.backbone(x)

        out: torch.Tensor = self.head(features)
        return out
