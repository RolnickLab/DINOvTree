import json
from contextlib import nullcontext
from pathlib import Path
from typing import Final, cast

import torch
from torch import nn
from torchvision import transforms

from src.utils.head import LinearHead

_REPO_DIR: Final[str] = str(Path(__file__).resolve().parents[4] / "external" / "facebookresearch_dinov3")
_URL_FILE: Final[Path] = Path(__file__).resolve().parent / "dinov3_urls.json"

_USER_URLS: dict[str, str] = {}
if _URL_FILE.exists():
    with open(_URL_FILE, "r") as f:
        _USER_URLS = json.load(f)

_BACKBONES: Final[dict[str, tuple[str, str, int]]] = {
    "dinov3_vits16": ("dinov3_vits16", _USER_URLS.get("dinov3_vits16", ""), 384),
    "dinov3_vits16plus": ("dinov3_vits16plus", _USER_URLS.get("dinov3_vits16plus", ""), 384),
    "dinov3_vitb16": ("dinov3_vitb16", _USER_URLS.get("dinov3_vitb16", ""), 768),
    "dinov3_vitl16": ("dinov3_vitl16", _USER_URLS.get("dinov3_vitl16", ""), 1024),
    "dinov3_vith16plus": ("dinov3_vith16plus", _USER_URLS.get("dinov3_vith16plus", ""), 1280),
    "dinov3_vit7b16": ("dinov3_vit7b16", _USER_URLS.get("dinov3_vit7b16", ""), 4096),
    "dinov3_vitl16_sat": ("dinov3_vitl16", _USER_URLS.get("dinov3_vitl16_sat", ""), 1024),
    "dinov3_vit7b16_sat": ("dinov3_vit7b16", _USER_URLS.get("dinov3_vit7b16_sat", ""), 4096),
}


class DINOv3(nn.Module):
    """Wrapper for DINOv3 models supporting feature extraction and task-specific heads."""

    def __init__(
        self,
        return_features: bool,
        version: str,
        freeze_backbone: bool,
        classification: bool = False,
        height: bool = False,
        num_classes: int = 0,
        avg_pool: bool = True,
        use_sigmoid_for_height: bool = False,
        max_height: float = 1.0,
    ) -> None:
        """
        Initialize the DINOv3 model.

        Args:
            return_features: If True, returns raw patch and CLS tokens.
            version: Model variant key from _BACKBONES.
            freeze_backbone: If True, backbone parameters are locked.
            classification: Enable classification head.
            height: Enable height regression head.
            num_classes: Number of output classes.
            avg_pool: If True, concatenates CLS token with global average pooled patches.
            use_sigmoid_for_height: Activation choice for height head.
            max_height: Scaling factor for height predictions.
        """
        super().__init__()
        self.return_features = return_features
        self.avg_pool = avg_pool

        backbone, self.normalize = self.create_backbone(version)
        self.embed_dim: int = _BACKBONES[version][2]

        self.feature_model = ModelWithIntermediateLayers(
            backbone,
            freeze_backbone,
            self.return_features,
        )

        if not self.return_features:
            out_dim = (1 + int(self.avg_pool)) * self.embed_dim
            self.head = LinearHead(
                in_dim=out_dim,
                classification=classification,
                height=height,
                num_classes=num_classes,
                use_sigmoid_for_height=use_sigmoid_for_height,
                max_height=max_height,
            )

    def create_backbone(
        self,
        version: str = "dinov3_vitl16",
    ) -> tuple[nn.Module, transforms.Normalize]:
        """
        Load the pretrained DINOv3 backbone and determine normalization stats.

        Args:
            version: The model variant key to look up in the _BACKBONES registry.

        Returns:
            A tuple containing:
                - backbone_instance: The PyTorch module loaded from torch.hub.
                - normalize: A torchvision transform with model-specific mean and std.

        Raises:
            ValueError: If the provided version is not found in _BACKBONES.
        """
        if version not in _BACKBONES:
            raise ValueError(f"Unsupported version '{version}'.")

        model_name, pretrained_weights, _ = _BACKBONES[version]

        if "sat" in model_name:
            normalize = transforms.Normalize(
                mean=(0.430, 0.411, 0.296),
                std=(0.213, 0.156, 0.143),
            )
        else:
            normalize = transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            )
        backbone_instance = cast(
            nn.Module,
            torch.hub.load(
                _REPO_DIR,
                model_name,
                source="local",
                weights=pretrained_weights,
            ),
        )

        return backbone_instance, normalize

    def get_output_dim(self) -> int:
        """
        Retrieve the embedding dimension of the backbone.

        Returns:
            The integer embedding dimension (e.g., 384, 768, 1024).
        """
        return self.embed_dim

    def forward(
        self, img: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor | None, torch.Tensor | None]:
        """
        Perform a forward pass through the backbone and optional heads.

        Args:
            img: Input image tensor of shape (B, 3, H, W).

        Returns:
            If return_features is True:
                A tuple of (patch_tokens, cls_token).
            If return_features is False:
                A tuple of (class_logits, height_preds).
        """
        img = self.normalize(img)

        features = self.feature_model(img)

        if self.return_features:
            return features

        patch_tokens, cls_token = features
        cls_token = cls_token.squeeze(1)

        if self.avg_pool:
            avg_pool_token = torch.mean(patch_tokens, dim=1)
            x = torch.cat((cls_token, avg_pool_token), dim=-1)
        else:
            x = cls_token

        return self.head(x)


class ModelWithIntermediateLayers(nn.Module):
    """Utility to extract intermediate layers from a DINOv3 backbone."""

    def __init__(
        self,
        feature_model: nn.Module,
        freeze_backbone: bool,
        return_features: bool,
    ) -> None:
        """
        Initialize the feature extraction utility.

        Args:
            feature_model: The backbone nn.Module instance.
            freeze_backbone: If True, disables gradients for all backbone parameters.
            return_features: Determines whether to reshape spatial tokens (H, W).
        """
        super().__init__()
        self.freeze_backbone = freeze_backbone
        self.feature_model = feature_model
        self.return_features = return_features

        if self.freeze_backbone:
            self.feature_model.eval()

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Extract the last layer tokens from the backbone.

        Args:
            images: Input image tensor of shape (B, 3, H, W).

        Returns:
            A tuple of (patch_tokens, cls_token) extracted from the final layer.
        """
        ctx = torch.no_grad() if self.freeze_backbone else nullcontext()
        get_layers_fn = getattr(self.feature_model, "get_intermediate_layers", None)
        if get_layers_fn is None:
            raise AttributeError(
                f"The backbone {type(self.feature_model).__name__} does not " "support 'get_intermediate_layers'."
            )
        with ctx:
            outputs = get_layers_fn(images, n=1, reshape=self.return_features, return_class_token=True)
            return outputs[0]
