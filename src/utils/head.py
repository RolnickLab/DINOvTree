import torch
import torch.nn as nn


class LinearHead(nn.Module):
    """
    Linear output heads.

    Provides classification and/or height regression layers designed to be trained on top of backbone features.
    """

    def __init__(
        self,
        in_dim: int,
        classification: bool,
        height: bool,
        num_classes: int = 1000,
        use_sigmoid_for_height: bool = False,
        max_height: float = 1.0,
    ) -> None:
        """
        Initialize the LinearHead module.

        Args:
            in_dim: Dimensionality of the input features.
            classification: Whether to include a classification layer.
            height: Whether to include a height regression layer.
            num_classes: Number of target classes for classification.
            use_sigmoid_for_height: Whether to use Sigmoid activation for height.
            max_height: Scaling factor applied to Sigmoid height predictions.
        """
        super().__init__()
        self.classification = classification
        self.height = height
        self.use_sigmoid_for_height = use_sigmoid_for_height
        self.max_height = max_height

        if classification:
            self.classifier = nn.Linear(in_dim, num_classes)

        if height:
            self.regressor = nn.Sequential(
                nn.Linear(in_dim, 1),
                nn.Sigmoid() if use_sigmoid_for_height else nn.ReLU(),
            )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """
        Execute the forward pass for enabled tasks.

        Args:
            x: Input feature tensor of shape (B, 1, C).

        Returns:
            A tuple containing:
                - class_logits: Predicted logits (B, num_classes) or None.
                - height_preds: Predicted heights (B,) or None.
        """
        class_logits: torch.Tensor | None = None
        height_preds: torch.Tensor | None = None

        if self.classification:
            class_logits = self.classifier(x).squeeze(1)

        if self.height:
            raw_preds = self.regressor(x).squeeze(1)
            if self.use_sigmoid_for_height:
                height_preds = raw_preds * self.max_height
            else:
                height_preds = raw_preds

        return class_logits, height_preds
