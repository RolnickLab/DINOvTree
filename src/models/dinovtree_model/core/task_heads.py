from typing import Final

import torch
from omegaconf import DictConfig
from torch import nn

from src.models.dinovtree_model.core.modules import (
    MLP,
    CrossAttention,
    PositionEmbeddingSine,
    TokenFusion,
)


class TaskHeads(nn.Module):
    """Multi-task head architecture of DINOvTree."""

    def __init__(self, cfg: DictConfig, backbone_dim: int) -> None:
        """
        Initialize the TaskHeads module.

        Args:
            cfg: Configuration object containing architecture flags
                (e.g., use_shared_mlp, tasks, fusion_method_cls).
            backbone_dim: The dimensionality of the features provided by the backbone.
        """
        super().__init__()
        self.cfg: Final = cfg

        self.classification: bool = "classification" in cfg.tasks
        self.height: bool = "height" in cfg.tasks

        self.norm1 = nn.LayerNorm(backbone_dim)
        if self.cfg.use_task_specific_mlp:
            if self.classification:
                self.mlp_cls = MLP(
                    in_features=backbone_dim, hidden_features=backbone_dim * 4, out_features=backbone_dim
                )
            if self.height:
                self.mlp_height = MLP(
                    in_features=backbone_dim, hidden_features=backbone_dim * 4, out_features=backbone_dim
                )
        elif self.cfg.use_shared_mlp:
            self.mlp = MLP(in_features=backbone_dim, hidden_features=backbone_dim * 4, out_features=backbone_dim)
        self.norm2 = nn.LayerNorm(backbone_dim)

        if self.cfg.use_position_embedding:
            self.position_embedding = PositionEmbeddingSine(backbone_dim // 2, normalize=True)

        self._setup_attention_and_queries(backbone_dim)

        if self.classification:
            self._setup_classification_head(backbone_dim)
        if self.height:
            self._setup_height_head(backbone_dim)

    def _setup_attention_and_queries(self, dim: int) -> None:
        """
        Configure cross-attention layers and query parameters.

        Args:
            dim: The embedding dimension for queries and KV pairs.
        """
        if self.cfg.use_shared_query:
            self.query_token = nn.Parameter(torch.randn(1, 1, dim))
        else:
            if self.classification:
                self.query_token_cls = nn.Parameter(torch.randn(1, 1, dim))
            if self.height:
                self.query_token_height = nn.Parameter(torch.randn(1, 1, dim))

        if not self.cfg.use_shared_query and not self.cfg.use_shared_attn_weights:
            if self.classification:
                self.cross_attn_cls = CrossAttention(dim_q=dim, dim_kv=dim)
            if self.height:
                self.cross_attn_height = CrossAttention(dim_q=dim, dim_kv=dim)
        else:
            self.cross_attn_shared = CrossAttention(dim_q=dim, dim_kv=dim)

    def _setup_classification_head(self, dim: int) -> None:
        """
        Initialize the classification fusion and linear layers.

        Args:
            dim: Input dimension from the attention stage.
        """
        self.tf_cls = TokenFusion(self.cfg.fusion_method_cls)
        out_dim = self.tf_cls.get_output_dim(dim)
        self.norm_cls = nn.LayerNorm(out_dim)
        self.classifier = nn.Linear(out_dim, self.cfg.n_classes)

    def _setup_height_head(self, dim: int) -> None:
        """
        Initialize the height regression fusion and linear layers.

        Args:
            dim: Input dimension from the attention stage.
        """
        self.tf_height = TokenFusion(self.cfg.fusion_method_height)
        out_dim = self.tf_height.get_output_dim(dim)
        self.norm_height = nn.LayerNorm(out_dim)
        act = nn.Sigmoid() if self.cfg.use_sigmoid_for_height else nn.ReLU()
        self.regressor = nn.Sequential(nn.Linear(out_dim, 1), act)

    def forward(self, features: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """
        Perform the forward pass of the TaskHeads module.

        Args:
            features: A tuple containing the spatial patch tokens and the
                global classification token.

        Returns:
            A tuple containing the classification logits and height predictions.
        """
        patch_tokens, backbone_cls_token = features
        cls_token = backbone_cls_token.unsqueeze(1)
        B, C, H, W = patch_tokens.shape

        patch_tokens_flat = patch_tokens.flatten(2).transpose(1, 2)
        x = self._apply_mlp(patch_tokens_flat)

        if isinstance(x, dict):
            x = {key: self.norm2(val) for key, val in x.items()}
        else:
            x = self.norm2(x)

        if self.cfg.use_position_embedding:
            x_pos = self.position_embedding(patch_tokens_flat, H, W)
        else:
            shape_ref = next(iter(x.values())) if isinstance(x, dict) else x
            x_pos = torch.zeros_like(shape_ref)

        outputs = self._compute_cross_attention(x, x_pos, B)

        logits = self._process_cls(cls_token, outputs["cls"]) if self.classification else None
        height = self._process_height(cls_token, outputs["height"]) if self.height else None

        return logits, height

    def _apply_mlp(self, x: torch.Tensor) -> torch.Tensor | dict[str, torch.Tensor]:
        """
        Apply shared or task-specific MLPs to the patch tokens.

        Args:
            x: Flattened patch tokens of shape (B, N, C).

        Returns:
            A single tensor if using a shared MLP, or a dictionary of tensors
            mapping task names to refined features if using task-specific MLPs.
        """
        if self.cfg.use_task_specific_mlp:
            x_norm = self.norm1(x)
            x_dict: dict[str, torch.Tensor] = {}
            if self.classification:
                x_dict["cls"] = x + self.mlp_cls(x_norm)
            if self.height:
                x_dict["height"] = x + self.mlp_height(x_norm)
            return x_dict

        if self.cfg.use_shared_mlp:
            x_norm = self.norm1(x)
            return x + self.mlp(x_norm)

        return x

    def _compute_cross_attention(
        self,
        x: torch.Tensor | dict[str, torch.Tensor],
        x_pos: torch.Tensor,
        batch_size: int,
    ) -> dict[str, torch.Tensor]:
        """
        Execute the cross-attention mechanism based on the sharing configuration.

        Args:
            x: Input features from the MLP stage (shared tensor or dict of tensors).
            x_pos: Positional embedding tensor.
            batch_size: The current batch size to expand queries correctly.

        Returns:
            A dictionary containing the attention outputs mapped by task name.
        """
        outputs: dict[str, torch.Tensor] = {}
        if not self.cfg.use_shared_query and not self.cfg.use_shared_attn_weights:
            if self.classification:
                q_cls = self.query_token_cls.expand(batch_size, -1, -1)
                k_cls, v_cls = self._get_kv(x, x_pos, "cls")
                cls_attn_out, _ = self.cross_attn_cls(q_cls, k_cls, v_cls)
                outputs["cls"] = cls_attn_out

            if self.height:
                q_height = self.query_token_height.expand(batch_size, -1, -1)
                k_height, v_height = self._get_kv(x, x_pos, "height")
                height_attn_out, _ = self.cross_attn_height(q_height, k_height, v_height)
                outputs["height"] = height_attn_out

            return outputs

        if self.cfg.use_shared_query:
            q = self.query_token.expand(batch_size, -1, -1)
        else:
            queries = []
            task_order = []
            if self.classification:
                queries.append(self.query_token_cls.expand(batch_size, -1, -1))
                task_order.append("cls")
            if self.height:
                queries.append(self.query_token_height.expand(batch_size, -1, -1))
                task_order.append("height")
            q = torch.cat(queries, dim=1)

        k, v = self._get_kv(x, x_pos, "")
        attn_out, _ = self.cross_attn_shared(q, k, v)

        if self.cfg.use_shared_query:
            if self.classification:
                outputs["cls"] = attn_out
            if self.height:
                outputs["height"] = attn_out
        else:
            for i, task_name in enumerate(task_order):
                outputs[task_name] = attn_out[:, i : i + 1, :]
        return outputs

    def _get_kv(
        self, x: torch.Tensor | dict[str, torch.Tensor], pos: torch.Tensor, task: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generate Key and Value pairs for cross-attention.

        Args:
            x: Feature source (either shared tensor or task-specific dict).
            pos: Positional embedding tensor of shape (B, N, C).
            task: The task name key to look up in x if it is a dictionary.

        Returns:
            A tuple of (Key, Value) tensors.
        """
        v = x[task] if isinstance(x, dict) else x
        k = v + pos
        return k, v

    def _process_cls(self, cls_token: torch.Tensor, task_token: torch.Tensor) -> torch.Tensor:
        """
        Fuse the backbone CLS token with the task-specific token and classify.

        Args:
            cls_token: The original CLS token from the backbone.
            task_token: The token produced by cross-attention for the classification task.

        Returns:
            Class logits of shape (B, num_classes).
        """
        x = self.tf_cls(cls_token, task_token)
        return self.classifier(self.norm_cls(x)).squeeze(1)

    def _process_height(self, cls_token: torch.Tensor, task_token: torch.Tensor) -> torch.Tensor:
        """
        Fuse the backbone CLS token with the task token and regress height.

        Args:
            cls_token: The original CLS token from the backbone.
            task_token: The token produced by cross-attention for the height task.

        Returns:
            Height predictions of shape (B,).
        """
        x = self.tf_height(cls_token, task_token)
        h = self.regressor(self.norm_height(x)).squeeze((1, 2))
        return h * self.cfg.max_height if self.cfg.use_sigmoid_for_height else h
