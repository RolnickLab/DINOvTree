import math
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionEmbeddingSine(nn.Module):
    """
    2D sine-cosine positional encoding for Transformer architectures.

    Generates sinusoidal embeddings based on spatial coordinates, outputting a tensor of shape (B, N, C) where C = 2 *
    num_pos_feats.
    """

    def __init__(self, num_pos_feats: int = 64, normalize: bool = False, scale: float | None = None) -> None:
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        Generate positional embeddings.

        Args:
            x: Input tensor used to determine device and batch size (B, ...).
            h: Height of the spatial grid.
            w: Width of the spatial grid.

        Returns:
            Positional embeddings of shape (B, H*W, 2*num_pos_feats).
        """
        B = x.shape[0]

        y_embed = torch.arange(H, device=x.device).unsqueeze(1).repeat(1, W)
        x_embed = torch.arange(W, device=x.device).unsqueeze(0).repeat(H, 1)
        y_embed = y_embed.unsqueeze(0).repeat(B, 1, 1)
        x_embed = x_embed.unsqueeze(0).repeat(B, 1, 1)

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = 10000 ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t

        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)

        pos = torch.cat((pos_y, pos_x), dim=3)
        pos = pos.view(B, H * W, -1)

        return pos


class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) as used in Vision Transformers and MLP-Mixer.

    This module implements a standard two-layer feed-forward network with a configurable activation function and
    dropout.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        act_layer: Callable = nn.GELU,
        bias: bool = True,
        drop: float = 0.0,
    ) -> None:
        """
        Initialize the MLP module.

        Args:
            in_features: Number of input features.
            hidden_features: Number of features in the hidden layer.
            out_features: Number of output features.
            act_layer: The activation layer constructor or factory.
            bias: Whether to use bias in linear layers.
            drop: Dropout probability.
        """
        super().__init__()

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform the forward pass.

        Args:
            x: Input tensor of shape (..., in_features).

        Returns:
            Output tensor of shape (..., out_features).
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class CrossAttention(nn.Module):
    """Standard multi-head cross-attention mechanism."""

    def __init__(
        self,
        dim_q: int,
        dim_kv: int,
        num_heads: int = 8,
        attn_dropout: float = 0.0,
    ) -> None:
        """
        Initialize the CrossAttention module.

        Args:
            dim_q: Query embedding dimension.
            dim_kv: Key and Value embedding dimension.
            num_heads: Number of attention heads.
            attn_dropout: Dropout probability for attention weights.
        """
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = dim_q // num_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(dim_q, dim_q)
        self.k_proj = nn.Linear(dim_kv, dim_q)
        self.v_proj = nn.Linear(dim_kv, dim_q)

        self.out_proj = nn.Linear(dim_q, dim_q)
        self.dropout = nn.Dropout(attn_dropout)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: None | torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Perform the cross-attention forward pass.

        Args:
            q: Query tensor of shape (B, Nq, Cq).
            k: Key tensor of shape (B, Nk, Ck).
            v: Value tensor of shape (B, Nk, Ck).
            mask: Optional attention mask of shape (B, Nq, Nk).

        Returns:
            A tuple containing:
                - Output tensor: (B, Nq, Cq).
                - Attention weights: (B, num_heads, Nq, Nk).
        """
        B, Nq, _ = q.shape
        _, Nk, _ = k.shape

        q = self.q_proj(q).view(B, Nq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(k).view(B, Nk, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(v).view(B, Nk, self.num_heads, self.head_dim).transpose(1, 2)

        attn_logits = (q @ k.transpose(-2, -1)) * self.scale

        if mask is not None:
            attn_logits = attn_logits.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(attn_logits, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, Nq, -1)
        out = self.out_proj(out)

        return out, attn


class TokenFusion(nn.Module):
    """
    Module for fusing class (CLS) and task-specific tokens.

    Supports multiple fusion strategies: addition, concatenation, or using one token exclusively.
    """

    def __init__(self, method: str = "add") -> None:
        """
        Initialize the TokenFusion module.

        Args:
            method: The fusion strategy. One of 'add', 'concat',
                'task_token_only', or 'cls_token_only'.
        """
        super().__init__()

        self._methods: dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = {
            "add": lambda c, t: c + t,
            "concat": lambda c, t: torch.cat([c, t], dim=-1),
            "task_token_only": lambda c, t: t,
            "cls_token_only": lambda c, t: c,
        }

        if method not in self._methods:
            raise ValueError(f"Method '{method}' is not supported. " f"Choose from: {list(self._methods.keys())}")

        self.method = method

    def forward(self, cls_token: torch.Tensor, task_token: torch.Tensor) -> torch.Tensor:
        """
        Fuse the provided tokens using the initialized method.

        Args:
            cls_token: The class token tensor.
            task_token: The task-specific token tensor.

        Returns:
            The fused token tensor.
        """

        return self._methods[self.method](cls_token, task_token)

    def get_output_dim(self, input_dim: int) -> int:
        """
        Calculate the output dimension after fusion based on the input dimension and fusion method.

        Args:
            input_dim: The dimension of the input tokens.
        Returns:
            The dimension of the output token after fusion.
        """
        if self.method == "concat":
            return input_dim * 2
        else:
            return input_dim
