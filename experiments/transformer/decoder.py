import copy

import torch
import torch.nn as nn

from pydantic.dataclasses import dataclass

from experiments import (
    ActivationTypeLiteral,
    LayerNormLiteral,
    LayerNorm,
    Dtype,
    Device,
    WeightInitLiteral,
)
from experiments.linear import FeedForwardNetwork, GLUFeedForwardNetwork


@dataclass
class DecoderOnlyLayerConfig:
    """
    A data model containing config parameters for the DecoderOnlyLayer.

    Args:
        n_hidden (int, optional): number of nodes in the feed-forward network. Default is 2048
        dropout (float, optional): the dropout probability for the feed-forward network. Default is 0.1 (10%)
        weight_init_fn (WeightInitLiteral | None, optional): the type of weight initialization to use for the feed-forward network. Default is "kaiming_uniform"
        activation (ActivationTypeLiteral, optional): the type of activation function to use in the feed-forward network. Default is "relu"
        norm_type (LayerNormLiteral, optional) the type of layer normalization to use. Default is "standard"
        use_swiglu (bool, optional): a flag for using GLU Feed-Forward networks instead of regular ones. Default is False
    """

    n_hidden: int = 2048
    dropout: float = 0.1
    weight_init_fn: WeightInitLiteral | None = "kaiming_uniform"
    activation: ActivationTypeLiteral = "relu"
    norm_type: LayerNormLiteral = "standard"
    use_swiglu: bool = False


class DecoderOnlyLayer(nn.Module):
    """
    A basic decoder only transformer layer. Layer normalization is applied first.

    Args:
        d_model (int): the number of input features
        n_heads (int): number of multi-head attention heads
        config (DecoderOnlyLayerConfig): a model containing the modules configuration settings
        dtype (torch.dtype, optional): a custom torch datatype for all tensors. Default is None
        device (torch.device, optional): the compute device to load tensors onto. Default is None
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        config: DecoderOnlyLayerConfig,
        *,
        dtype: Dtype | None = None,
        device: Device | None = None,
    ) -> None:
        super().__init__()

        self.self_attn = nn.MultiheadAttention(
            d_model,
            n_heads,
            dtype=dtype,
            device=device,
        )
        self.norm1 = LayerNorm(
            d_model,
            config.norm_type,
            dtype=dtype,
            device=device,
        )
        self.dropout1 = nn.Dropout(config.dropout)

        if config.use_swiglu:
            self.fc = GLUFeedForwardNetwork(
                d_model,
                config.n_hidden,
                inner_dim_to_multiple=256,
                dtype=dtype,
                device=device,
            )
        else:
            self.fc = FeedForwardNetwork(
                d_model,
                config.n_hidden,
                config.dropout,
                config.weight_init_fn,
                config.activation,
                dtype=dtype,
                device=device,
            )

        self.norm2 = LayerNorm(
            d_model,
            config.norm_type,
            dtype=dtype,
            device=device,
        )
        self.dropout2 = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """Performs a forward pass through the network."""
        # Self-attention block with residual connection
        residual = x
        x = self.norm1(x)
        x = residual + self.dropout1(self.self_attn(x, mask))

        # Feed-forward block with residual connection
        residual = x
        x = self.norm2(x)
        x = residual + self.dropout2(self.fc(x))
        return x


class TransformerDecoder(nn.Module):
    """
    A transformer decoder made up of a stack of N decoder only layers.

    Args:
        layer (DecoderOnlyLayer): a single decoder-only layer to be copied
        n_layers (int): number of sub-decoder layers
    """

    def __init__(self, layer: DecoderOnlyLayer, n_layers: int) -> None:
        super().__init__()

        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layers)])

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Runs a forward pass through all decoder layers.

        Args:
            x (torch.Tensor): the sequence to pass through the decoder. Shape `(batch_size, seq_len, d_model)`
            mask (torch.Tensor, optional): an optional attention mask. Default is None

        """
        for layer in self.layers:
            x = layer(x, mask)

        return x
