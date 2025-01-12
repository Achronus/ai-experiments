from typing import Any

import torch
import torch.nn as nn

from pydantic.dataclasses import dataclass

from experiments import Device, Dtype
from experiments.lcm.encoding import PositionEncoder
from experiments.lcm.padding import PaddingMask


@dataclass
class PreNetConfig:
    """
    A data model containing config parameters for the PreNet layer.

    Args:
        bias (bool, optional): a flag to enable layer bias terms. Default is True
        weight_norm (bool, optional): a flag to normalize the layers weights. Default is False
        weight_init_fn (WeightInitLiteral | None, optional): the type of weight initialization to use. Default is "kaiming_uniform"
        activation (ActivationTypeLiteral | None, optional): the type of activation function. Default is None
        dropout_p (float, optional): the dropout probability applied to the layer. Default is 0.0
        layer_norm_style (LayerNormLiteral | None, optional): the type of layer normalization to use. Default is None
        scale_embeddings (bool, optional): a flag to enable embedding scaling by `in_features` before adding positional encoding. Default is 0.006
    """

    scale_embeddings: bool = False
    embedding_std: float = 0.006

    def model_dump(self) -> dict[str, Any]:
        """Returns all fields and their values as a dictionary."""
        return self.__dict__


class PreNet(nn.Module):
    """
    A Base-LCM PreNet.

    Args:
        d_model (int): the model embedding dimension
        d_embed (int): the embedding dimension of the sentence encoder
        pos_encoder (PositionEncoder): the positional encoder to use
        d_timestep_embed (int, optional): the embedding dimension of diffusion timesteps (if relevant). Defaults to 0
        scale_embeddings (bool, optional): a flag to enable embedding scaling by `d_model`. Applies before adding positions and being passed through the linear layer. Defaults to False
        dropout (float, optional): the dropout probability. Default is 0.0
    """

    def __init__(
        self,
        d_model: int,
        d_embed: int,
        pos_encoder: PositionEncoder,
        *,
        d_timestep_embed: int = 0,
        scale_embeddings: bool = False,
        dropout: float = 0.0,
        dtype: Dtype | None = None,
        device: Device | None = None,
    ) -> None:
        super().__init__()

        self.pos_encoder = pos_encoder

        self.dtype = dtype
        self.device = device

        self.embed_scale: float = d_model**0.5 if scale_embeddings else 1.0
        print(f"Using PreNet with embeddings scaler = {self.embed_scale}")

        self.fc = nn.Linear(
            d_embed + d_timestep_embed,
            d_model,
            dtype=dtype,
            device=device,
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, padding_mask: PaddingMask | None
    ) -> tuple[torch.Tensor, PaddingMask | None]:
        """Performs a forward pass through the network."""
        x = self.fc(self.embed_scale * x)
        x = self.pos_encoder(x, padding_mask)
        x = self.dropout(x)
        return x, padding_mask
