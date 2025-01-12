from abc import abstractmethod

import torch
import torch.nn as nn


from experiments.lcm.linear import Linear
from experiments import (
    Dtype,
    Device,
    WeightInitLiteral,
    ActivationTypeLiteral,
    ActivationEnum,
)


class FFN(nn.Module):
    """
    A base class for feed-forward networks.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()

        self.d_model = d_model

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs a forward pass through the network."""


class FeedForwardNetwork(FFN):
    """
    A standard multi-layer perceptron network with two layers.

    Args:
        d_model (int): the number of input features
        n_hidden (int): the number of hidden nodes in the layers
        dropout (float, optional): the dropout probability. Default is 0.1
        weight_init_fn (WeightInitLiteral | None, optional): the type of weight initialization to use. Default is "kaiming_uniform"
        activation (ActivationTypeLiteral | None, optional): the type of activation function. Default is "relu"
        dtype (torch.dtype, optional): a custom torch datatype for all tensors. Default is None
        device (torch.device, optional): the compute device to load tensors onto. Default is None
    """

    def __init__(
        self,
        d_model: int,
        n_hidden: int,
        dropout: float = 0.1,
        weight_init_fn: WeightInitLiteral | None = "kaiming_uniform",
        activation: ActivationTypeLiteral | None = "relu",
        *,
        dtype: Dtype | None = None,
        device: Device | None = None,
    ) -> None:
        super().__init__(d_model)

        self.fc = nn.Sequential(
            Linear(
                d_model,
                n_hidden,
                init_fn=weight_init_fn,
                dtype=dtype,
                device=device,
            ),
            ActivationEnum.get(activation),
            nn.Dropout(dropout),
            Linear(
                n_hidden,
                d_model,
                init_fn=weight_init_fn,
                dtype=dtype,
                device=device,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs a forward pass through the network."""
        return self.fc(x)


class GLUFeedForwardNetwork(FFN):
    """
    A multi-layer perceptron network with two layers: _inner_ and _outer_ that uses a gated linear unit (GLU) on the _inner_.

    Args:
        d_model (int): the number of input features
        n_hidden (int): the number of hidden nodes in the layers
        activation (ActivationTypeLiteral | None, optional): the type of activation function to apply to the outputs of the gate projection. Default is "silu"
        inner_dim_scale (float, optional): the scale factor for the dimensionality of the inner projection layer. Default is 2 / 3
        inner_dim_to_multiple (int, optional): the rounded up value for the inner projection layer. Default is 1
        dropout (float, optional): the dropout probability. Default is 0.0
        dtype (torch.dtype, optional): a custom torch datatype for all tensors. Default is None
        device (torch.device, optional): the compute device to load tensors onto. Default is None
    """

    def __init__(
        self,
        d_model: int,
        n_hidden: int,
        *,
        activation: ActivationTypeLiteral = "silu",
        inner_dim_scale: float = 2 / 3,
        inner_dim_to_multiple: int = 1,
        dropout: float = 0.0,
        dtype: Dtype | None = None,
        device: Device | None = None,
    ) -> None:
        super().__init__(d_model)

        self.inner_dim_scale = inner_dim_scale
        self.inner_dim_to_multiple = inner_dim_to_multiple

        if inner_dim_scale != 1.0:
            inner_dim = int(n_hidden * inner_dim_scale)

        if inner_dim_to_multiple != 1:
            inner_dim = inner_dim_to_multiple * (
                (inner_dim + inner_dim_to_multiple - 1) // inner_dim_to_multiple
            )

        self.gate_proj = nn.Sequential(
            Linear(d_model, n_hidden, dtype=dtype, device=device),
            ActivationEnum.get(activation),
        )

        self.inner = Linear(
            d_model,
            n_hidden,
            dtype=dtype,
            device=device,
        )

        self.outer = nn.Sequential(
            nn.Dropout(dropout),
            Linear(
                n_hidden,
                d_model,
                dtype=dtype,
                device=device,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs a forward pass through the network."""
        gate = self.gate_proj(x)
        x = self.inner(x)

        x = x * gate
        return self.outer(x)
