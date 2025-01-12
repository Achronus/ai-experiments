from typing import Callable, Self

import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments import Device, Dtype
from experiments.weight import init_linear_kaiming_uniform


class Linear(nn.Module):
    """
    Applies a linear transformation to incoming data using weights and biases.

    This class is similar to `torch.nn.Linear` with the addition of customizable parameter initialization.

    Args:
        in_features (int): the number of input features
        out_features (int): the number of output features
        bias (bool, optional): a flag to enable bias nodes in the layer. Default is True
        init_fn (Callable, optional): the type of weight initialization to apply. Default is 'kaiming_uniform'
        dtype (torch.dtype, optional): a custom torch datatype for all tensors. Default is None
        device (torch.device, optional): the compute device to load tensors onto. Default is None
    """

    weight: nn.Parameter
    bias: nn.Parameter | None

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        *,
        init_fn: Callable[[Self], None] | None = None,
        dtype: Dtype | None = None,
        device: Device | None = None,
    ) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.init_fn = init_fn
        self.dtype = dtype
        self.device = device

        self.weight = nn.Parameter(
            torch.empty(
                (out_features, in_features),
                dtype=dtype,
                device=device,
            )
        )

        if bias:
            self.bias = nn.Parameter(
                torch.empty(
                    out_features,
                    dtype=dtype,
                    device=device,
                )
            )
        else:
            self.register_buffer("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset the parameters and buffers of the module."""
        if self.init_fn is not None:
            self.init_fn(self)
        else:
            init_linear_kaiming_uniform(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        return F.linear(x, self.weight, self.bias)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"
