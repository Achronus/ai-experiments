import torch
import torch.nn as nn

from pydantic.dataclasses import dataclass

from experiments import Device, Dtype, ActivationEnum, ActivationTypeLiteral
from experiments.linear import Linear
from experiments.weight import WeightInitLiteral, get_init_fn


@dataclass
class LinearProjectionConfig:
    """
    A data model containing config parameters for the LinearProjection layer.

    Args:
        bias (bool, optional): a flag to enable layer bias terms. Default is True
        weight_norm (bool, optional): a flag to normalize the layers weights. Default is False
        weight_init_fn (WeightInitLiteral | None, optional): the type of weight initialization to use. Default is "kaiming_uniform"
        activation (ActivationTypeLiteral | None, optional): the type of activation function. Default is None
    """

    bias: bool = True
    weight_norm: bool = False
    weight_init_fn: WeightInitLiteral | None = "kaiming_uniform"
    activation: ActivationTypeLiteral | None = None


class LinearProjection(nn.Module):
    """
    Applies a linear transformation to incoming data using weights and biases
    with customization such as the type of weight initialization and activation function to apply.

    Args:
        in_features (int): the number of input features
        out_features (int): the number of output features
        config (LinearProjectionConfig): configuration settings for the layer
        dtype (torch.dtype, optional): a custom torch datatype for all tensors. Default is None
        device (torch.device, optional): the compute device to load tensors onto. Default is None
    """

    weight: nn.Parameter
    bias: nn.Parameter | None
    activation_fn: nn.Module | None

    def __init__(
        self,
        in_features: int,
        out_features: int,
        config: LinearProjectionConfig,
        dtype: Dtype | None = None,
        device: Device | None = None,
    ) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.config = config
        self.dtype = dtype
        self.device = device

        self.init_fn = get_init_fn(config.weight_init_fn)
        linear = Linear(
            in_features,
            out_features,
            bias=config.bias,
            init_fn=config.weight_init_fn,
            dtype=dtype,
            device=device,
        )

        self.fc = (
            torch.nn.utils.parametrizations.weight_norm(linear)
            if config.weight_norm
            else linear
        )

        if config.activation is not None:
            # Some activation functions have parameters (e.g., PReLU)
            # so need to load on device
            self.activation_fn = ActivationEnum.get(config.activation).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        x = self.fc(x)

        if self.activation_fn is not None:
            x = self.activation_fn(x)

        return x
