import torch
import torch.nn as nn

from pydantic.dataclasses import dataclass

from experiments import Device, Dtype, ActivationEnum, ActivationTypeLiteral


@dataclass
class LinearProjectionConfig:
    """
    A data model containing config parameters for the LinearProjection layer.

    Args:
        in_features (int): the number of input features (sonar_dim)
        out_features (int): the number of output features (model_dim)
        activation (ActivationTypeLiteral | ActivationEnum): the type of activation function
        weight_norm (bool, optional): a flag to normalize the layers weights. Default is False
    """

    in_features: int
    out_features: int
    activation: ActivationTypeLiteral | ActivationEnum
    weight_norm: bool = False


class LinearProjection(nn.Module):
    """
    A Linear projection layer commonly used for mapping from one embedding format to another.

    Args:
        config (LinearProjectionConfig): a model containing the layers configuration settings
        dtype (torch.dtype, optional): a custom torch datatype for all tensors. Default is None
        device (torch.device, optional): the compute device to load tensors onto. Default is None
    """

    def __init__(
        self,
        config: LinearProjectionConfig,
        dtype: Dtype | None = None,
        device: Device | None = None,
    ) -> None:
        super().__init__()

        linear = nn.Linear(
            config.in_features,
            config.out_features,
            dtype=dtype,
            device=device,
        )

        self.fc = (
            torch.nn.utils.parametrizations.weight_norm(linear)
            if config.weight_norm
            else linear
        )
        self.activation_fn = ActivationEnum.get(config.activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        return self.activation_fn(self.fc(x))
