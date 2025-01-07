import torch

from experiments.activation import ActivationEnum, ActivationTypeLiteral
from experiments.linear import LinearProjection, LinearProjectionConfig


Dtype = torch.dtype
"""Type alias for `torch.dtype`."""

Device = torch.device
"""Type alias for `torch.device`."""


__all__ = [
    "ActivationEnum",
    "ActivationTypeLiteral",
    "LinearProjectionConfig",
    "LinearProjection",
]
