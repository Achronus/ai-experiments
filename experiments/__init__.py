import torch

from experiments.activation import ActivationEnum, ActivationTypeLiteral
from experiments.weight import WeightInitLiteral, get_init_fn


Dtype = torch.dtype
"""Type alias for `torch.dtype`."""

Device = torch.device
"""Type alias for `torch.device`."""


__all__ = [
    "ActivationEnum",
    "ActivationTypeLiteral",
    "WeightInitLiteral",
    "get_init_fn",
]
