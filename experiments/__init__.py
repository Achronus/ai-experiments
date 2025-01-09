import torch

from experiments.activation import ActivationEnum, ActivationTypeLiteral
from experiments.normalize import LayerNormLiteral, LayerNorm
from experiments.weight import WeightInitLiteral, get_init_fn


Dtype = torch.dtype
"""Type alias for `torch.dtype`."""

Device = torch.device
"""Type alias for `torch.device`."""


__all__ = [
    "ActivationEnum",
    "ActivationTypeLiteral",
    "LayerNormLiteral",
    "LayerNorm",
    "WeightInitLiteral",
    "get_init_fn",
]
