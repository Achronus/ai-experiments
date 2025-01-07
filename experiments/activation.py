from enum import Enum
from typing import Literal, Self

import torch.nn as nn


ActivationTypeLiteral = Literal[
    "relu",
    "tanh",
    "elu",
    "leaky_relu",
    "prelu",
    "selu",
    "silu",
    "softsign",
    "sigmoid",
    "hardsigmoid",
]


class ActivationEnum(Enum):
    """An Enum for PyTorch activation functions."""

    RELU = nn.ReLU()
    TANH = nn.Tanh()
    ELU = nn.ELU()
    LEAKY_RELU = nn.LeakyReLU()
    PRELU = nn.PReLU()
    SELU = nn.SELU()
    SILU = nn.GELU()
    SOFTSIGN = nn.Softsign()
    SIGMOID = nn.Sigmoid()
    HARDSIGMOID = nn.Hardsigmoid()

    @classmethod
    def get(cls, name: str | Self) -> nn.Module:
        """Get the activation function."""
        if isinstance(name, cls):
            return name.value
        try:
            return cls[name.upper()].value
        except KeyError:
            raise ValueError(f"Unsupported activation function: {name}")
