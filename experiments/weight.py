from typing import Callable, Literal
import math

from pydantic import validate_call
import torch.nn as nn


WeightInitLiteral = Literal[
    "xavier",
    "sonar",
    "zero",
    "trunc_normal",
    "kaiming_uniform",
]


@validate_call()
def get_init_fn(style: WeightInitLiteral | None) -> Callable | None:
    """Retrieve a weight initialization function based on a given style."""
    if style is None:
        return None

    mapping = {
        "xavier": init_linear_xavier,
        "sonar": init_linear_sonar,
        "zero": init_linear_zero,
        "trunc_normal": init_linear_trunc_normal,
        "kaiming_uniform": init_linear_kaiming_uniform,
    }

    return mapping[style]


def init_linear_xavier(layer: nn.Module) -> None:
    """Performs Xavier weight initialization as described in this paper: https://proceedings.mlr.press/v9/glorot10a.html."""
    nn.init.xavier_normal(layer.weight)

    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


def init_linear_sonar(layer: nn.Module, sonar_std: float = 0.006) -> None:
    """Performs SONAR weight initialization as described in this paper: https://arxiv.org/abs/2308.11466."""
    std = sonar_std * (3 / layer.in_features) ** 0.5
    nn.init.uniform(layer.weight, a=-std, b=std)

    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


def init_linear_zero(layer: nn.Module) -> None:
    """Performs ZerO weight initialization as described in this paper: https://arxiv.org/abs/2110.12661."""
    nn.init.zeros_(layer.weight)

    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


def init_linear_trunc_normal(layer: nn.Module) -> None:
    """Performs Truncated Normal weight initialization."""
    nn.init.trunc_normal_(layer.weight, std=1e-3)

    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


def init_linear_kaiming_uniform(layer: nn.Module) -> None:
    """Performs Kaiming Uniform weight initialization as described in this paper: https://arxiv.org/abs/1502.01852."""
    nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))

    if layer.bias is not None:
        fan_in = layer.weight.size(1)

        m = 1
        if layer.weight.ndim > 2:
            for size in layer.weight.shape[2:]:
                m *= size

        fan_in *= m

        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(layer.bias, -bound, bound)
