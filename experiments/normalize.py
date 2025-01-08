from typing import Callable, Literal

import torch.nn as nn
from pydantic import validate_call

from experiments import Dtype, Device


LayerNormLiteral = Literal["standard", "rms", "unit"]


@validate_call()
def build_layer_norm(style: LayerNormLiteral | None) -> Callable | None:
    """Creates a layer normalization function based on a given style."""
    if style is None:
        return None

    mapping = {
        "standard": build_standard_layer_norm,
        "rms": build_rms_layer_norm,
        "unit": build_unit_layer_norm,
    }

    return mapping[style]


def build_standard_layer_norm(
    model_dim: int,
    *,
    dtype: Dtype | None = None,
    device: Device | None = None,
) -> nn.LayerNorm:
    """Creates a PyTorch layer normalization module based on this paper: https://arxiv.org/abs/1607.06450."""
    return nn.LayerNorm(model_dim, bias=True, dtype=dtype, device=device)


def build_rms_layer_norm(
    model_dim: int,
    *,
    dtype: Dtype | None = None,
    device: Device | None = None,
) -> nn.RMSNorm:
    """Creates a PyTorch RMS layer normalization module based on this paper: https://arxiv.org/abs/1910.07467."""
    return nn.RMSNorm(model_dim, bias=False, dtype=dtype, device=device)


def build_unit_layer_norm(
    model_dim: int,
    *,
    dtype: Dtype | None = None,
    device: Device | None = None,
) -> nn.LayerNorm:
    """Creates a PyTorch layer normalization module without learnable mean and variance."""
    return nn.LayerNorm(
        model_dim,
        bias=False,
        elementwise_affine=False,
        dtype=dtype,
        device=device,
    )
