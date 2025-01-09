from typing import Literal

import torch
import torch.nn as nn

from experiments import Dtype, Device


LayerNormLiteral = Literal["standard", "rms", "unit"]


class LayerNorm(nn.Module):
    """
    A custom layer normalization module that supports multiple normalization styles.

    Model style options:
    - Standard - based on the paper https://arxiv.org/abs/1607.06450.
    - RMS - based on the paper https://arxiv.org/abs/1910.07467.
    - Unit - standard without learnable mean and variance.

    Args:
        d_model (int): the number of input features
        style (LayerNormLiteral, optional): the type of layer normalization to use. Default is "standard"
        dtype (torch.dtype, optional): a custom torch datatype for all tensors. Default is None
        device (torch.device, optional): the compute device to load tensors onto. Default is None
    """

    def __init__(
        self,
        d_model: int,
        style: LayerNormLiteral = "standard",
        *,
        dtype: Dtype | None = None,
        device: Device | None = None,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.style = style

        if style == "standard":
            self.norm = nn.LayerNorm(d_model, bias=True, dtype=dtype, device=device)
        elif style == "rms":
            self.norm = nn.RMSNorm(d_model, bias=False, dtype=dtype, device=device)
        elif style == "unit":
            self.norm = nn.LayerNorm(
                d_model,
                bias=False,
                elementwise_affine=False,
                dtype=dtype,
                device=device,
            )
        else:
            raise ValueError(f"Unknown normalization style: '{style}'")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalizes the input tensor."""
        self.norm(x)

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, style={self.style}"
