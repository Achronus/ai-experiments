import torch
import torch.nn as nn


class GLUFeedForwardNetwork(nn.Module):
    """
    A multi-layer perceptron network with a gated linear unit (GLU).

    Args:
        d_model (int): the number of input features
        n_hidden (int): the number of hidden nodes in the layers
        inner_dim_scale (float, optional): the scale factor for the dimensionality of the inner projection layer. Default is 2 / 3
        inner_dim_to_multiple (int, optional): the rounded up value for the inner projection layer. Default is 1
        dropout_p (float, optional): the dropout probability. Default is 0.0
    """

    dropout: nn.Dropout | None

    def __init__(
        self,
        d_model: int,
        n_hidden: int,
        *,
        inner_dim_scale: float = 2 / 3,
        inner_dim_to_multiple: int = 1,
        dropout_p: float = 0.0,
    ) -> None:
        super().__init__(d_model)

        self.inner_dim_scale = inner_dim_scale
        self.inner_dim_to_multiple = inner_dim_to_multiple

        inner_dim = self._calc_inner_dim(n_hidden)

        self.gate_proj = nn.Sequential(
            nn.Linear(d_model, inner_dim),
            nn.SiLU(),
        )

        self.inner = nn.Linear(d_model, inner_dim)
        self.outer = nn.Linear(inner_dim, d_model)

        if dropout_p > 0.0:
            self.dropout = nn.Dropout(dropout_p)
        else:
            self.register_module("dropout", None)

    def _calc_inner_dim(self, n_hidden: int) -> int:
        """
        Dynamically calculates the `hidden_dim` for the linear layers.
        Used as an effective way to improve GLU efficiency and memory
        optimization for devices.
        """
        if self.inner_dim_scale != 1.0:
            inner_dim = int(n_hidden * self.inner_dim_scale)

        if self.inner_dim_to_multiple != 1:
            inner_dim = self.inner_dim_to_multiple * (
                (inner_dim + self.inner_dim_to_multiple - 1)
                // self.inner_dim_to_multiple
            )

        return inner_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs a forward pass through the network."""
        gate = self.gate_proj(x)
        x = self.inner(x)

        x = x * gate

        if self.dropout is not None:
            self.dropout(x)

        return self.outer(x)

    def extra_repr(self) -> str:
        """:meta private:"""
        s = super().extra_repr()

        return (
            f"{s}, "
            f"inner_dim_scale={self.inner_dim_scale:G}, "
            f"inner_dim_to_multiple={self.inner_dim_to_multiple}"
        )
