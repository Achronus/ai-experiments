from abc import abstractmethod
from typing import Callable, Self, override

import torch
import torch.nn as nn

from experiments import Device
from experiments.lcm.padding import PaddingMask


class PositionEncoder(nn.Module):
    """
    Encodes sequences with positional information.

    Args:
        encoding_dim (int): the number of position encodings. The last dimension of input sequences must have the same dimensionality
        max_seq_len (int | None): the maximum allowed length for input sequences. Often set to the context length of the model. If `None`, sequences can have arbitrary length
    """

    def __init__(self, encoding_dim: int, max_seq_len: int | None) -> None:
        super().__init__()

        self.encoding_dim = encoding_dim
        self.max_seq_len = max_seq_len

    def forward(self, x: torch.Tensor, mask: PaddingMask | None) -> torch.Tensor:
        """
        Returns a copy of `x` with encoded positional information.

        Args:
            x (torch.Tensor): the input sequences to encode. Must have shape `(batch_size, seq_len, encode_dim)`
            mask (PaddingMask | None): the padding mask of `x`. Must have shape `(batch_size, seq_len)`
        """
        if self.max_seq_len is not None:
            start_step = 0

            if (seq_len := start_step + x.size(-2)) > self.max_seq_len:
                raise ValueError(
                    f"The input sequence length ({seq_len}) must be less than or equal to the max sequence length ({self.max_seq_len})."
                )

        return self._forward(x, mask)

    @abstractmethod
    def _forward(self, x: torch.Tensor, mask: PaddingMask | None) -> torch.Tensor:
        """Performs a forward pass through the encoder."""
        pass  # pragma: no cover


class RotaryEncoder(PositionEncoder):
    """
    Encodes sequences with relative positional information based on the paper: https://doi.org/10.48550/arxiv.2104.09864.

    Args:
        encoding_dim (int): the number of position encodings. The last dimension of input sequences must have the same dimensionality
        max_seq_len (int | None): the maximum allowed length for input sequences. Often set to the context length of the model. If `None`, sequences can have arbitrary length
        theta (float, optional): the coefficient of the long-term decay. Default is 10,000
        freqs_init_fn (Callable | None): a callable to initialize the frequency table. The encoder is passed as an argument and must return a `torch.Tensor` holding the frequency table. If `None`, frequencies are initialized as described in the paper. Default is None
        device (torch.device, optional): the compute device to load tensors onto. Default is None
    """

    freqs: torch.Tensor

    def __init__(
        self,
        encoding_dim: int,
        max_seq_len: int | None,
        *,
        theta: float = 10_000.0,
        freqs_init_fn: Callable[[Self], torch.Tensor] | None = None,
        device: Device | None = None,
    ) -> None:
        super().__init__(encoding_dim, max_seq_len)

        self.encoding_dim = encoding_dim
        self.max_seq_len = max_seq_len

        self.theta = theta
        self.freqs_init_fn = freqs_init_fn
        self.device = device

        if encoding_dim % 2 != 0:
            raise ValueError(f"'encoding_dim={encoding_dim}', but must be even.")

        freqs = torch.empty(
            (max_seq_len, encoding_dim // 2, 2),
            device=device,
            dtype=torch.float32,
        )

        self.register_buffer("freqs", freqs, persistent=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Resets the non-persistent buffers."""
        if self.max_seq_len is None:
            raise ValueError("'max_seq_len' is 'None'.")

        device = self.freqs.device
        complex_freqs = torch.view_as_complex(self.freqs)

        # (S)
        steps = torch.arange(self.max_seq_len, device=device, dtype=torch.float32)

        if self.freqs_init_fn is None:
            # (E / 2)
            indices = torch.arange(
                0, self.encoding_dim, step=2, device=device, dtype=torch.float32
            )
            freqs = 1.0 / (self.theta ** (indices / self.encoding_dim))
        else:
            freqs = self.freqs_init_fn(self)

        # (S) x (E / 2) -> (S, E / 2)
        freqs = torch.outer(steps, freqs)

        # (S, E / 2)
        torch.polar(torch.ones_like(freqs), freqs, out=complex_freqs)

    @override
    def _forward(self, x: torch.Tensor, mask: PaddingMask | None) -> torch.Tensor:
        seq_len = x.size(-2)
        start_step = 0

        complex_freqs = torch.view_as_complex(self.freqs)
        complex_freqs = complex_freqs[start_step : start_step + seq_len]

        # (*, S, E) -> (*, S, E / 2, 2)
        x = x.unflatten(-1, (-1, 2))

        complex_x = torch.view_as_complex(x.float())
        complex_x = complex_x * complex_freqs

        # (*, S, E / 2, 2) -> (*, S , E)
        fp32_x = torch.view_as_real(complex_x).flatten(-2)
        return fp32_x.type_as(x)
