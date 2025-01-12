from typing import Self

import torch

from experiments import Device


class PaddingMask:
    """
    A sequence padding mask.

    Args:
        seq_lens (torch.Tensor): an array containing the length of a sequences. Shape must be `(batch_size)`
        batch_seq_len (int): the sequence length of the mask
    """

    def __init__(self, seq_lens: torch.Tensor, batch_seq_len: int) -> None:
        self.seq_lens = seq_lens
        self.batch_seq_len = batch_seq_len

        self.bool_tensor = None
        self.float_tensor = None

    def as_bool(self) -> torch.Tensor:
        """Creates the padding mask a boolean tensor."""
        if self.bool_tensor is None:
            self.bool_tensor = to_padding_mask(self.seq_lens, self.batch_seq_len)

        return self.bool_tensor

    def as_float(self, x: torch.Tensor) -> torch.Tensor:
        """Creates the padding mask as a float tensor."""
        if self.float_tensor is None:
            bool_mask = self.as_bool()

            # (batch_size, seq_len)
            mask = torch.zeros_like(bool_mask, dtype=x.dtype)
            self.float_tensor = torch.where(bool_mask, mask, -torch.inf)

        return self.float_tensor

    def trim(self, size: int) -> Self:
        """
        Return a new trimmed padding mask.

        Args:
            size (int): the sequence trim amount
        """
        return PaddingMask(self.seq_lens - size, self.batch_seq_len - size)

    def to(self, device: Device) -> Self:
        """
        Loads onto a device.

        Args:
            device (torch.device): the target device
        """
        if self.seq_lens.device == device:
            return self

        return PaddingMask(self.seq_lens.to(device), self.batch_seq_len)


def to_padding_mask(seq_lens: torch.Tensor, batch_seq_len: int) -> torch.Tensor:
    """
    Convert a sequence length array to a boolean padding mask tensor.

    Args:
        seq_lens (torch.Tensor): an array containing the length of a sequences. Shape must be `(batch_size)`
        batch_seq_len (int): the sequence length of the mask
    """
    batch_size = seq_lens.size(0)

    # (batch_size, seq_len)
    indices = torch.arange(batch_seq_len, device=seq_lens.device).expand(batch_size, -1)

    # (batch_size) -> (batch_size, seq_len)
    lengths = seq_lens.unsqueeze(1).expand(-1, batch_seq_len)

    return indices < lengths
