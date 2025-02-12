from collections import OrderedDict
from typing import Optional, Tuple
import numpy as np

import torch
import torch.nn as nn

from experiments.lnn.cell import NCPLiquidCell
from experiments.lnn.wiring import Wiring


class LiquidNCPNetwork(nn.Module):
    def __init__(
        self,
        in_features: int,
        n_neurons: int,
        out_features: int,
        *,
        sparsity_level: float = 0.5,
        seed: int = 64,
    ) -> None:
        """
        A Liquid Neural Circuit Policy network with three layers:
        1. Inter (includes sensory inputs)
        2. Command
        3. Motor (output)

        Parameters:
            in_features (int): number of inputs (sensory nodes)
            n_neurons (int): number of decision nodes (inter and command nodes).
                Nodes are set automatically based on the following:
                ```python
                command_neurons = max(int(0.4 * n_neurons), 1)
                inter_neurons = n_neurons - command_neurons
                ```
            out_features (int): number of out features (motor nodes)
            sparsity_level (float, optional): controls the connection sparsity
                between neurons. Must be a value between `[0.1, 0.9]`. When `0.1` neurons are very dense, when `0.9` they are very sparse. Default
                is '0.5'
            seed (int, optional): random seed for reproducibility. Default is '64'
        """
        super().__init__()

        torch.manual_seed(seed)
        np.random.seed(seed)

        self.in_features = in_features
        self.n_neurons = n_neurons
        self.out_features = out_features

        self.n_units = n_neurons + out_features  # inter + command + motor

        self._wiring = Wiring(
            in_features,
            n_neurons,
            out_features,
            sparsity_level=sparsity_level,
        )
        self._masks, self._counts = self._wiring.data()

        names = ["inter", "command", "motor"]
        layers = [
            NCPLiquidCell(
                in_features,
                self._counts.inter,
                self._masks.inter,
            ),
            NCPLiquidCell(
                self._counts.inter,
                self._counts.command,
                self._masks.command,
            ),
            NCPLiquidCell(
                self._counts.command,
                self._counts.motor,
                self._masks.motor,
            ),
        ]
        self.layers = OrderedDict([(name, layer) for name, layer in zip(names, layers)])

        self.ncp = nn.Sequential(self.layers)
        self._out_sizes = [layer.n_hidden for layer in self.layers.values()]

    def _ncp_forward(
        self, x: torch.Tensor, hidden: torch.Tensor, ts: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs a single timestep through the network layers.

        Splits the hidden state into respective chunks for each layer (out_features)
        to maintain their own independent hidden state dynamics. Then, merges them
        together to create a new hidden state.

        Parameters:
            x (torch.Tensor): the current batch of data for the timestep with
                shape: `(batch_size, features)`
            hidden (torch.Tensor): the current hidden state
            ts (float): the current time interval between events

        Returns:
            h,new_h_state (Tuple[torch.Tensor, torch.Tensor]): current hidden
            state (network prediction) and merged hidden state from all
            layers (updated state memory).
        """
        h_state = torch.split(hidden, self._out_sizes, dim=1)

        new_h_state = []

        # Handle layer independence
        for i, layer in enumerate(self.layers.values()):
            h = layer.forward(x, h_state[i], ts)
            x = h  # (batch_size, layer_out_features)
            new_h_state.append(h)

        new_h_state = torch.cat(new_h_state, dim=1)  # (batch_size, n_units)
        return h, new_h_state

    def _validate_timespans(self, timespans: torch.Tensor, seq_len: int) -> None:
        """A helper method to validate the timespan tensor."""
        if timespans.dim() != 1:
            raise ValueError(
                f"Timespans should be 1-dimensional, got: '{timespans.shape=}'"
            )

        if len(timespans) != seq_len:
            raise ValueError(
                f"Timespans length '{len(timespans)}' doesn't match: '{seq_len=}'"
            )

    def forward(
        self,
        x: torch.Tensor,
        h_state: Optional[torch.Tensor] = None,
        timespans: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs a forward pass through the network.

        Parameters:
            x (torch.Tensor): an input tensor of shape: `(batch_size,
                seq_len, features)` or `(batch_size, seq_len)`.
                When `x` is 2-dimensional, it is automatically expanded to
                `(batch_size, seq_len, 1)` within the method.
                - `batch_size` the number of samples per timestep.
                - `seq_len` the temporal dimension (e.g., timesteps,
                frames, tokens, audio samples).
                - `features` the features at each timestep (e.g.,
                image features, joint coordinates, word embeddings, raw amplitude
                values).
            h_state (torch.Tensor, optional): initial hidden state of the RNN with
                shape: `(batch_size, hidden_size)`. Default is 'None'.
                - `batch_size` the number of samples.
                - `hidden_size` the total number of hidden neurons.
            timespans (torch.Tensor, optional): a 1-dimensional tensor of shape
                `(seq_len,)`. Represents the time intervals between events.
                Used for event-based data. When `None` defaults to `1.0` for all
                timesteps. Default is 'None'
        Returns:
            y_pred,h_state (Tuple[torch.Tensor, torch.Tensor]): the network
            prediction and the final hidden state.
        """
        if x.dim() not in (2, 3):
            raise ValueError(
                f"Unsupported dimensionality: '{x.shape=}'. Should be 2 or 3 dimensional."
            )

        if x.dim() == 2:
            x = x.unsqueeze(-1)

        batch_size, seq_len, features = x.size()

        if timespans is not None:
            self._validate_timespans(timespans, seq_len)

        if h_state is None:
            h_state = torch.zeros((batch_size, self.n_units))

        output_sequence = []
        for t in range(seq_len):
            inputs = x[:, t]  # (batch_size, features)
            ts = 1.0 if timespans is None else timespans[t]

            h_out, h_state = self._ncp_forward(inputs, h_state, ts)
            output_sequence.append(h_out)

        # (batch_size, seq_len, out_features)
        y_pred = torch.stack(output_sequence, dim=1)
        return y_pred, h_state
