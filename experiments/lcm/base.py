import torch
import torch.nn as nn

from experiments import Device, Dtype
from experiments.config import LCMSettings
from experiments.lcm.linear import LinearProjection
from experiments.lcm.sonar import SonarNormalizer


class BaseLCM(nn.Module):
    """
    A Base-LCM architecture as described in this paper: https://arxiv.org/pdf/2412.08821.

    Args:
        norm (SonarNormalizer): a SonarNormalizer fit to a set of embeddings
        config (LCMSettings): a model containing the modules configuration settings
        dtype (torch.dtype, optional): a custom torch datatype for all tensors. Default is None
        device (torch.device, optional): the compute device to load tensors onto. Default is None
    """

    def __init__(
        self,
        norm: SonarNormalizer,
        config: LCMSettings,
        dtype: Dtype | None = None,
        device: Device | None = None,
    ) -> None:
        super().__init__()

        self.config = config

        self.prenet = PreNet(config.prenet, dtype, device)
        self.postnet = LinearProjection(
            in_features,
            out_features,
            config.postnet,
            dtype,
            device,
        )
        self.norm = norm

        if not self.norm._is_fitted:
            raise RuntimeError(
                "The normalizer must be `fit` to a set of embeddings first!"
            )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        pre_embeds = self.prenet(self.norm.normalize(embeddings))
        x = pre_embeds  # Update with decoder
        return self.norm.denormalize(self.postnet(x))
