import torch
import torch.nn as nn

from pydantic.dataclasses import dataclass

from experiments import Device, Dtype
from experiments.linear import LinearProjectionConfig, LinearProjection
from experiments.sonar import SonarNormalizer


@dataclass
class BaseLCMConfig:
    """
    A data model containing config parameters for the Base-LCM module.

    Args:
        norm (SonarNormalizer): a SonarNormalizer fit to a set of embeddings
        prenet_config (LinearProjectConfig): a model containing the PreNet layer config settings
        postnet_config (LinearProjectConfig): a model containing the PostNet layer config settings
    """

    norm: SonarNormalizer
    prenet_config: LinearProjectionConfig
    postnet_config: LinearProjectionConfig


class BaseLCM(nn.Module):
    """
    A Base-LCM architecture as described in this paper: https://arxiv.org/pdf/2412.08821.

    Args:
        config (BaseLCMConfig): a model containing the modules configuration settings
        dtype (torch.dtype, optional): a custom torch datatype for all tensors. Default is None
        device (torch.device, optional): the compute device to load tensors onto. Default is None
    """

    def __init__(
        self,
        config: BaseLCMConfig,
        dtype: Dtype | None = None,
        device: Device | None = None,
    ) -> None:
        super().__init__()

        self.config = config

        self.prenet = LinearProjection(config.prenet_config, dtype, device)
        self.postnet = LinearProjection(config.postnet_config, dtype, device)
        self.norm = config.norm

        if not self.norm._is_fitted:
            raise RuntimeError(
                "The normalizer must be `fit` to a set of embeddings first!"
            )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        pre_embeds = self.prenet(self.norm.normalize(embeddings))
        x = pre_embeds  # Update with decoder
        return self.norm.denormalize(self.postnet(x))
