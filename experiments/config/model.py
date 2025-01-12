from typing import Literal
from pydantic import BaseModel

from experiments import WeightInitLiteral, ActivationTypeLiteral
from experiments.lcm.transformer.decoder import DecoderOnlyLayerConfig


class PreNetSettings(BaseModel):
    """
    Base-LCM PreNet settings.

    Args:

    """


class PostNetSettings(BaseModel):
    """
    Base-LCM PostNet settings.

    Args:
        bias (bool, optional): a flag to enable layer bias terms. Default is True
        weight_norm (bool, optional): a flag to normalize the layers weights. Default is False
        weight_init_fn (WeightInitLiteral | None, optional): the type of weight initialization to use. Default is "kaiming_uniform"
        activation (ActivationTypeLiteral | None, optional): the type of activation function. Default is None
    """

    bias: bool = True
    weight_norm: bool = False
    weight_init_fn: WeightInitLiteral | None = "kaiming_uniform"
    activation: ActivationTypeLiteral | None = None


class PositionalEncodingSettings(BaseModel):
    """
    Positional encoding settings.

    Args:
        style (Literal["rope"]): the type of positional encoding
    """

    style: Literal["rope"]
    theta: float = 10_000.0


class LCMSettings(BaseModel):
    """
    LCM config settings.

    Args:
        d_model (int): number of input features
        n_layers (int): number of decoder layers
        n_attn_heads (int): number of attention heads

        dropout_p (float, optional): the dropout probability applied to the module output. Default is 0.0
        pre_linear_bias (bool, optional): a flag to enable or disable the PreNet layer's bias term. Default is True
        scale_embeddings (bool, optional): a flag for scaling the embeddings by `model_dim` before adding position embeddings (and before the PreNet). Default is False
        weight_normalization (bool, optional): a flag for normalizing module linear layers weights. Default is False
        embedding_std (float, optional): a custom standard deviation for the SONAR embeddings.
            Most SONAR embeddings have a distribution with the mean close to 0 and std close to 0.006. Initializing embedding-like parameters from a similar distribution
            is recommend to minimize their disruption during model training. Default is 0.006
    """

    d_model: int
    n_layers: int
    n_attn_heads: int

    prenet: PreNetSettings
    decoder: DecoderOnlyLayerConfig
    postnet: PostNetSettings

    # PreNet
    dropout_p: float = 0.0
    pre_linear_bias: bool = True
    scale_embeddings: bool = False
    weight_normalization: bool = False
    embedding_std: float = 0.006


class SonarSettings(BaseModel):
    """SONAR config settings."""

    std: float = 0.006
