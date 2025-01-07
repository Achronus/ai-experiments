from pydantic import BaseModel

from experiments.linear import LinearProjectionConfig


class LCMSettings(BaseModel):
    """LCM config settings."""

    prenet: LinearProjectionConfig
    postnet: LinearProjectionConfig
