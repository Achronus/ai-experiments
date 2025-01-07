from pathlib import Path
from typing import Any, Type
import yaml

from pydantic import BaseModel
from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
    PydanticBaseSettingsSource,
    YamlConfigSettingsSource,
)

from experiments.config import LCMSettings
from experiments.sonar import SonarNormalizerConfig


class ExperimentConfig(BaseSettings):
    """Experiment configuration settings. Automatically loaded from the local `config.yaml` file."""

    lcm: LCMSettings | None = None
    sonar: SonarNormalizerConfig = SonarNormalizerConfig()

    model_config = SettingsConfigDict(
        extra="ignore",
        yaml_file="config.yaml",
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: Type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        yaml_file = settings_cls.model_config.get("yaml_file")

        if not Path(yaml_file).exists():
            raise FileNotFoundError(
                f"'{yaml_file}' does not exist in the local directory."
            )

        return (YamlConfigSettingsSource(settings_cls),)


def load_yaml(filepath: Path | str) -> dict[str, Any]:
    """Loads a YAML file as a dictionary."""
    if not Path(filepath).exists():
        raise FileNotFoundError("File does not exist.")

    yaml_file = str(filepath).split(".")[-1] == "yaml"

    if not yaml_file:
        raise RuntimeError("Incorrect file type provided. Must be a 'yaml' file.")

    with open(filepath, "r") as f:
        yaml_config = yaml.safe_load(f)

    return yaml_config


def load_config(filepath: Path | str) -> ExperimentConfig:
    """(Deprecated) loads a YAML file as an ExperimentConfig model."""
    yaml_config = load_yaml(filepath)
    return ExperimentConfig(**yaml_config)
