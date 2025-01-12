from experiments.lcm.config import ExperimentConfig
from experiments.lcm.sonar import SonarNormalizer
from experiments.lcm import BaseLCM
from experiments.lcm.transformer import TransformerDecoder, DecoderOnlyLayer

import torch


def main(ds_mixture: list[str], save_path: str) -> None:
    """Performs the basic pipeline for using the LCM model."""
    config = ExperimentConfig()

    embeddings: torch.Tensor = []

    normalizer = SonarNormalizer(config.sonar)
    normalizer.fit(embeddings)

    torch.save(
        {
            "normalizer": normalizer.state_dict(),
            "dataset_mixture": ds_mixture,
        },
        save_path,
    )

    print(f"Normalizer saved to: {save_path}")

    decoder = TransformerDecoder(
        DecoderOnlyLayer(
            config.d_model,
            config.n_attn_heads,
            config.decoder,
        ),
        config.num_attn_heads,
    )

    lcm = BaseLCM(normalizer, config.lcm)
    y_pred = lcm.forward(embeddings)


if __name__ == "__main__":
    ds_mixture = []
    save_path = "saved/normalizer.pt"

    main(ds_mixture, save_path)
