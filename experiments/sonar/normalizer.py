from typing import Literal

import torch
import torch.nn as nn
import numpy as np

from pydantic.dataclasses import dataclass

from experiments import Device, Dtype


class FFTInterface:
    """
    A Fast Fourier Transform (FFT) interface.

    Duplicated from [Meta's LCM GitHub Repository](https://github.com/facebookresearch/large_concept_model/blob/main/lcm/models/sonar_normalizer/builder.py#L58).
    """

    @staticmethod
    def fft_transform(x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        if dtype in [torch.float16, torch.bfloat16]:
            x = x.to(dtype=torch.float32)
        x = torch.fft.rfft(x, norm="backward")
        return torch.concat([torch.real(x), torch.imag(x)[..., 1:-1]], dim=-1).to(dtype)

    @staticmethod
    def fft_inverse_transform(x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] % 2 == 0
        dtype = x.dtype
        if dtype in [torch.float16, torch.bfloat16]:
            x = x.to(dtype=torch.float32)
        rr, im = torch.split(
            x,
            [x.shape[-1] // 2 + 1, x.shape[-1] // 2 - 1],
            dim=-1,
        )
        im = torch.concat(
            [torch.zeros_like(im[..., :1]), im, torch.zeros_like(im[..., :1])], dim=-1
        )
        x = torch.fft.irfft(rr + im * 1j)
        return x.to(dtype)


@dataclass
class SonarNormalizerConfig:
    """
    A data model containing config parameters for the SonarNormalizer layer.

    Args:
        dim (int, optional): the dimension of the features to be normalized. Default is 1024
        clip_proba (float, optional): clips the features before normalizing between range `[clip_proba, 1 - clip_proba]`. Default is None
        with_fft (bool, optional): a flag to apply FFT transforms to the raw input before all other transforms. Default is False
        quantile_min (float, optional): the lower quantile used to measure the IQR when estimating the scale with a `robust` scaler. Default is 0.25
        quantile_max (float, optional): the upper quantile used to measure the IQR when estimating the scale with a `robust` scaler. Default is 0.75
        norm_method (Literal["standard", "robust", "gaussian_robust"], optional): the normalization technique to use. Default is gaussian_robust.
        (1) "standard": `center = mean`, `scale = std`
        (2) "robust": `center = median`, `scale = IQR = Qmax - Qmin`
        (3) "gaussian_robust": `center = median`, `scale = IQR / k`,
            where k = `stats.norm.ppf(q_max / 100.0) - stats.norm.ppf(q_min / 100.0)`
            i.e. `scale = scale = 0.7413 x IQR` if `q_min=0.25` and `q_max=0.75`.
            This is the robust normalization of https://arxiv.org/pdf/2307.05445.
    """

    dim: int = 1024
    clip_proba: float | None = None
    with_fft: bool = False
    quantile_min: float = 0.25
    quantile_max: float = 0.75
    norm_method: Literal["standard", "robust", "gaussian_robust"] = "gaussian_robust"


class SonarNormalizer(nn.Module, FFTInterface):
    """
    This SonarNormalizer follows the robust normalization introduced in
    https://arxiv.org/abs/2307.05445.

    Quoting from the paper: "Due to the very long-tailed feature distribution, typical mean and standard deviation statistics will be
    heavily biased. We thus propose a robust alternative based on the feature distribution quantiles. We
    take the median as the center of the distribution and approximate its scale using the Normalized
    InterQuartile Range (IQR) for a normal distribution: 0.7413 * IQR".

    Args:
        config (SonarNormalizerConfig): a model containing the layers configuration settings
        dtype (torch.dtype, optional): a custom torch datatype for all tensors. Default is None
        device (torch.device, optional): the compute device to load tensors onto. Default is None
    """

    center: torch.Tensor
    scale: torch.Tensor
    clip_min: torch.Tensor
    clip_max: torch.Tensor

    def __init__(
        self,
        config: SonarNormalizerConfig,
        dtype: Dtype | None = None,
        device: Device | None = None,
    ) -> None:
        super().__init__()

        self.config = config
        self.dtype = dtype
        self.device = device

        self._is_fitted = False

        self.register_buffer(
            "center", torch.zeros(self.config.dim, dtype=self.dtype, device=self.device)
        )
        self.register_buffer(
            "scale", torch.ones(self.config.dim, dtype=self.dtype, device=self.device)
        )

        if self.config.clip_proba is not None:
            self.register_buffer(
                "clip_min",
                torch.ones(self.config.dim, dtype=self.dtype, device=self.device),
            )
            self.register_buffer(
                "clip_max",
                torch.ones(self.config.dim, dtype=self.dtype, device=self.device),
            )

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        if not self._is_fitted:
            raise RuntimeError("The `fit` method must be called first.")

        if self.config.with_fft:
            x = self.fft_transform(x)

        x = (x - self.center) / self.scale

        if self.config.clip_proba is not None:
            x = torch.clamp(x, min=self.clip_min, max=self.clip_max)
        return x

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        if not self._is_fitted:
            raise RuntimeError("The `fit` method must be called first.")

        if self.config.clip_proba is not None:
            x = torch.clamp(x, min=self.clip_min, max=self.clip_max)

        x = (x * self.scale) + self.center

        if self.config.with_fft:
            x = self.fft_inverse_transform(x)
        return x

    def _quantile(self, x: np.ndarray, q: float, clip_t: torch.Tensor) -> torch.Tensor:
        """A helper method to compute the quantile tensors used in the `fit()` method."""
        return torch.quantile(torch.from_numpy(x), q=q, dim=0).to(
            dtype=clip_t.dtype, device=clip_t.device
        )

    @torch.no_grad()
    def fit(self, x: torch.Tensor) -> None:
        """Fits the normalizer to the data and calculates preliminary values."""
        if self.config.norm_method in ["robust", "gaussian_robust"]:
            from sklearn.preprocessing import RobustScaler

            _scaler = RobustScaler(
                unit_variance=self.config.norm_method == "gaussian_robust",
                quantile_range=(self.config.quantile_min, self.config.quantile_max),
            )
        elif self.config.norm_method == "standard":
            from sklearn.preprocessing import StandardScaler

            _scaler = StandardScaler()

        assert x.shape[-1] == self.config.dim
        assert len(x.shape) == 2

        if self.config.with_fft:
            x = self.fft_transform(x)

        x: np.ndarray = _scaler.fit_transform(x.cpu().float().numpy())

        _center: np.ndarray = (
            _scaler.mean_ if self.config.norm_method == "standard" else _scaler.center_
        )
        _scale: np.ndarray = _scaler.scale_

        self.center[:] = torch.tensor(
            _center,
            dtype=self.center.dtype,
            device=self.center.device,
        )

        self.scale[:] = torch.tensor(
            _scale,
            dtype=self.scale.dtype,
            device=self.scale.device,
        )

        if self.config.clip_proba is not None:
            self.clip_min[:] = self._quantile(x, self.config.clip_proba, self.clip_min)
            self.clip_max[:] = self._quantile(
                x,
                1 - self.config.clip_proba,
                self.clip_max,
            )

        self._is_fitted = True
