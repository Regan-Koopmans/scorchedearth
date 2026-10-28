"""Baselines: embeddings with no learned weights. Any foundation model must beat these.

Each takes patches (N, C, H, W) in reflectance and the task's Bands, like an Adapter.
"""

import numpy as np

from .bands import NIR, NIR_NARROW, RED, SWIR2, Bands
from .models import MODELS


def register(name: str):
    def wrap(fn):
        MODELS[name] = fn
        return fn

    return wrap


@register("band_stats")
def band_stats(x: np.ndarray, bands: Bands) -> np.ndarray:
    """Per-band mean and std. 2 x bands dims."""
    return np.concatenate([x.mean(axis=(2, 3)), x.std(axis=(2, 3))], axis=1)


def _nir(x: np.ndarray, bands: Bands) -> np.ndarray:
    """Narrow NIR (B8A) if the task has it, else broad NIR (B8)."""
    i = bands.index(NIR_NARROW)
    return x[:, i if i is not None else bands.nearest([NIR])[0]]


@register("spectral_indices")
def spectral_indices(x: np.ndarray, bands: Bands) -> np.ndarray:
    """Mean and std of NDVI and NBR, plus band means. 4 + bands dims."""
    nir, red, swir2 = _nir(x, bands), x[:, bands.nearest([RED])[0]], x[:, bands.nearest([SWIR2])[0]]
    ndvi = (nir - red) / (nir + red + 1e-6)
    nbr = (nir - swir2) / (nir + swir2 + 1e-6)
    idx = np.stack([ndvi, nbr], axis=1)
    return np.concatenate([idx.mean(axis=(2, 3)), idx.std(axis=(2, 3)), x.mean(axis=(2, 3))], axis=1)


@register("random_projection")
def random_projection(x: np.ndarray, bands: Bands, dim: int = 256, seed: int = 0) -> np.ndarray:
    """Random linear projection of 8x8-downsampled pixels. Sanity floor."""
    n, c, h, w = x.shape
    small = x.reshape(n, c, 8, h // 8, 8, w // 8).mean(axis=(3, 5)).reshape(n, -1)
    proj = np.random.default_rng(seed).standard_normal((small.shape[1], dim)) / np.sqrt(dim)
    return small @ proj
