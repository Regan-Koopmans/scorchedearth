"""Embedding models. An embedder maps patches (N, 6, H, W) float32 to embeddings (N, D).

Register new models with @register. Baselines set the floor any learned model must beat.
"""

from collections.abc import Callable

import numpy as np

Embedder = Callable[[np.ndarray], np.ndarray]
MODELS: dict[str, Embedder] = {}


def register(name: str):
    def wrap(fn: Embedder) -> Embedder:
        MODELS[name] = fn
        return fn

    return wrap


@register("band_stats")
def band_stats(x: np.ndarray) -> np.ndarray:
    """Per-band mean and std. 12 dims."""
    return np.concatenate([x.mean(axis=(2, 3)), x.std(axis=(2, 3))], axis=1)


@register("spectral_indices")
def spectral_indices(x: np.ndarray) -> np.ndarray:
    """Mean and std of NDVI and NBR, plus band means. 10 dims."""
    blue, green, red, nir, swir1, swir2 = x.transpose(1, 0, 2, 3)
    ndvi = (nir - red) / (nir + red + 1e-6)
    nbr = (nir - swir2) / (nir + swir2 + 1e-6)
    idx = np.stack([ndvi, nbr], axis=1)
    return np.concatenate(
        [idx.mean(axis=(2, 3)), idx.std(axis=(2, 3)), x.mean(axis=(2, 3))], axis=1
    )


@register("random_projection")
def random_projection(x: np.ndarray, dim: int = 256, seed: int = 0) -> np.ndarray:
    """Random linear projection of downsampled pixels. Sanity check."""
    n, c, h, w = x.shape
    small = x.reshape(n, c, 8, h // 8, 8, w // 8).mean(axis=(3, 5)).reshape(n, -1)
    proj = np.random.default_rng(seed).standard_normal((small.shape[1], dim)) / np.sqrt(dim)
    return small @ proj

