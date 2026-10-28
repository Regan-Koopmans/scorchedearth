"""Model registry. An embedder maps patches (N, 6, H, W) float32 to embeddings (N, D).

Pretrained models register themselves in foundation.py.
"""

from collections.abc import Callable

import numpy as np

Embedder = Callable[[np.ndarray], np.ndarray]
MODELS: dict[str, Embedder] = {}
