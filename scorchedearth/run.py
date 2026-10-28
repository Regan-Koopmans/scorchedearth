"""Run models against the benchmark and save results."""

import json
import time
from pathlib import Path

import numpy as np

from . import data, tasks
from .models import MODELS

RESULTS = Path("results")


def embed(fn, x: np.ndarray, batch: int = 64) -> tuple[np.ndarray, float]:
    """Embed in batches. Returns embeddings and wall-clock seconds, excluding a warmup batch."""
    fn(x[:batch])
    t = time.perf_counter()
    out = np.concatenate([fn(x[i : i + batch]) for i in range(0, len(x), batch)])
    return out, time.perf_counter() - t


def subsample(x: np.ndarray, y: np.ndarray, n: int | None, seed: int = 0):
    """Fixed random subset, identical for every model."""
    if n is None or n >= len(x):
        return x, y
    idx = np.sort(np.random.default_rng(seed).choice(len(x), n, replace=False))
    return x[idx], y[idx]


def run(model_names, limit=None, n_train=None, n_test=None) -> dict:
    train = subsample(*tasks.patchify(*data.load_split("training", limit=limit)), n_train)
    test = subsample(*tasks.patchify(*data.load_split("validation", limit=limit)), n_test)
    RESULTS.mkdir(exist_ok=True)

    out = {}
    for name in model_names:
        etr, ttr = embed(MODELS[name], train[0])
        ete, tte = embed(MODELS[name], test[0])
        scores = tasks.evaluate(etr, train[1], ete, test[1])
        n = len(etr) + len(ete)
        scores |= {
            "dim": etr.shape[1],
            "bytes_per_embedding": etr.shape[1] * 4,
            "patches_per_sec": round(n / (ttr + tte), 1),
            "ms_per_patch": round(1000 * (ttr + tte) / n, 3),
        }
        (RESULTS / f"{name}.json").write_text(json.dumps(scores, indent=2))
        out[name] = scores
        del etr, ete
        print(name, {k: round(v, 3) if isinstance(v, float) else v for k, v in scores.items()})
    return out
