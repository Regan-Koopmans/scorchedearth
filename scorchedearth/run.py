"""Run models against benchmark tasks and save results.

Embeddings are cached per (task, model, split) under cache/, so re-scoring is cheap.
Results go to results/<task>/<model>.json.
"""

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from . import tasks
from .datasets import TASKS
from .models import MODELS

RESULTS = Path("results")
CACHE = Path("cache")


def embed(fn, x: np.ndarray, bands, batch: int = 64) -> tuple[np.ndarray, float]:
    """Embed in batches. Returns embeddings and wall-clock seconds, excluding a warmup batch."""
    fn(x[:batch], bands)
    t = time.perf_counter()
    out = np.concatenate([fn(x[i : i + batch], bands) for i in range(0, len(x), batch)])
    return out, time.perf_counter() - t


def subsample(x: np.ndarray, y: np.ndarray, n: int | None, seed: int = 0):
    """Fixed random subset, identical for every model."""
    if n is None or n >= len(x):
        return x, y
    idx = np.sort(np.random.default_rng(seed).choice(len(x), n, replace=False))
    return x[idx], y[idx]


def _fingerprint(x: np.ndarray) -> str:
    """Cheap content hash, so a cache entry is never reused for different patches."""
    h = hashlib.sha1(np.ascontiguousarray(x[:: max(1, len(x) // 64)]).tobytes())
    h.update(str(x.shape).encode())
    return h.hexdigest()[:12]


def cached_embed(task: str, model: str, split: str, x: np.ndarray, bands) -> tuple[np.ndarray, float | None]:
    path = CACHE / task / model / f"{split}-{_fingerprint(x)}.npz"
    if path.exists():
        z = np.load(path)
        return z["emb"], float(z["seconds"])
    emb, secs = embed(MODELS[model], x, bands)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, emb=emb, seconds=secs)
    return emb, secs


def run(task_names, model_names, limit=None, n_train=None, n_test=None) -> dict:
    out = {}
    for task_name in task_names:
        td = TASKS[task_name].load(limit)
        x_tr, y_tr = subsample(td.x_train, td.y_train, n_train)
        x_te, y_te = subsample(td.x_test, td.y_test, n_test)
        print(f"[{task_name}] {len(x_tr):,} train / {len(x_te):,} test patches, {x_tr.shape[1]} bands")
        (RESULTS / task_name).mkdir(parents=True, exist_ok=True)
        for name in model_names:
            etr, ttr = cached_embed(task_name, name, "train", x_tr, td.bands)
            ete, tte = cached_embed(task_name, name, "test", x_te, td.bands)
            scores = tasks.evaluate(etr, y_tr, ete, y_te, td.kind, td.threshold, td.skip)
            n = len(etr) + len(ete)
            scores |= {
                "dim": etr.shape[1],
                "bytes_per_embedding": etr.shape[1] * 4,
                "patches_per_sec": round(n / (ttr + tte), 1),
                "ms_per_patch": round(1000 * (ttr + tte) / n, 3),
            }
            (RESULTS / task_name / f"{name}.json").write_text(json.dumps(scores, indent=2))
            out[(task_name, name)] = scores
            print(f"  {name}", {k: round(v, 3) if isinstance(v, float) else v for k, v in scores.items()})
    return out
