"""Shared task machinery: patch cutting, label rules, probe scoring.

Each dataset module (scorchedearth/datasets/) returns a TaskData: train and test patches,
the bands they hold, and per-patch targets. Scoring depends only on the label kind:

- binary:     positive if the patch's target fraction exceeds `threshold`.
              Linear and kNN F1, AUC, fraction R^2, retrieval P@10. A task can switch off
              metrics that don't fit it via `skip`.
- multiclass: target is the patch's majority class. Linear and kNN macro F1, accuracy.
"""

from dataclasses import dataclass, field

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, r2_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from .bands import Bands

PATCH = 64
MAX_NODATA = 0.1


@dataclass
class TaskData:
    name: str
    bands: Bands
    kind: str  # "binary" or "multiclass"
    x_train: np.ndarray  # (N, C, PATCH, PATCH) float32 reflectance
    y_train: np.ndarray  # binary: target fraction in [0, 1]; multiclass: class id
    x_test: np.ndarray
    y_test: np.ndarray
    threshold: float = 0.5  # binary only: fraction above which a patch is positive
    classes: list[str] = field(default_factory=list)  # multiclass only
    skip: tuple[str, ...] = ()  # metrics that don't fit this task, e.g. "knn_f1"


def grid(a: np.ndarray, size: int = PATCH) -> np.ndarray:
    """Cut (N, ..., H, W) into non-overlapping (N * g * g, ..., size, size) tiles."""
    n, *lead, h, w = a.shape
    g = h // size
    a = a[..., : g * size, : g * size].reshape(n, *lead, g, size, g, size)
    k = len(lead)
    order = (0, k + 1, k + 3, *range(1, k + 1), k + 2, k + 4)
    return a.transpose(order).reshape(-1, *lead, size, size)


def patchify(scenes: np.ndarray, masks: np.ndarray, size: int = PATCH, positive: int = 1):
    """Binary tasks. Masks use -1 for no data. Returns patches and positive-class fraction,
    dropping patches with more than MAX_NODATA missing."""
    x, m = grid(scenes, size), grid(masks, size)
    valid = m >= 0
    keep = (1 - valid.mean(axis=(1, 2))) <= MAX_NODATA
    frac = (m == positive).sum(axis=(1, 2)) / np.maximum(valid.sum(axis=(1, 2)), 1)
    return x[keep], frac[keep].astype(np.float32)


def patchify_majority(scenes: np.ndarray, masks: np.ndarray, size: int = PATCH, nodata: int = 0, min_share: float = 0.5):
    """Multiclass tasks. Returns patches and their majority class, keeping patches where
    that class covers at least `min_share` of valid pixels."""
    x, m = grid(scenes, size), grid(masks, size).reshape(-1, size * size)
    keep, label = [], []
    for row in m:
        v = row[row != nodata]
        if len(v) < (1 - MAX_NODATA) * row.size:
            keep.append(False), label.append(-1)
            continue
        vals, counts = np.unique(v, return_counts=True)
        top = counts.argmax()
        ok = counts[top] >= min_share * len(v)
        keep.append(ok), label.append(int(vals[top]) if ok else -1)
    keep = np.array(keep)
    return x[keep], np.array(label)[keep]


def _precision_at_k(emb: np.ndarray, y: np.ndarray, k: int = 10) -> float:
    pos = np.flatnonzero(y)
    if len(pos) == 0:
        return float("nan")
    e = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
    sims = e[pos] @ e.T
    sims[np.arange(len(pos)), pos] = -np.inf
    top = np.argpartition(-sims, k, axis=1)[:, :k]
    return float(y[top].mean())


def evaluate(train_emb, y_train, test_emb, y_test, kind: str = "binary", threshold: float = 0.5, skip=()) -> dict:
    scaler = StandardScaler().fit(train_emb)
    xtr, xte = scaler.transform(train_emb), scaler.transform(test_emb)

    if kind == "multiclass":
        linear = LogisticRegression(max_iter=2000, class_weight="balanced").fit(xtr, y_train)
        knn = KNeighborsClassifier(n_neighbors=20).fit(xtr, y_train)
        pred_lin, pred_knn = linear.predict(xte), knn.predict(xte)
        return {
            "linear_f1": f1_score(y_test, pred_lin, average="macro"),
            "knn_f1": f1_score(y_test, pred_knn, average="macro"),
            "linear_acc": accuracy_score(y_test, pred_lin),
            "n_train": len(y_train),
            "n_test": len(y_test),
            "n_classes": int(len(np.unique(y_train))),
        }

    ytr, yte = y_train > threshold, y_test > threshold
    for split, y in (("train", ytr), ("test", yte)):
        if y.all() or not y.any():
            raise ValueError(f"{split} split has only one class ({int(y.sum())} of {len(y)} positive). "
                             "Use more data: raise --limit or --n-train / --n-test.")
    linear = LogisticRegression(max_iter=2000, class_weight="balanced").fit(xtr, ytr)
    out = {
        "linear_f1": f1_score(yte, linear.predict(xte)),
        "linear_auc": roc_auc_score(yte, linear.predict_proba(xte)[:, 1]),
    }
    if "knn_f1" not in skip:
        out["knn_f1"] = f1_score(yte, KNeighborsClassifier(n_neighbors=20).fit(xtr, ytr).predict(xte))
    if "fraction_r2" not in skip:
        out["fraction_r2"] = r2_score(y_test, Ridge(alpha=1.0).fit(xtr, y_train).predict(xte))
    out["retrieval_p@10"] = _precision_at_k(xte, yte)
    return out | {"n_train": len(ytr), "n_test": len(yte), "test_pos_rate": float(yte.mean())}
