"""Benchmark tasks. Each scene is cut into square patches; embeddings are scored per patch.

- burned:   binary classification, patch is >50% burned (linear probe + kNN)
- fraction: regression of burned fraction (ridge probe)
- retrieval: precision@k of burned patches retrieving burned patches
"""

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import f1_score, r2_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

PATCH = 64
MAX_NODATA = 0.1


def patchify(scenes: np.ndarray, masks: np.ndarray, size: int = PATCH):
    """Cut scenes into non-overlapping patches. Drops patches with too much missing data."""
    n, c, h, w = scenes.shape
    g = h // size
    x = scenes[:, :, : g * size, : g * size].reshape(n, c, g, size, g, size)
    x = x.transpose(0, 2, 4, 1, 3, 5).reshape(-1, c, size, size)
    m = masks[:, : g * size, : g * size].reshape(n, g, size, g, size)
    m = m.transpose(0, 1, 3, 2, 4).reshape(-1, size, size)
    valid = m >= 0
    keep = (1 - valid.mean(axis=(1, 2))) <= MAX_NODATA
    frac = (m == 1).sum(axis=(1, 2)) / np.maximum(valid.sum(axis=(1, 2)), 1)
    return x[keep], frac[keep].astype(np.float32)


def _precision_at_k(emb: np.ndarray, y: np.ndarray, k: int = 10) -> float:
    pos = np.flatnonzero(y)
    if len(pos) == 0:
        return float("nan")
    e = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
    sims = e[pos] @ e.T
    sims[np.arange(len(pos)), pos] = -np.inf
    top = np.argpartition(-sims, k, axis=1)[:, :k]
    return float(y[top].mean())


def evaluate(train_emb, train_frac, test_emb, test_frac) -> dict:
    scaler = StandardScaler().fit(train_emb)
    xtr, xte = scaler.transform(train_emb), scaler.transform(test_emb)
    ytr, yte = train_frac > 0.5, test_frac > 0.5

    linear = LogisticRegression(max_iter=2000, class_weight="balanced").fit(xtr, ytr)
    knn = KNeighborsClassifier(n_neighbors=20).fit(xtr, ytr)
    ridge = Ridge(alpha=1.0).fit(xtr, train_frac)

    return {
        "linear_f1": f1_score(yte, linear.predict(xte)),
        "linear_auc": roc_auc_score(yte, linear.predict_proba(xte)[:, 1]),
        "knn_f1": f1_score(yte, knn.predict(xte)),
        "fraction_r2": r2_score(test_frac, ridge.predict(xte)),
        "retrieval_p@10": _precision_at_k(xte, yte),
        "n_train": len(ytr),
        "n_test": len(yte),
        "test_pos_rate": float(yte.mean()),
    }
