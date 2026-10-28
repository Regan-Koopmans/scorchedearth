"""Sen1Floods11 (Cloud to Street). Flood water, 11 events across 6 continents.

Hand-labelled 512x512 chips. Sentinel-2 L1C, 13 bands, scaled 0-10000.
Labels: 1 = water, 0 = not water, -1 = no data.
Split: the dataset's own train / test lists. Bolivia, held out by the authors, is not used.
Task: binary, patch is more than half water. Optical bands only; Sentinel-1 radar is not
used yet, so every model sees the same input.
"""

import tarfile
from pathlib import Path

import numpy as np
import tifffile
from huggingface_hub import hf_hub_download

from ..bands import S2_13, Bands
from ..tasks import TaskData, patchify

NAME = "floods"
REPO = "blumenstiel/Sen1Floods11"
DATA_DIR = Path("data/sen1floods11")
ROOT = DATA_DIR / "sen1floods11_v1.1"

# Per-band reflectance stats over the labelled training chips.
BANDS = Bands(
    tuple(S2_13),
    mean=(0.1500, 0.1253, 0.1227, 0.1062, 0.1335, 0.2298, 0.2770, 0.2558, 0.3007, 0.0459, 0.0062, 0.1935, 0.1070),
    std=(0.0328, 0.0352, 0.0375, 0.0532, 0.0452, 0.0687, 0.0882, 0.0851, 0.1010, 0.0294, 0.0147, 0.0809, 0.0604),
)


def download() -> Path:
    if (ROOT / "data").exists():
        return ROOT
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    archive = hf_hub_download(REPO, "sen1floods11_v1_1.tar.gz", repo_type="dataset")
    with tarfile.open(archive) as tar:
        tar.extractall(DATA_DIR, filter="data")
    return ROOT


def _ids(split: str) -> list[str]:
    return [l.strip() for l in (ROOT / "splits" / f"flood_{split}_data.txt").read_text().split() if l.strip()]


def load_split(split: str, limit: int | None = None):
    ids = _ids(split)[:limit]
    d = ROOT / "data"
    scenes = np.stack([tifffile.imread(d / "S2L1CHand" / f"{i}_S2Hand.tif").astype(np.float32) / 10000 for i in ids])
    masks = np.stack([tifffile.imread(d / "LabelHand" / f"{i}_LabelHand.tif").astype(np.int8) for i in ids])
    return scenes, masks


def event_years() -> dict[str, int]:
    """Year of each flood event, from the dataset's Sentinel-2 acquisition dates.
    Chip IDs prefixed "Mekong" belong to the event the metadata calls Cambodia."""
    import json

    md = json.loads((ROOT / "Sen1Floods11_Metadata.geojson").read_text())
    years = {f["properties"]["location"]: int(f["properties"]["s2_date"][:4]) for f in md["features"]}
    years["Mekong"] = years["Cambodia"]
    return years


def patch_footprints(split: str, limit: int | None = None):
    """Footprint, label and event year of every kept patch, in the same order as load().
    Footprints are (crs, west, south, east, north) in the chip's CRS (EPSG:4326)."""
    import rasterio

    from ..tasks import MAX_NODATA, PATCH, grid

    years = event_years()
    feet, labels, yrs = [], [], []
    split_name = {"training": "train", "validation": "test"}.get(split, split)
    for i in _ids(split_name)[:limit]:
        mask = tifffile.imread(ROOT / "data" / "LabelHand" / f"{i}_LabelHand.tif").astype(np.int8)
        with rasterio.open(ROOT / "data" / "S2L1CHand" / f"{i}_S2Hand.tif") as r:
            crs, t, h = r.crs, r.transform, r.height
        m = grid(mask[None])
        valid = m >= 0
        keep = (1 - valid.mean(axis=(1, 2))) <= MAX_NODATA
        frac = (m == 1).sum(axis=(1, 2)) / np.maximum(valid.sum(axis=(1, 2)), 1)
        g = h // PATCH
        for k in np.flatnonzero(keep):
            r_, c_ = divmod(int(k), g)
            w, n = t.c + c_ * PATCH * t.a, t.f + r_ * PATCH * t.e
            feet.append((crs, w, n + PATCH * t.e, w + PATCH * t.a, n))
            labels.append(float(frac[k]))
            yrs.append(years[i.split("_")[0]])
    return feet, np.array(labels, np.float32), np.array(yrs)


def load(limit: int | None = None) -> TaskData:
    x_tr, y_tr = patchify(*load_split("train", limit))
    x_te, y_te = patchify(*load_split("test", limit))
    return TaskData(NAME, BANDS, "binary", x_tr, y_tr, x_te, y_te)
