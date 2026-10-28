"""Multi-temporal crop classification (IBM / NASA, Clark University). US land cover, 2022.

224x224 HLS chips, three dates x six bands, scaled 0-10000.
Labels: 13 classes from the USDA Cropland Data Layer; 0 = no data.
Split: the dataset's training / validation chips.
Task: multiclass, majority class of each 64x64 patch. Fields are small at 30 m, so a patch
is kept when its majority class covers at least 30% of it (median share is ~42%).
Only the middle date is used, so every model sees one image; multi-date input would
favour time-series models.
"""

import tarfile
from pathlib import Path

import numpy as np
import tifffile
from huggingface_hub import hf_hub_download

from ..bands import HLS, Bands
from ..tasks import TaskData, patchify_majority

NAME = "crop_type"
REPO = "ibm-nasa-geospatial/multi-temporal-crop-classification"
DATA_DIR = Path("data/crop")
DATE = 1  # middle of three observations
MIN_SHARE = 0.3

CLASSES = ["no data", "natural vegetation", "forest", "corn", "soybeans", "wetlands", "developed/barren",
           "open water", "winter wheat", "alfalfa", "fallow/idle cropland", "cotton", "sorghum", "other"]

# Middle-date reflectance stats over all 3,083 training chips.
BANDS = Bands(
    tuple(HLS),
    mean=(0.0506, 0.0852, 0.0924, 0.3273, 0.2691, 0.1761),
    std=(0.0301, 0.0381, 0.0622, 0.0843, 0.1017, 0.1005),
)


def download() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for split in ("training", "validation"):
        if not (DATA_DIR / f"{split}_chips").exists():
            archive = hf_hub_download(REPO, f"{split}_chips.tgz", repo_type="dataset")
            with tarfile.open(archive) as tar:
                tar.extractall(DATA_DIR, filter="data")
    return DATA_DIR


def chip_paths(split: str, limit: int | None = None) -> list[tuple[Path, Path]]:
    """Image and mask path for each chip, in load order."""
    d = DATA_DIR / f"{split}_chips"
    # The training archive carries macOS "._" sidecar files beside every chip; skip them.
    imgs = [p for p in sorted(d.glob("*_merged.tif")) if not p.name.startswith(".")]
    pairs = [(p, d / p.name.replace("_merged.tif", ".mask.tif")) for p in imgs]
    return [(i, m) for i, m in pairs if m.exists()][:limit]  # tolerate partial downloads


def load_split(split: str, limit: int | None = None):
    pairs = chip_paths(split, limit)
    if not pairs:
        raise FileNotFoundError(f"No crop chips in {DATA_DIR / f'{split}_chips'}. Run `download` first.")
    scenes, masks = [], []
    for img, mask in pairs:
        a = tifffile.imread(img)  # H, W, 18
        a = np.moveaxis(a, -1, 0).astype(np.float32) / 10000
        scenes.append(a[DATE * 6 : DATE * 6 + 6])
        masks.append(tifffile.imread(mask).astype(np.int16))
    return np.stack(scenes), np.stack(masks)


YEAR = 2022


def patch_footprints(split: str, limit: int | None = None):
    """Footprint and label of every kept patch, in the same order as load(). Each footprint
    is (crs, west, south, east, north) in the chip's own CRS, for precomputed lookups."""
    import rasterio

    from ..tasks import PATCH, majority

    feet, labels = [], []
    for img, mask in chip_paths(split, limit):
        keep, lab = majority(tifffile.imread(mask).astype(np.int16)[None], min_share=MIN_SHARE)
        with rasterio.open(img) as r:
            crs, t = r.crs, r.transform
        g = r.height // PATCH
        for k in np.flatnonzero(keep):
            i, j = divmod(int(k), g)  # grid() orders tiles row by row
            w, n = t.c + j * PATCH * t.a, t.f + i * PATCH * t.e
            feet.append((crs, w, n + PATCH * t.e, w + PATCH * t.a, n))
            labels.append(int(lab[k]))
    return feet, np.array(labels)


def load(limit: int | None = None) -> TaskData:
    x_tr, y_tr = patchify_majority(*load_split("training", limit), min_share=MIN_SHARE)
    x_te, y_te = patchify_majority(*load_split("validation", limit), min_share=MIN_SHARE)
    return TaskData(NAME, BANDS, "multiclass", x_tr, y_tr, x_te, y_te, classes=CLASSES)
