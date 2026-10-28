"""Landslide4Sense (IARAI). Landslides from four regions, Sentinel-2 plus terrain.

128x128 chips at ~10 m. 14 bands: Sentinel-2 B1-B12 without B8A, then slope and DEM.
The release divides each band by a per-band constant; the S2 bands are rescaled back to
reflectance using factors fitted to the training set. Slope and DEM are dropped.
Labels: 1 = landslide, 0 = not.
Split: the dataset's train / validation lists.
Task: binary. Landslides are small, so a 64x64 patch is positive if any pixel is landslide.
Positive patches are still mostly ground, so fraction R^2 carries no signal and is not reported.
"""

from pathlib import Path

import h5py
import numpy as np
from huggingface_hub import snapshot_download

from ..bands import S2_BY_NAME, Bands
from ..tasks import TaskData, patchify

NAME = "landslides"
REPO = "ibm-nasa-geospatial/Landslide4sense"
DATA_DIR = Path("data/landslide4sense")

S2_NAMES = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9", "B10", "B11", "B12"]

# Release value x SCALE = reflectance. Fitted so per-band training medians match
# Sentinel-2 L1C medians from Sen1Floods11's training chips. An approximation: the
# release's true divisors are not published.
SCALE = (0.1493, 0.1215, 0.1131, 0.0935, 0.1130, 0.2238, 0.2766, 0.2512, 0.0265, 0.0013, 0.1831, 0.0924)

BANDS = Bands(
    tuple(S2_BY_NAME[n] for n in S2_NAMES),
    mean=(0.1423, 0.1175, 0.1150, 0.0972, 0.1273, 0.2375, 0.2853, 0.2633, 0.0377, 0.0022, 0.2061, 0.1054),
    std=(0.0186, 0.0198, 0.0261, 0.0326, 0.0368, 0.0691, 0.0886, 0.0853, 0.0088, 0.0005, 0.0859, 0.0528),
)


def download() -> Path:
    snapshot_download(REPO, repo_type="dataset", local_dir=DATA_DIR)
    return DATA_DIR


def load_split(split: str, limit: int | None = None):
    imgs = sorted((DATA_DIR / "images" / split).glob("image_*.h5"), key=lambda p: int(p.stem.split("_")[1]))[:limit]
    scenes, masks = [], []
    scale = np.array(SCALE, np.float32)[:, None, None]
    for p in imgs:
        img = h5py.File(p)["img"][()].astype(np.float32)  # H, W, 14
        scenes.append(np.moveaxis(img[..., :12], -1, 0) * scale)
        mask_path = DATA_DIR / "annotations" / split / p.name.replace("image_", "mask_")
        masks.append(h5py.File(mask_path)["mask"][()].astype(np.int8))
    if not scenes:
        raise FileNotFoundError(f"No Landslide4Sense chips in {DATA_DIR / 'images' / split}. Run `download` first.")
    return np.stack(scenes), np.stack(masks)


def load(limit: int | None = None) -> TaskData:
    x_tr, y_tr = patchify(*load_split("train", limit))
    x_te, y_te = patchify(*load_split("validation", limit))
    return TaskData(NAME, BANDS, "binary", x_tr, y_tr, x_te, y_te, threshold=0.0, skip=("fraction_r2",))
