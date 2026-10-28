"""HLS Burn Scars (IBM / NASA). Wildfire scars, contiguous US, 2018-2021.

Scenes: 512x512, 6 bands (Blue, Green, Red, NIR, SWIR1, SWIR2), reflectance.
Masks: 1 = burned, 0 = not burned, -1 = no data.
Task: binary, patch is more than half burned.
"""

import tarfile
from pathlib import Path

import numpy as np
import tifffile
from huggingface_hub import hf_hub_download

from ..bands import HLS, Bands
from ..tasks import TaskData, patchify

NAME = "burn_scars"
REPO = "ibm-nasa-geospatial/hls_burn_scars"
DATA_DIR = Path("data")
# Per-band reflectance stats over the training split.
BANDS = Bands(
    tuple(HLS),
    mean=(0.0566, 0.0818, 0.1031, 0.1711, 0.2284, 0.1746),
    std=(0.0312, 0.0389, 0.0533, 0.0713, 0.0835, 0.0729),
)


def download(data_dir: Path = DATA_DIR) -> Path:
    """Fetch and extract the dataset (~2.6 GB). Skips if already present."""
    if (data_dir / "training").exists():
        return data_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    archive = hf_hub_download(REPO, "hls_burn_scars.tar.gz", repo_type="dataset")
    with tarfile.open(archive) as tar:
        tar.extractall(data_dir, filter="data")
    return data_dir


def _pairs(split_dir: Path) -> list[tuple[Path, Path]]:
    """Match each scene to its mask by shared filename prefix."""
    scenes = sorted(split_dir.glob("*_merged.tif"))
    pairs = []
    for scene in scenes:
        prefix = scene.name.removesuffix("_merged.tif")
        masks = [m for m in split_dir.glob(f"{prefix}*.tif") if m != scene]
        if masks:
            pairs.append((scene, masks[0]))
    return pairs


def _chw(a: np.ndarray) -> np.ndarray:
    """Return array as (C, H, W)."""
    if a.ndim == 2:
        return a[None]
    return a if a.shape[0] < a.shape[-1] else np.moveaxis(a, -1, 0)


def load_split(split: str, data_dir: Path = DATA_DIR, limit: int | None = None):
    """Load a split. Returns scenes (N, 6, 512, 512) float32 and masks (N, 512, 512) int8."""
    pairs = _pairs(data_dir / split)[:limit]
    if not pairs:
        raise FileNotFoundError(f"No scene/mask pairs in {data_dir / split}. Run `download` first.")
    scenes = np.stack([_chw(tifffile.imread(s)).astype(np.float32) for s, _ in pairs])
    masks = np.stack([_chw(tifffile.imread(m))[0].astype(np.int8) for _, m in pairs])
    return scenes, masks


def load(limit: int | None = None) -> TaskData:
    x_tr, y_tr = patchify(*load_split("training", limit=limit))
    x_te, y_te = patchify(*load_split("validation", limit=limit))
    return TaskData(NAME, BANDS, "binary", x_tr, y_tr, x_te, y_te)
