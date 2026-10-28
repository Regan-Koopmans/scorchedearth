"""Band metadata. Every task describes its input bands; adapters map them to each model.

Values handed to models are surface or top-of-atmosphere reflectance in [0, 1].
Tasks whose source data is on another scale convert it in their loader.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Band:
    name: str
    nm: float  # centre wavelength
    width: float  # bandwidth, nm


# Canonical bands, keyed by the name tasks and adapters use.
BLUE = Band("blue", 490, 65)
GREEN = Band("green", 560, 35)
RED = Band("red", 665, 30)
RE1 = Band("re1", 705, 15)
RE2 = Band("re2", 740, 15)
RE3 = Band("re3", 783, 20)
NIR = Band("nir", 842, 115)
NIR_NARROW = Band("nir_narrow", 865, 20)
COASTAL = Band("coastal", 443, 20)
WATER_VAPOUR = Band("water_vapour", 945, 20)
CIRRUS = Band("cirrus", 1375, 30)
SWIR1 = Band("swir1", 1610, 90)
SWIR2 = Band("swir2", 2190, 180)

# Sentinel-2 L1C, canonical B1..B12 order including B8A and B10.
S2_13 = [COASTAL, BLUE, GREEN, RED, RE1, RE2, RE3, NIR, NIR_NARROW, WATER_VAPOUR, CIRRUS, SWIR1, SWIR2]
S2_BY_NAME = dict(zip(["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12"], S2_13))

# HLS: the six bands Landsat and Sentinel-2 share.
HLS = [BLUE, GREEN, RED, NIR_NARROW, SWIR1, SWIR2]


@dataclass(frozen=True)
class Bands:
    """What a task's input tensor holds: one Band per channel, plus dataset stats."""

    bands: tuple[Band, ...]
    mean: tuple[float, ...]  # per-band reflectance mean over the task's data
    std: tuple[float, ...]

    @property
    def nm(self) -> list[float]:
        return [b.nm for b in self.bands]

    @property
    def widths(self) -> list[float]:
        return [b.width for b in self.bands]

    def nearest(self, targets: list[Band]) -> list[int]:
        """Index of the closest available band, by wavelength, for each target."""
        have = np.array(self.nm)
        return [int(np.abs(have - t.nm).argmin()) for t in targets]

    def index(self, band: Band) -> int | None:
        """Exact position of a band, or None if the task lacks it."""
        return self.bands.index(band) if band in self.bands else None
