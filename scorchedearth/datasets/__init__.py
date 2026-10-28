"""Benchmark datasets. Each module exposes NAME, download() and load(limit) -> TaskData."""

from . import burn_scars, crop_type, floods, landslides

TASKS = {m.NAME: m for m in (burn_scars, floods, landslides, crop_type)}
