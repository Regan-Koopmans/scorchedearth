# ScorchedEarth

A benchmark for geospatial embeddings, built on the
[HLS Burn Scars](https://huggingface.co/datasets/ibm-nasa-geospatial/hls_burn_scars) dataset:
804 Harmonized Landsat/Sentinel-2 scenes (512x512, 6 bands) of US wildfires, 2018–2021,
with pixel-level burn masks.

## How it works

Each scene is cut into 64x64 patches. Patches with more than 10% missing data are dropped.
A model embeds each patch into a vector. The embeddings are frozen and scored on:

| Metric | Task |
|---|---|
| `linear_f1`, `linear_auc` | Logistic probe: is the patch more than 50% burned? |
| `knn_f1` | kNN (k=20) on the same label |
| `fraction_r2` | Ridge probe: burned fraction of the patch |
| `retrieval_p@10` | Burned patches retrieving burned patches, cosine similarity |
| `patches_per_sec`, `ms_per_patch` | Embedding throughput, batch 64, after one warmup batch |
| `dim`, `bytes_per_embedding` | Embedding size at float32 |

Train on the `training` split, score on `validation`.

## Usage

```sh
uv run main.py download              # ~2.6 GB, into data/
uv run main.py run --limit 50        # quick test
uv run main.py run --n-train 12000 --n-test 6000   # fixed-seed subsample, all models
uv run main.py run --models prithvi_eo2_300m,dofa_base
uv run main.py report                # results/*.json -> index.html, one bar chart per metric
```

## Models

| Family | Models | Band handling |
|---|---|---|
| HLS-native | Prithvi-EO 2.0 tiny, 100M, 300M, 600M | same six bands |
| Any-sensor | DOFA-B/L, Panopticon, Copernicus-FM, OlmoEarth v1 Base/Large, v1.2 Base | true HLS wavelengths |
| Sentinel-2 | SSL4EO MoCo R50, DINO ViT-S, MAE ViT-B/L, DeCUR R50, SoftCon, CROMA, Satlas | missing S2 bands copied from nearest wavelength |
| RGB vision | DINOv2 ViT-B, DINOv3 ViT-B (web), DINOv3 ViT-L (satellite), Scale-MAE | red, green, blue |

Deep models get patches resized to their pretraining size (224 px, CROMA 120, Satlas 256).
OlmoEarth is resolution-aware and takes 64 px at 30 m. Adapters live in `scorchedearth/foundation.py`.

Not included: TerraMind (its terratorch release conflicts with the torchgeo used here), Clay (no packaged loader),
CROMA-L (torchgeo's weight checksum fails).

## Adding a model

Subclass `Adapter` in `scorchedearth/foundation.py`, implement `load()` and `forward()`, decorate with `@register`.
`forward` receives a device tensor `(N, 6, H, W)` already resized to `size`, and returns `(N, D)`.

Bands, in order: blue, green, red, NIR, SWIR1, SWIR2. Values are reflectance.
