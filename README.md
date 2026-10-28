# ScorchedEarth

A benchmark for geospatial embeddings across remote sensing tasks. Each task cuts imagery
into 64x64 patches, every model embeds each patch, and frozen embeddings are scored by
simple probes. The same models run on every task.

## Tasks

| Task | Dataset | Input | Label |
|---|---|---|---|
| `burn_scars` | [HLS Burn Scars](https://huggingface.co/datasets/ibm-nasa-geospatial/hls_burn_scars) | HLS, 6 bands, 30 m | patch more than half burned |
| `floods` | [Sen1Floods11](https://github.com/cloudtostreet/Sen1Floods11) | Sentinel-2 L1C, 13 bands, 10 m | patch more than half water |
| `landslides` | [Landslide4Sense](https://github.com/iarai/Landslide4Sense-2022) | Sentinel-2, 12 bands, 10 m | patch contains any landslide |
| `crop_type` | [Multi-temporal crop](https://huggingface.co/datasets/ibm-nasa-geospatial/multi-temporal-crop-classification) | HLS, 6 bands, middle of 3 dates | majority land-cover class (13) |

Each task lives in `scorchedearth/datasets/` and returns patches, a `Bands` description
(name, wavelength and dataset stats per channel) and labels. Adapters read the `Bands` to
feed each model what it expects, so a task brings any band set it likes.

Not yet included: building damage (xView2 is gated on Hugging Face), deforestation (ForestNet,
needs `tacoreader`). Radar and elevation bands are left out so every model sees the same input.

## Metrics

Binary tasks:

| Metric | Question |
|---|---|
| `linear_f1`, `linear_auc` | Logistic probe: is the patch positive? |
| `knn_f1` | Do a patch's 20 nearest neighbours share its label? |
| `fraction_r2` | Ridge probe: what fraction of the patch is positive? |
| `retrieval_p@10` | Search with a positive patch: are the top 10 positive? |

Multiclass tasks report macro `linear_f1`, macro `knn_f1` and `linear_acc`.
All tasks report `patches_per_sec`, `ms_per_patch`, `dim` and `bytes_per_embedding`.

## Usage

```sh
uv run main.py download --tasks floods,landslides       # into data/
uv run main.py run --tasks floods --limit 20            # quick test
uv run main.py run --n-train 12000 --n-test 6000        # all tasks, all models, fixed-seed sample
uv run main.py run --tasks burn_scars --models prithvi_eo2_300m,dofa_base
uv run main.py report                                   # results -> index.html
```

Embeddings are cached in `cache/<task>/<model>/`, so re-scoring costs seconds.
Results go to `results/<task>/<model>.json`. The leaderboard has a tab per task plus an
Overall tab ranking models by mean rank across finished tasks; `index.html#floods` links to one.

## Models

| Family | Models | Band handling |
|---|---|---|
| Baselines | band stats, NDVI + NBR, random projection | any band set; no learned weights |
| HLS-native | Prithvi-EO 2.0 tiny, 100M, 300M, 600M | the six HLS bands, nearest available |
| Any-sensor | DOFA-B/L, Panopticon, Copernicus-FM, OlmoEarth v1 Base/Large, v1.2 Base | the task's true wavelengths |
| Sentinel-2 | SSL4EO MoCo R50, DINO ViT-S, MAE ViT-B/L, DeCUR R50, SoftCon, CROMA, Satlas | each expected band, or nearest wavelength if the task lacks it |
| RGB vision | DINOv2 ViT-B, DINOv3 ViT-B (web), DINOv3 ViT-L (satellite), Scale-MAE | red, green, blue |

Deep models get patches resized to their pretraining size (224 px, CROMA 120, Satlas 256).
OlmoEarth is resolution-aware and takes 64 px at 30 m. Adapters live in `scorchedearth/foundation.py`.

Not included: TerraMind (its terratorch release conflicts with the torchgeo used here), Clay (no packaged loader),
CROMA-L (torchgeo's weight checksum fails).

## Adding a model

Subclass `Adapter` in `scorchedearth/foundation.py`, implement `load()` and `forward()`, decorate with `@register`.
`forward(x, bands)` receives a device tensor `(N, C, H, W)` already resized to `size`, and the
task's `Bands`. It returns `(N, D)`. Values are reflectance in [0, 1].

## Adding a task

Add a module to `scorchedearth/datasets/` with `NAME`, `download()` and `load(limit) -> TaskData`,
then list it in `datasets/__init__.py`. Use `tasks.patchify` for binary masks or
`tasks.patchify_majority` for class maps.
