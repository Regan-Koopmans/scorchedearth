# ScorchedEarth

A benchmark for geospatial embeddings across remote sensing tasks. Each task cuts imagery
into 64x64 patches, every model embeds each patch, and the frozen embeddings are scored by
simple probes. The same models run on every task, so results compare across tasks.

The leaderboard is `index.html`: an Overall tab plus one tab per task.

## Tasks

| Task | Dataset | Input | Label | Patches (train / test) |
|---|---|---|---|---|
| `burn_scars` | [HLS Burn Scars](https://huggingface.co/datasets/ibm-nasa-geospatial/hls_burn_scars) | HLS, 6 bands, 30 m | patch more than half burned | 12,000 / 6,000 sampled |
| `floods` | [Sen1Floods11](https://github.com/cloudtostreet/Sen1Floods11) | Sentinel-2 L1C, 13 bands, 10 m | patch more than half water | 12,000 / 4,641 |
| `landslides` | [Landslide4Sense](https://github.com/iarai/Landslide4Sense-2022) | Sentinel-2, 12 bands, 10 m | patch contains any landslide | 12,000 / 980 |
| `crop_type` | [Multi-temporal crop](https://huggingface.co/datasets/ibm-nasa-geospatial/multi-temporal-crop-classification) | HLS, 6 bands, middle of 3 dates | majority of 13 land-cover classes | 12,000 / 5,058 |

Training patches are a fixed random sample (seed 0), identical for every model. Each task
lives in `scorchedearth/datasets/` and returns patches, a `Bands` description (name,
wavelength and dataset stats per channel) and labels. Adapters read the `Bands` to feed each
model what it expects, so a task can bring any band set.

Task-specific choices:

- **Floods** uses the dataset's own train / test split; Bolivia, held out by the authors, is not used.
- **Landslides** are small, so a patch counts if it contains any landslide pixel. Fraction R²
  is not reported. The release divides each band by an unpublished constant; values are
  rescaled to reflectance with estimated factors, and bands B9 and B10 remain unreliable.
- **Crop type** keeps a patch when its majority class covers at least 30% of it, since fields
  are small at 30 m. Only the middle date is used, so every model sees one image.

Radar and elevation bands are left out so every model sees the same input.

Not yet included: building damage (xView2 is gated on Hugging Face) and deforestation
(ForestNet needs `tacoreader`).

## Metrics

Binary tasks (burn scars, floods, landslides):

| Metric | Question |
|---|---|
| `linear_f1`, `linear_auc` | Logistic probe: is the patch positive? |
| `knn_f1` | Do a patch's 20 nearest neighbours share its label? No training. |
| `fraction_r2` | Ridge probe: what fraction of the patch is positive? |
| `retrieval_p@10` | Search with a positive patch: are the 10 nearest also positive? |

Crop type reports macro `linear_f1`, macro `knn_f1` and `linear_acc`.

Every run also records `patches_per_sec` and `bytes_per_embedding`. The leaderboard shows
these once, on Overall: speed averaged over the tasks a model has run, size (the same on
every task) and a plot of embedding size against mean linear F1. Baselines are left out of
the speed and size views.

Caveats: one split, one seed, and probe regularisation is fixed rather than tuned per model.
A bootstrap check on floods gives 95% intervals of about ±0.012 linear F1, and the top three
flood models overlap, so gaps under about 0.02 may be noise. Error bars are not yet on the
page. Speeds are from one Apple M4 Pro (MPS), batch 64, after a warmup batch.

RGB-only models (DINOv2, DINOv3, Scale-MAE) score low because they never see near- or
shortwave-infrared, where burn scars and water stand out. Trying other RGB preparations on
DINOv3 (satellite) moved burn scar linear F1 by at most 0.06, and none reached the random
projection baseline.

## Usage

```sh
uv run main.py download --tasks floods,landslides       # into data/
uv run main.py run --tasks floods --limit 20            # quick test
uv run main.py run --n-train 12000 --n-test 6000        # all tasks, all models
uv run main.py run --tasks burn_scars --models prithvi_eo2_300m,dofa_base
uv run main.py precompute --tasks crop_type,floods      # look up AlphaEarth embeddings
uv run main.py report                                   # results -> index.html
```

Embeddings are cached in `cache/<task>/<model>/`, so re-scoring costs seconds, and an
interrupted run resumes from the last finished model. Results go to
`results/<task>/<model>.json`. `index.html#floods` links straight to a task.

## Models

| Family | Models | Band handling |
|---|---|---|
| Baselines | band stats, NDVI + NBR, random projection | any band set; no learned weights |
| HLS-native | Prithvi-EO 2.0 tiny, 100M, 300M, 600M | the six HLS bands, nearest available |
| Any-sensor | DOFA-B/L, Panopticon, Copernicus-FM, OlmoEarth v1 Base/Large, v1.2 Base | the task's true wavelengths |
| Sentinel-2 | SSL4EO MoCo R50, DINO ViT-S, MAE ViT-B/L, DeCUR R50, SoftCon, CROMA, Satlas | each expected band, or nearest wavelength if missing |
| RGB vision | DINOv2 ViT-B, DINOv3 ViT-B (web), DINOv3 ViT-L (satellite), Scale-MAE | red, green, blue |

Deep models get patches resized to their pretraining size (224 px; CROMA 120, Satlas 256).
OlmoEarth is resolution-aware and takes 64 px at 30 m. Adapters live in
`scorchedearth/foundation.py`; baselines in `scorchedearth/baselines.py`.

Not included: TerraMind (its terratorch release conflicts with the torchgeo used here),
Clay (no packaged loader), CROMA-L (torchgeo's weight checksum fails).

## Precomputed embeddings

Some embeddings are published ready-made, one vector per place, rather than as a model to
run. `scorchedearth/precomputed.py` looks them up for each patch's location and averages the
stored pixels under it. Nothing is run, so speed does not apply, and a precomputed source
may see more than our single image. It is reported separately from the live models.

| Source | Coverage | Access | Status |
|---|---|---|---|
| AlphaEarth Foundations (Google / Google DeepMind) | global, annual 2017–2025, 64 dims per 10 m pixel | anonymous, via the [TGE Labs copy](https://source.coop/tge-labs/aef) on Source Cooperative (unofficial), CC-BY 4.0 | crop type done (macro linear F1 0.759); floods in progress |
| OlmoEarth v1.3 embeddings (Ai2) | annual, 128 dims per 10 m pixel; global 2021–2025, partial before | anonymous, [Source Cooperative](https://source.coop/olmoearth/olmoearth-v1-3-embeddings), OlmoEarth Artifact License | access verified; reader not yet written |

How each source builds its embedding (from the papers): AlphaEarth reads up to a year of
Sentinel-2, Sentinel-1 and Landsat, computes each pixel from a 1.28 km tile around it, and
outputs unit-length vectors. OlmoEarth v1.3 summarises a calendar year of Sentinel-1,
Sentinel-2 and Landsat 8/9. Either way the embedding covers a year, so a fire or flood on a
known date is diluted into the yearly summary, and crop type, a season-long pattern,
favours them over single-image live models.

AlphaEarth patches are averaged from its 80 m overview, which matches full resolution to
cosine similarity ≥ 0.9997 on spot checks and lets one read serve a whole file. Rescaling
averaged vectors to unit length changes crop type scores by at most 0.005, so they are left
as averaged. Floods use each event's year; the 2016 India event uses AlphaEarth's first
layer, 2017, since none exists for 2016.

Only tasks whose imagery carries coordinates can use precomputed embeddings: burn scars,
floods and crop type. Landslides has none.

Not usable yet: [Tessera](https://github.com/ucam-eo/tessera). Its `geotessera` library
reaches only 3 of 40 sampled US crop patches; its Zarr store on Source Cooperative is
untested. Major TOM embedding sets on Hugging Face are open but hold one image per location
at dates not chosen by us. Clay and Earth Index cover mostly 2024 onward.

## Adding a model

Subclass `Adapter` in `scorchedearth/foundation.py`, implement `load()` and `forward()`, and
decorate it with `@register`. `forward(x, bands)` receives a device tensor `(N, C, H, W)`,
already resized to `size`, and the task's `Bands`. It returns `(N, D)`. Values are
reflectance in [0, 1].

## Adding a task

Add a module to `scorchedearth/datasets/` with `NAME`, `download()` and
`load(limit) -> TaskData`, then list it in `datasets/__init__.py`. Use `tasks.patchify` for
binary masks or `tasks.patchify_majority` for class maps. To support precomputed
embeddings, also provide `YEAR` and `patch_footprints(split)`, which returns footprints and
labels in load order, plus optionally a per-patch year.

## Licences and credits

HLS Burn Scars and Multi-temporal crop: IBM / NASA, CC-BY 4.0. Sen1Floods11: Cloud to
Street. Landslide4Sense: IARAI. AlphaEarth Foundations Satellite Embedding dataset: produced
by Google and Google DeepMind, CC-BY 4.0. OlmoEarth v1.3 embeddings: Ai2, OlmoEarth Artifact
License. `scorchedearth/vendor/prithvi_mae.py` is IBM's
Apache-2.0 code with its licence header intact.
