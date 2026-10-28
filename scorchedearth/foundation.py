"""Pretrained foundation models.

Input to every embedder: patches (N, C, H, W), reflectance in [0, 1], plus the task's
`Bands`, which names each channel and gives its wavelength and dataset stats.

Each adapter maps the task's bands to what the model saw in pretraining:
- Spectrally-aware models get the true wavelengths of every task band.
- Fixed-band models (Sentinel-2, HLS) get each expected band from the task, or the
  nearest available wavelength when the task lacks it. Same rule as OlmoEarth's v1
  normaliser.
- RGB models get red, green, blue.
Patches are resized to each model's pretraining size. Pooled output is the embedding.
"""

import json

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

from .bands import BLUE, GREEN, HLS, RED, S2_13, S2_BY_NAME, Bands
from .models import MODELS

DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"


class Adapter:
    """Lazy-loading wrapper. Subclasses implement load() and forward(x, bands) on a device tensor."""

    name = ""
    size = 224

    def __init__(self):
        self.model = None

    def load(self):
        raise NotImplementedError

    def forward(self, x: torch.Tensor, bands: Bands) -> torch.Tensor:
        raise NotImplementedError

    def __call__(self, x: np.ndarray, bands: Bands) -> np.ndarray:
        if self.model is None:
            self.load()
            self.model = self.model.eval().to(DEVICE)
        with torch.inference_mode():
            t = torch.from_numpy(x).to(DEVICE)
            if self.size and t.shape[-1] != self.size:
                t = F.interpolate(t, size=self.size, mode="bilinear", align_corners=False)
            return self.forward(t, bands).float().cpu().numpy()


def stats(bands: Bands, idx: list[int] | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Task's per-band mean and std as (1, C, 1, 1) device tensors, optionally reindexed."""
    mean, std = np.array(bands.mean), np.array(bands.std)
    if idx is not None:
        mean, std = mean[idx], std[idx]
    to = lambda a: torch.tensor(a, dtype=torch.float32, device=DEVICE).view(1, -1, 1, 1)
    return to(mean), to(std)


def standardise(x: torch.Tensor, bands: Bands) -> torch.Tensor:
    mean, std = stats(bands)
    return (x - mean) / std


def register(cls):
    MODELS[cls.name] = cls()
    return cls


# --- Prithvi: pretrained on HLS, same 6 bands ---------------------------------------


class _Prithvi(Adapter):
    repo = weights = ""

    def load(self):
        from .vendor.prithvi_mae import PrithviViT

        cfg = json.loads(open(hf_hub_download(self.repo, "config.json")).read())["pretrained_cfg"]
        self.size = cfg["img_size"]
        model = PrithviViT(
            img_size=cfg["img_size"], patch_size=cfg["patch_size"], num_frames=1,
            in_chans=cfg["in_chans"], embed_dim=cfg["embed_dim"], depth=cfg["depth"],
            num_heads=cfg["num_heads"], mlp_ratio=cfg["mlp_ratio"],
            coords_encoding=cfg.get("coords_encoding", []),
        )
        state = torch.load(hf_hub_download(self.repo, self.weights), map_location="cpu", weights_only=True)
        enc = {k.removeprefix("encoder."): v for k, v in state.items()
               if k.startswith("encoder.") and "pos_embed" not in k}
        missing, unexpected = model.load_state_dict(enc, strict=False)
        assert not unexpected and all("pos_embed" in k for k in missing), (missing, unexpected)
        self.model = model
        # Stats are on the HLS 0-10000 scale.
        self.mean = torch.tensor(cfg["mean"], device=DEVICE).view(1, -1, 1, 1) / 10000
        self.std = torch.tensor(cfg["std"], device=DEVICE).view(1, -1, 1, 1) / 10000

    def forward(self, x, bands):
        x = x[:, bands.nearest(HLS)]
        tokens = self.model.forward_features((x - self.mean) / self.std)[-1]
        return tokens[:, 1:].mean(dim=1)


@register
class Prithvi300M(_Prithvi):
    name = "prithvi_eo2_300m"
    repo = "ibm-nasa-geospatial/Prithvi-EO-2.0-300M"
    weights = "Prithvi_EO_V2_300M.pt"


@register
class Prithvi600M(_Prithvi):
    name = "prithvi_eo2_600m"
    repo = "ibm-nasa-geospatial/Prithvi-EO-2.0-600M"
    weights = "Prithvi_EO_V2_600M.pt"


@register
class Prithvi100M(_Prithvi):
    name = "prithvi_eo2_100m"
    repo = "ibm-nasa-geospatial/Prithvi-EO-2.0-100M-TL"
    weights = "Prithvi_EO_V2_100M_TL.pt"


@register
class PrithviTiny(_Prithvi):
    name = "prithvi_eo2_tiny"
    repo = "ibm-nasa-geospatial/Prithvi-EO-2.0-tiny-TL"
    weights = "Prithvi_EO_V2_tiny_TL.pt"


# --- Spectrally-aware: take arbitrary bands with wavelengths ------------------------


@register
class DOFA(Adapter):
    name = "dofa_base"
    build, weights = "dofa_base_patch16_224", "DOFABase16_Weights"

    def load(self):
        import torchgeo.models as tg

        self.model = getattr(tg, self.build)(getattr(tg, self.weights).DOFA_MAE)
        self.model.head = torch.nn.Identity()

    def forward(self, x, bands):
        # DOFA pretraining standardised each band per sensor. Without its stats for every
        # sensor, standardise each band with the task's own dataset stats.
        return self.model.forward_features(standardise(x, bands), [w / 1000 for w in bands.nm])


@register
class DOFALarge(DOFA):
    name = "dofa_large"
    build, weights = "dofa_large_patch16_224", "DOFALarge16_Weights"


@register
class Panopticon(Adapter):
    name = "panopticon"

    def load(self):
        import torchgeo.models as tg

        self.model = tg.panopticon_vitb14(tg.Panopticon_Weights.VIT_BASE14)

    def forward(self, x, bands):
        ids = torch.tensor(bands.nm, device=x.device).expand(x.shape[0], -1)
        return self.model({"imgs": standardise(x, bands), "chn_ids": ids})


@register
class CopernicusFM(Adapter):
    name = "copernicus_fm"

    def load(self):
        import torchgeo.models as tg

        self.model = tg.copernicusfm_base(tg.CopernicusFM_Base_Weights.CopernicusFM_ViT)
        self.model.head = torch.nn.Identity()
        # The spectral expansion runs in float64, which MPS lacks. Compute it on CPU.
        pe = self.model.patch_embed_spectral
        for name in ("spectrum_central_expansion", "spectrum_bandwidth_expansion"):
            mod = getattr(pe, name)
            mod.forward = lambda x, d, f=mod.forward: f(x.cpu(), d).to(x.device)

    def forward(self, x, bands):
        meta = torch.full((x.shape[0], 4), float("nan"), device=x.device)
        return self.model.forward_features(
            standardise(x, bands), meta, wavelengths=bands.nm, bandwidths=bands.widths,
            input_mode="spectral", kernel_size=16,
        )


@register
class OlmoEarth(Adapter):
    name = "olmoearth_base"
    size = None  # resolution-aware; native 30 m input
    model_id, version = "OLMOEARTH_V1_BASE", "v1"

    def load(self):
        from olmoearth_pretrain_minimal import ModelID, Normalizer, load_model_from_id
        from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.utils.constants import Modality

        self.model = load_model_from_id(getattr(ModelID, self.model_id), load_weights=True)
        self.normalizer = Normalizer(model_type=self.version)
        self.modality = Modality.SENTINEL2_L2A
        # B02 B03 B04 B08 B05 B06 B07 B8A B11 B12 B01 B09, as canonical bands.
        self.order = [S2_BY_NAME[b.replace("B0", "B")] for b in self.modality.band_order]

    def forward(self, x, bands):
        from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.utils.datatypes import MaskedOlmoEarthSample

        b, _, h, w = x.shape
        arr = np.full((b, h, w, 1, len(self.order)), np.nan, np.float32)
        hwc = (x.permute(0, 2, 3, 1) * 10000).cpu().numpy()
        for dst, band in enumerate(self.order):
            src = bands.index(band)
            if src is not None:
                arr[..., 0, dst] = hwc[..., src]
        norm = torch.from_numpy(self.normalizer.normalize(self.modality, arr)).float().to(DEVICE)
        ts = torch.zeros(b, 1, 3, dtype=torch.long, device=DEVICE)
        ts[..., 0], ts[..., 1], ts[..., 2] = 15, 7, 2020  # mid-summer; dates unknown per patch
        sample = MaskedOlmoEarthSample(
            timestamps=ts, sentinel2_l2a=norm,
            sentinel2_l2a_mask=torch.zeros(b, h, w, 1, 3, dtype=torch.long, device=DEVICE),
        )
        tm = self.model.encoder(sample, patch_size=8, input_res=30, fast_pass=True)["tokens_and_masks"]
        return tm.sentinel2_l2a.mean(dim=(1, 2, 3, 4))


@register
class OlmoEarthLarge(OlmoEarth):
    name = "olmoearth_large"
    model_id = "OLMOEARTH_V1_LARGE"


@register
class OlmoEarth12(OlmoEarth):
    name = "olmoearth_v1_2_base"
    model_id, version = "OLMOEARTH_V1_2_BASE", "v1.2"


# --- Fixed 13-band Sentinel-2 models (SSL4EO-S12, CROMA, Satlas) ---------------------


def _timm_pool(model, x):
    return model.forward_head(model.forward_features(x), pre_logits=True)


class _SSL4EO(Adapter):
    weights = None
    build = None

    def load(self):
        import torchgeo.models as tg

        w = eval(f"tg.{self.weights}")
        self.model = getattr(tg, self.build)(w)
        self.norm = w.transforms[-1]
        self.model.reset_classifier(0) if hasattr(self.model, "reset_classifier") else None

    def forward(self, x, bands):
        x = x[:, bands.nearest(S2_13)] * 10000
        return _timm_pool(self.model, self.norm(x))


@register
class SSL4EO_MoCo_R50(_SSL4EO):
    name = "ssl4eo_moco_resnet50"
    weights, build = "ResNet50_Weights.SENTINEL2_ALL_MOCO", "resnet50"


@register
class SSL4EO_DINO_ViTS(_SSL4EO):
    name = "ssl4eo_dino_vits16"
    weights, build = "ViTSmall16_Weights.SENTINEL2_ALL_DINO", "vit_small_patch16_224"


@register
class SSL4EO_MAE_ViTB(_SSL4EO):
    name = "ssl4eo_mae_vitb16"
    weights, build = "ViTBase16_Weights.SENTINEL2_ALL_MAE", "vit_base_patch16_224"


@register
class SSL4EO_MAE_ViTL(_SSL4EO):
    name = "ssl4eo_mae_vitl16"
    weights, build = "ViTLarge16_Weights.SENTINEL2_ALL_MAE", "vit_large_patch16_224"


@register
class DeCUR_R50(_SSL4EO):
    name = "decur_resnet50"
    weights, build = "ResNet50_Weights.SENTINEL2_ALL_DECUR", "resnet50"


@register
class SoftCon_ViTB(_SSL4EO):
    name = "softcon_vitb14"
    weights, build = "ViTBase14_DINOv2_Weights.SENTINEL2_ALL_SOFTCON", "vit_base_patch14_dinov2"


@register
class CROMA(Adapter):
    name = "croma_base"
    size = 120  # optical encoder takes 12 S2 bands (no B10)

    def load(self):
        import torchgeo.models as tg

        self.model = tg.croma_base(tg.CROMABase_Weights.CROMA_VIT, modalities=["optical"])

    def forward(self, x, bands):
        idx = bands.nearest([b for name, b in S2_BY_NAME.items() if name != "B10"])
        # CROMA scales each band from mean - 2 std .. mean + 2 std to [0, 1], using dataset stats.
        mean, std = stats(bands, idx)
        lo, hi = mean - 2 * std, mean + 2 * std
        x = ((x[:, idx] - lo) / (hi - lo)).clamp(0, 1)
        return self.model(x_optical=x)["optical_GAP"]


@register
class Satlas(Adapter):
    name = "satlas_swinb_s2"
    size = 256

    def load(self):
        import torchgeo.models as tg

        w = tg.Swin_V2_B_Weights.SENTINEL2_SI_MS_SATLAS
        self.model = tg.swin_v2_b(w)
        self.model.head = torch.nn.Identity()
        self.order = [S2_BY_NAME[b] for b in ("B4", "B3", "B2", "B5", "B6", "B7", "B8", "B11", "B12")]
        self.norm = w.transforms[-1]

    def forward(self, x, bands):
        x = x[:, bands.nearest(self.order)] * 10000
        # Satlas: RGB were 8-bit (reflectance / 10000 * 255 * ~3.2 brightening); others / 8160.
        x[:, :3] = (x[:, :3] / 10000 * 255 * 3.2).clamp(0, 255)
        return self.model(self.norm(x).clamp(0, 1))


# --- RGB vision models: DINOv3 natural images vs satellite ---------------------------


def display_rgb(x, bands):
    """Red, green, blue reflectance brightened to display range [0, 1]."""
    return (x[:, bands.nearest([RED, GREEN, BLUE])] * 3.0).clamp(0, 1)


class _DINOv3(Adapter):
    timm_name = ""

    def load(self):
        import timm

        self.model = timm.create_model(self.timm_name, pretrained=True, num_classes=0, img_size=self.size)
        cfg = self.model.pretrained_cfg
        self.mean = torch.tensor(cfg["mean"], device=DEVICE).view(1, 3, 1, 1)
        self.std = torch.tensor(cfg["std"], device=DEVICE).view(1, 3, 1, 1)

    def forward(self, x, bands):
        return self.model((display_rgb(x, bands) - self.mean) / self.std)


@register
class DINOv3_Web(_DINOv3):
    name = "dinov3_vitb16_web"
    timm_name = "vit_base_patch16_dinov3.lvd1689m"


@register
class DINOv3_Sat(_DINOv3):
    name = "dinov3_vitl16_sat"
    timm_name = "vit_large_patch16_dinov3.sat493m"


@register
class DINOv2_Web(_DINOv3):
    name = "dinov2_vitb14_web"
    timm_name = "vit_base_patch14_dinov2.lvd142m"


@register
class ScaleMAE(Adapter):
    name = "scalemae_large"

    def load(self):
        import torchgeo.models as tg

        self.model = tg.scalemae_large_patch16(tg.ScaleMAELarge16_Weights.FMOW_RGB, res=30.0)
        self.model.reset_classifier(0)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)

    def forward(self, x, bands):
        return self.model((display_rgb(x, bands) - self.mean) / self.std)

