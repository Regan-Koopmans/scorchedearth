"""Pretrained foundation models.

Input to every embedder: HLS patches (N, 6, H, W), reflectance in [0, 1],
bands blue, green, red, nir (B8A), swir1 (B11), swir2 (B12).

Each adapter maps those bands to what the model saw in pretraining:
- Spectrally-aware models get true HLS wavelengths.
- Fixed 13-band Sentinel-2 models get missing bands copied from the nearest
  available wavelength. Same rule as OlmoEarth's v1 normaliser.
- RGB models get red, green, blue.
Patches are resized to each model's pretraining size. Pooled output is the embedding.
"""

import json

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

from .models import MODELS

DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

# HLS S30 band centres and widths, nm.
HLS_WAVELENGTHS = [490, 560, 665, 865, 1610, 2190]
HLS_BANDWIDTHS = [65, 35, 30, 20, 90, 180]

# Sentinel-2 L1C band centres, nm, in canonical B1..B12 order (13 bands incl. B8A, B10).
S2_13 = {"B1": 443, "B2": 490, "B3": 560, "B4": 665, "B5": 705, "B6": 740, "B7": 783,
         "B8": 842, "B8A": 865, "B9": 945, "B10": 1375, "B11": 1610, "B12": 2190}


def nearest_band_map(target_nm: list[float]) -> list[int]:
    """For each target wavelength, index of the nearest HLS band."""
    src = np.array(HLS_WAVELENGTHS)
    return [int(np.abs(src - t).argmin()) for t in target_nm]


class Adapter:
    """Lazy-loading wrapper. Subclasses implement load() and forward(t) on a device tensor."""

    name = ""
    size = 224

    def __init__(self):
        self.model = None

    def load(self):
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if self.model is None:
            self.load()
            self.model = self.model.eval().to(DEVICE)
        with torch.inference_mode():
            t = torch.from_numpy(x).to(DEVICE)
            if self.size and t.shape[-1] != self.size:
                t = F.interpolate(t, size=self.size, mode="bilinear", align_corners=False)
            return self.forward(t).float().cpu().numpy()


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

    def forward(self, x):
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


# --- Spectrally-aware: take arbitrary bands with wavelengths ------------------------


@register
class DOFA(Adapter):
    name = "dofa_base"

    def load(self):
        import torchgeo.models as tg

        self.model = tg.dofa_base_patch16_224(tg.DOFABase16_Weights.DOFA_MAE)
        self.model.head = torch.nn.Identity()

    def forward(self, x):
        # DOFA pretraining fed reflectance on the 0-10000 scale, then standardised per band.
        # Without per-sensor stats, standardise each band with dataset-level HLS stats.
        x = (x - HLS_MEAN) / HLS_STD
        return self.model.forward_features(x, [w / 1000 for w in HLS_WAVELENGTHS])


@register
class Panopticon(Adapter):
    name = "panopticon"

    def load(self):
        import torchgeo.models as tg

        self.model = tg.panopticon_vitb14(tg.Panopticon_Weights.VIT_BASE14)

    def forward(self, x):
        x = (x - HLS_MEAN) / HLS_STD
        ids = torch.tensor(HLS_WAVELENGTHS, device=x.device).expand(x.shape[0], -1)
        return self.model({"imgs": x, "chn_ids": ids})


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

    def forward(self, x):
        x = (x - HLS_MEAN) / HLS_STD
        meta = torch.full((x.shape[0], 4), float("nan"), device=x.device)
        return self.model.forward_features(
            x, meta, wavelengths=HLS_WAVELENGTHS, bandwidths=HLS_BANDWIDTHS,
            input_mode="spectral", kernel_size=16,
        )


@register
class OlmoEarth(Adapter):
    name = "olmoearth_base"
    size = None  # resolution-aware; native 30 m input

    def load(self):
        from olmoearth_pretrain_minimal import ModelID, NormalizerV1, load_model_from_id
        from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.utils.constants import Modality

        self.model = load_model_from_id(ModelID.OLMOEARTH_V1_BASE, load_weights=True)
        self.normalizer = NormalizerV1()
        self.modality = Modality.SENTINEL2_L2A
        order = self.modality.band_order  # B02 B03 B04 B08 B05 B06 B07 B8A B11 B12 B01 B09
        hls = {"B02": 0, "B03": 1, "B04": 2, "B8A": 3, "B11": 4, "B12": 5}
        self.slots = [(order.index(b), i) for b, i in hls.items()]
        self.n_bands = len(order)

    def forward(self, x):
        from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.utils.datatypes import MaskedOlmoEarthSample

        b, _, h, w = x.shape
        arr = np.full((b, h, w, 1, self.n_bands), np.nan, np.float32)
        hwc = (x.permute(0, 2, 3, 1) * 10000).cpu().numpy()
        for dst, src in self.slots:
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

    def forward(self, x):
        x = x[:, nearest_band_map(list(S2_13.values()))] * 10000
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

    def forward(self, x):
        idx = nearest_band_map([S2_13[b] for b in S2_13 if b != "B10"])
        # CROMA scales each band from mean - 2 std .. mean + 2 std to [0, 1], using dataset stats.
        lo, hi = (HLS_MEAN - 2 * HLS_STD)[:, idx], (HLS_MEAN + 2 * HLS_STD)[:, idx]
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
        self.order = [S2_13[b] for b in ("B4", "B3", "B2", "B5", "B6", "B7", "B8", "B11", "B12")]
        self.norm = w.transforms[-1]

    def forward(self, x):
        x = x[:, nearest_band_map(self.order)] * 10000
        # Satlas: RGB were 8-bit (reflectance / 10000 * 255 * ~3.2 brightening); others / 8160.
        x[:, :3] = (x[:, :3] / 10000 * 255 * 3.2).clamp(0, 255)
        return self.model(self.norm(x).clamp(0, 1))


# --- RGB vision models: DINOv3 natural images vs satellite ---------------------------


class _DINOv3(Adapter):
    timm_name = ""

    def load(self):
        import timm

        self.model = timm.create_model(self.timm_name, pretrained=True, num_classes=0)
        cfg = self.model.pretrained_cfg
        self.mean = torch.tensor(cfg["mean"], device=DEVICE).view(1, 3, 1, 1)
        self.std = torch.tensor(cfg["std"], device=DEVICE).view(1, 3, 1, 1)

    def forward(self, x):
        rgb = (x[:, [2, 1, 0]] * 3.0).clamp(0, 1)  # brighten reflectance to display range
        return self.model((rgb - self.mean) / self.std)


@register
class DINOv3_Web(_DINOv3):
    name = "dinov3_vitb16_web"
    timm_name = "vit_base_patch16_dinov3.lvd1689m"


@register
class DINOv3_Sat(_DINOv3):
    name = "dinov3_vitl16_sat"
    timm_name = "vit_large_patch16_dinov3.sat493m"


# Per-band stats of this dataset (reflectance), for models without sensor-specific stats.
HLS_MEAN = torch.tensor([0.0566, 0.0818, 0.1031, 0.1711, 0.2284, 0.1746], device=DEVICE).view(1, 6, 1, 1)
HLS_STD = torch.tensor([0.0312, 0.0389, 0.0533, 0.0713, 0.0835, 0.0729], device=DEVICE).view(1, 6, 1, 1)
