"""ScorchedEarth CLI.

    uv run main.py download [--tasks a,b]
    uv run main.py run [--tasks a,b] [--models a,b] [--limit N] [--n-train N] [--n-test N]
    uv run main.py report
"""

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

from scorchedearth import foundation  # noqa: F401  foundation registers models
from scorchedearth.datasets import TASKS
from scorchedearth.models import MODELS
from scorchedearth.run import RESULTS, run

# Display name, family (item quality), size, note. Family drives colour on the leaderboard.
MODEL_INFO = {
    "prithvi_eo2_300m": ("Prithvi-EO 2.0 300M", "unique", "300M", "Pretrained on HLS, same six bands."),
    "prithvi_eo2_600m": ("Prithvi-EO 2.0 600M", "unique", "600M", "Pretrained on HLS, same six bands."),
    "prithvi_eo2_100m": ("Prithvi-EO 2.0 100M", "unique", "100M", "Pretrained on HLS, same six bands."),
    "prithvi_eo2_tiny": ("Prithvi-EO 2.0 tiny", "unique", "5M", "Pretrained on HLS, same six bands."),
    "dofa_base": ("DOFA-B", "magic", "ViT-B", "Wavelength-conditioned patch embedding."),
    "dofa_large": ("DOFA-L", "magic", "ViT-L", "Wavelength-conditioned patch embedding."),
    "panopticon": ("Panopticon", "magic", "ViT-B", "Channel attention over any band set."),
    "copernicus_fm": ("Copernicus-FM", "magic", "ViT-B", "Wavelength and bandwidth hypernetwork."),
    "olmoearth_base": ("OlmoEarth v1 Base", "magic", "Base", "Missing S2 bands copied from nearest (v1 rule)."),
    "olmoearth_large": ("OlmoEarth v1 Large", "magic", "Large", "Missing S2 bands copied from nearest (v1 rule)."),
    "olmoearth_v1_2_base": ("OlmoEarth v1.2 Base", "magic", "Base", "Missing S2 bands zeroed (trained with band dropout)."),
    "ssl4eo_moco_resnet50": ("SSL4EO MoCo R50", "rare", "ResNet-50", "13-band S2. Missing bands copied from nearest."),
    "ssl4eo_dino_vits16": ("SSL4EO DINO ViT-S", "rare", "ViT-S", "13-band S2. Missing bands copied from nearest."),
    "ssl4eo_mae_vitb16": ("SSL4EO MAE ViT-B", "rare", "ViT-B", "13-band S2. Missing bands copied from nearest."),
    "ssl4eo_mae_vitl16": ("SSL4EO MAE ViT-L", "rare", "ViT-L", "13-band S2. Missing bands copied from nearest."),
    "decur_resnet50": ("DeCUR R50", "rare", "ResNet-50", "13-band S2. Missing bands copied from nearest."),
    "softcon_vitb14": ("SoftCon ViT-B", "rare", "ViT-B", "13-band S2. Missing bands copied from nearest."),
    "croma_base": ("CROMA", "rare", "ViT-B", "12-band S2 optical encoder."),
    "satlas_swinb_s2": ("Satlas Swin-B", "rare", "Swin-B", "9-band S2."),
    "dinov3_vitb16_web": ("DINOv3 ViT-B (web)", "set", "ViT-B", "RGB only. Natural images."),
    "dinov3_vitl16_sat": ("DINOv3 ViT-L (sat)", "set", "ViT-L", "RGB only. 493M satellite images."),
    "dinov2_vitb14_web": ("DINOv2 ViT-B (web)", "set", "ViT-B", "RGB only. Natural images."),
    "scalemae_large": ("Scale-MAE", "set", "ViT-L", "RGB only. fMoW, told input is 30 m."),
}


def report():
    # The leaderboard shows burn scars until it gains a task selector.
    rows = {p.stem: json.loads(p.read_text()) for p in sorted((RESULTS / "burn_scars").glob("*.json")) if p.stem in MODEL_INFO}
    models = []
    for key, scores in rows.items():
        label, family, params, note = MODEL_INFO.get(key, (key, "rare", "", ""))
        models.append({"id": key, "label": label, "family": family, "params": params, "note": note, "scores": scores})
    any_run = next(iter(rows.values()), {})
    conditions = (
        f"{len(models)} models. {any_run.get('n_train', 0):,} training and {any_run.get('n_test', 0):,} test patches "
        f"of 64\u00d764 px, fixed random sample. Speed on {platform.machine()} {torch_device()}, batch 64."
    )
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    data = json.dumps({"models": models, "conditions": conditions, "generated": generated}).replace("</", "<\\/")
    template = (Path(__file__).parent / "scorchedearth" / "leaderboard.html").read_text()
    Path("index.html").write_text(template.replace("__DATA__", data))
    print(f"Wrote index.html with {len(models)} models")


def torch_device() -> str:
    try:
        from scorchedearth.foundation import DEVICE
    except ImportError:
        return "CPU"
    return {"mps": "Apple GPU (MPS)", "cuda": "CUDA GPU"}.get(DEVICE, "CPU")


def main():
    p = argparse.ArgumentParser(prog="scorchedearth")
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download")
    d.add_argument("--tasks", default=",".join(TASKS))
    r = sub.add_parser("run")
    r.add_argument("--tasks", default=",".join(TASKS))
    r.add_argument("--models", default=",".join(MODELS))
    r.add_argument("--limit", type=int, help="Scenes or chips per split, for quick tests")
    r.add_argument("--n-train", type=int, help="Random training patches (fixed seed)")
    r.add_argument("--n-test", type=int, help="Random test patches (fixed seed)")
    sub.add_parser("report")
    args = p.parse_args()

    if args.cmd == "download":
        for t in args.tasks.split(","):
            print(f"{t}: data in {TASKS[t].download()}")
    elif args.cmd == "run":
        run(args.tasks.split(","), args.models.split(","), args.limit, args.n_train, args.n_test)
    elif args.cmd == "report":
        report()


if __name__ == "__main__":
    main()
