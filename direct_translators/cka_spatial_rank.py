"""A1/A2: fused linear CKA + spatial effective rank on 400 val tiles (seed 42).
A4: wavelet/Fourier shares for 64^2 and 32^2 Cross-decode on 240 hf_study tiles.

Reuses definitions from eval_c9_spatial_reff.py and eval_c9_fused_panel.py / hf_study.py.
"""
from __future__ import annotations

import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from c9_loader import import_train_c9  # noqa: E402
from src.cka import linear_cka  # noqa: E402
from src.config import load_config, resolve  # noqa: E402
from src.frequency import band_fractions, i_hf, luma as luma_np  # noqa: E402
from src.spatial_rank import spatial_reff  # noqa: E402
from src.tiles import pick_slide_spread  # noqa: E402

CFG = load_config(ROOT)
DATA = Path(resolve(ROOT, CFG.get("her2_root") or "")) if CFG.get("her2_root") else Path("__missing_her2__")
OUT_A12 = ROOT / "outputs" / "cka_spatial_rank.json"
OUT_A4 = ROOT / "outputs" / "frequency_grids.json"
STEMS = Path(resolve(ROOT, CFG.get("hf_stems") or "configs/hf_study_stems.json"))

CKPTS = {
    "pair_patchnce_c64": {
        "path": Path(resolve(ROOT, (CFG.get("translator_ckpts") or {}).get("pair_patchnce_c64") or "")),
        "label": "Pair PatchNCE C=64",
        "item": "A1",
    },
    "i2h_removed_c512": {
        "path": Path(resolve(ROOT, (CFG.get("translator_ckpts") or {}).get("i2h_removed_c512") or "")),
        "label": "IHC to H&E removed, C=512",
        "item": "A2",
    },
    "xdec_grid64": {
        "path": Path(resolve(ROOT, (CFG.get("translator_ckpts") or {}).get("xdec_grid64") or "")),
        "label": "Cross-decode C=64, 64x64 grid",
        "item": "A2",
        "force_z_hw": 64,
    },
    "xdec_grid32": {
        "path": Path(resolve(ROOT, (CFG.get("translator_ckpts") or {}).get("xdec_grid32") or "")),
        "label": "Cross-decode C=64, 32x32 grid",
        "item": "A2",
        "force_z_hw": 32,
    },
}


def load_rgb(path: str, tile: int = 256) -> torch.Tensor:
    im = Image.open(path).convert("RGB").resize((tile, tile), Image.BICUBIC)
    t = torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0)
    return t.permute(2, 0, 1) * 2.0 - 1.0


def summarise(vals: list[float], C: int) -> dict:
    a = np.asarray(vals, dtype=np.float64)
    med = float(np.median(a))
    q1 = float(np.percentile(a, 25))
    q3 = float(np.percentile(a, 75))
    return {
        "n_tiles": int(len(a)),
        "median": med,
        "Q1": q1,
        "Q3": q3,
        "IQR": q3 - q1,
        "mean": float(a.mean()),
        "std": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
        "r_eff_over_C_median": med / max(1, C),
        "C_measured": int(C),
    }


def make_uni_empty():
    import timm
    kw = dict(
        pretrained=False,
        img_size=224,
        patch_size=14,
        depth=24,
        num_heads=24,
        init_values=1e-5,
        embed_dim=1536,
        mlp_ratio=2.66667 * 2,
        num_classes=0,
        no_embed_class=True,
        mlp_layer=timm.layers.SwiGLUPacked,
        act_layer=nn.SiLU,
        reg_tokens=8,
        dynamic_img_size=True,
    )
    try:
        return timm.create_model("hf-hub:MahmoodLab/UNI2-h", **kw)
    except Exception:
        return timm.create_model(
            "vit_giant_patch14_dinov2",
            pretrained=False,
            img_size=224,
            patch_size=14,
            depth=24,
            num_heads=24,
            embed_dim=1536,
            num_classes=0,
            dynamic_img_size=True,
        )


def build_from_ckpt(ckpt: Path, device: str, force_z_hw: int | None = None):
    print(f"LOAD {ckpt}", flush=True)
    ck = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    args = ck.get("args") or {}
    if hasattr(args, "__dict__") and not isinstance(args, dict):
        args = vars(args)
    step = int(ck.get("step", -1))
    use_grn = bool(args.get("use_grn_decoder", False))
    use_grn_enc = bool(args.get("use_grn_encoder", False))
    layer_scale = float(args.get("layer_scale_init", 0.1) or 0.1)
    use_ls = not bool(args.get("no_layer_scale", False))
    use_ls_enc = bool(args.get("encoder_layer_scale", False))
    z_ch = args.get("z_ch")
    if z_ch is None:
        sd = ck["model"]
        bottle = sd.get("E_HE.bottle.weight")
        fw = sd["Fuse_HE.fuse.weight"]
        z_ch = int(bottle.shape[0]) if bottle is not None else int(fw.shape[1] - 256)
    else:
        z_ch = int(z_ch)
    computational_ch = int(args.get("computational_ch", 512) or 512)
    fuse_out_ch = int(args.get("fuse_out_ch", 512) or 512)
    expand_before_fuse = bool(args.get("expand_before_fuse", False))
    z_hw = int(force_z_hw if force_z_hw is not None else args.get("z_hw", 128) or 128)
    C9Paired = import_train_c9().C9Paired
    uni = make_uni_empty()
    model = C9Paired(
        uni_module=uni,
        taps=(13, 19),
        n_cls=1,
        n_reg=8,
        use_grn_decoder=use_grn,
        use_grn_encoder=use_grn_enc,
        layer_scale_init=layer_scale,
        use_layer_scale=use_ls,
        use_layer_scale_encoder=use_ls_enc,
        z_ch=z_ch,
        computational_ch=computational_ch,
        fuse_out_ch=fuse_out_ch,
        expand_before_fuse=expand_before_fuse,
        vib=False,
        z_hw=z_hw,
    )
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    if unexpected:
        raise RuntimeError(f"unexpected keys: {unexpected[:8]}")
    model.to(device).eval()
    meta = {
        "step": step,
        "z_ch": z_ch,
        "z_hw": z_hw,
        "ablate": args.get("ablate"),
        "nce_mode": args.get("nce_mode"),
        "missing_n": len(missing),
    }
    del ck
    return model, meta


def run_model(key: str, spec: dict, val_f: list[str], device: str) -> dict:
    path = Path(spec["path"])
    if not path.is_file():
        return {
            "status": "not_computed",
            "reason": f"checkpoint missing: {path}",
            "label": spec["label"],
            "item": spec["item"],
            "inputs": {"ckpt": str(path)},
        }
    t0 = time.time()
    model, meta = build_from_ckpt(path, device, force_z_hw=spec.get("force_z_hw"))
    pooled_he, pooled_ihc = [], []
    reffs_he, reffs_ihc = [], []
    C_meas = None
    z_hw_meas = None
    with torch.inference_mode():
        for i, ip in enumerate(val_f):
            hp = ip.replace("/IHC/", "/HE/").replace("\\IHC\\", "\\HE\\")
            he = load_rgb(hp).unsqueeze(0).to(device)
            ihc = load_rgb(ip).unsqueeze(0).to(device)
            zh = model.encode(he, "HE")[0]
            zi = model.encode(ihc, "IHC")[0]
            if C_meas is None:
                C_meas = int(zh.shape[0])
                z_hw_meas = int(zh.shape[1])
            pooled_he.append(zh.float().mean(dim=(1, 2)).cpu().numpy())
            pooled_ihc.append(zi.float().mean(dim=(1, 2)).cpu().numpy())
            reffs_he.append(spatial_reff(zh))
            reffs_ihc.append(spatial_reff(zi))
            if (i + 1) % 20 == 0 or i == 0:
                print(f"  {key} {i+1}/{len(val_f)} med_HE={np.median(reffs_he):.3f}", flush=True)
    X = np.stack(pooled_he, 0).astype(np.float64)
    Y = np.stack(pooled_ihc, 0).astype(np.float64)
    cka = float(linear_cka(X, Y))
    out = {
        "status": "computed",
        "label": spec["label"],
        "item": spec["item"],
        "inputs": {"ckpt": str(path), "n_tiles": len(val_f), "seed": 42, "data_root": str(DATA)},
        "n": len(val_f),
        "ckpt_meta": meta,
        "z_hw_measured": z_hw_meas,
        "linear_cka": cka,
        "spatial_reff_HE": summarise(reffs_he, C_meas),
        "spatial_reff_IHC": summarise(reffs_ihc, C_meas),
        "wall_s": time.time() - t0,
    }
    print(f"DONE {key} CKA={cka:.4f} med_r_eff_HE={out['spatial_reff_HE']['median']:.3f}", flush=True)
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return out


def run_a4(device: str) -> dict:
    """Generate 240 tiles for grid64/32 and measure frequency vs real IHC."""
    from src.frequency import summarise as summarise_freq

    stems = [s for s, _ in json.loads(STEMS.read_text(encoding="utf-8"))]
    real_rows = []
    for s in stems:
        rgb = np.asarray(
            Image.open(DATA / "IHC" / "val" / f"{s}.jpg").convert("RGB").resize((256, 256), Image.BICUBIC),
            dtype=np.float64,
        ) / 255.0
        y = luma_np(rgb)
        real_rows.append({"bands": band_fractions(y).tolist(), "i_hf": i_hf(y)})
    real = summarise_freq(real_rows)
    out = {
        "status": "computed",
        "n": len(stems),
        "inputs": {"stems": str(STEMS), "data": str(DATA)},
        "real_ihc": real,
        "cells": {},
    }
    for key in ("xdec_grid64", "xdec_grid32"):
        spec = CKPTS[key]
        path = Path(spec["path"])
        if not path.is_file():
            out["cells"][key] = {"status": "not_computed", "reason": f"missing ckpt {path}"}
            continue
        model, meta = build_from_ckpt(path, device, force_z_hw=spec.get("force_z_hw"))
        rows = []
        gen_dir = ROOT / "outputs" / "frequency_grids" / key
        gen_dir.mkdir(parents=True, exist_ok=True)
        with torch.inference_mode():
            for i, s in enumerate(stems):
                he_p = DATA / "HE" / "val" / f"{s}.jpg"
                he = load_rgb(str(he_p)).unsqueeze(0).to(device)
                gen = model.decode(model.encode(he, "HE"), "IHC")
                x = gen[0].detach().float().cpu().clamp(-1, 1)
                rgb = ((x + 1) * 0.5).permute(1, 2, 0).numpy()
                rgb = np.clip(rgb, 0, 1)
                Image.fromarray((rgb * 255).astype(np.uint8)).save(gen_dir / f"{s}.png")
                y = luma_np(rgb)
                rows.append({"bands": band_fractions(y).tolist(), "i_hf": i_hf(y)})
                if (i + 1) % 40 == 0:
                    print(f"A4 {key} {i+1}/{len(stems)}", flush=True)
        b = np.array([r["bands"] for r in rows])
        h = np.array([r["i_hf"] for r in rows])
        band_mean = b.mean(0).tolist()
        i_hf_mean = float(h.mean())
        out["cells"][key] = {
            "status": "computed",
            "label": spec["label"],
            "n": len(rows),
            "ckpt_meta": meta,
            "gen_dir": str(gen_dir),
            "band_frac_mean": band_mean,
            "i_hf_mean": i_hf_mean,
            "band_ratio_to_real": (np.array(band_mean) / np.array(real["band_frac_mean"])).tolist(),
            "i_hf_ratio_to_real": i_hf_mean / real["i_hf_mean"],
        }
        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        print(f"A4_DONE {key} bands={[round(x,4) for x in band_mean]} i_hf={i_hf_mean:.4f}", flush=True)
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="a12", help="a12,a4,all")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--models", default=",".join(CKPTS.keys()))
    args = ap.parse_args()
    if not CFG.get("her2_root"):
        raise SystemExit("Set her2_root in configs/chapter4.json")
    (ROOT / "outputs").mkdir(parents=True, exist_ok=True)

    val_all = sorted(glob.glob(str(DATA / "IHC" / "val" / "*.jpg")))
    val_f = pick_slide_spread(val_all, 400, 42)
    print(f"tiles={len(val_f)} device={args.device}", flush=True)

    if args.only in ("a12", "all"):
        report = {"models": {}, "n": 400, "seed": 42}
        want = [m.strip() for m in args.models.split(",") if m.strip()]
        for key in want:
            if key not in CKPTS:
                continue
            print(f"ROW_START {key}", flush=True)
            report["models"][key] = run_model(key, CKPTS[key], val_f, args.device)
            OUT_A12.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("A12_DONE", flush=True)

    if args.only in ("a4", "all"):
        a4 = run_a4(args.device)
        OUT_A4.write_text(json.dumps(a4, indent=2), encoding="utf-8")
        print("A4_CACHE_DONE", flush=True)


if __name__ == "__main__":
    main()
