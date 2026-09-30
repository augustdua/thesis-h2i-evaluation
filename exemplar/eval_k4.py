"""Local Tables 4.18 / 4.19 for MatchFormer shallow64 checkpoint 197.

Native 512x512 hard assembly on HER2Match val (bilinear 1024->512).
Uses training DAB (Ruifrok HED ch2) and e_qp = mean|DAB| + mean|RGB|.

  python -u _scratch/eval_shallow64_tables_418_419.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT))

from matchformer_pair import (  # noqa: E402
    Her2JpegPairDataset,
    compute_global_ihc_topk,
    dab_hed_ch2,
    patches_4x4,
)
from matchformer_shallow64 import (  # noqa: E402
    ShallowMatchFormer64,
    hard_from_S,
    scores_S,
)

BF_THR = 0.3  # target-DAB positive mask (same thr used in C9 brown-frac logs)


def psnr(pred01: torch.Tensor, tgt01: torch.Tensor) -> float:
    mse = F.mse_loss(pred01, tgt01).item()
    if mse <= 1e-12:
        return 99.0
    return float(10.0 * np.log10(1.0 / mse))


def ssim_batch(pred01: torch.Tensor, tgt01: torch.Tensor) -> float:
    from pytorch_msssim import ssim as ssim_fn

    return float(
        ssim_fn(pred01, tgt01, data_range=1.0, size_average=True, win_size=11).item()
    )


def save_png01(path: Path, img01: torch.Tensor) -> None:
    x = img01.detach().float().clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
    Image.fromarray((x * 255.0 + 0.5).astype(np.uint8)).save(path)


def main() -> None:
    from src.config import load_config, resolve

    cfg = load_config(ROOT)
    ap = argparse.ArgumentParser()
    ap.add_argument("--her2", default=resolve(ROOT, cfg.get("her2_root") or ""))
    ap.add_argument("--ckpt", default=resolve(ROOT, cfg.get("exemplar_k4_ckpt") or ""))
    ap.add_argument("--donor", default=resolve(ROOT, cfg.get("donor_dir") or ""))
    ap.add_argument("--out", default=str(ROOT / "outputs" / "exemplar_k4"))
    ap.add_argument("--split", default="val", choices=("val", "test"))
    ap.add_argument("--shard", default="", help="rank/world e.g. 0/4")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--max-tiles", type=int, default=0, help="0 = all in split")
    ap.add_argument("--skip-fid", action="store_true")
    ap.add_argument("--skip-lpips", action="store_true")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    blank = [name for name, val in (
        ("--her2", args.her2),
        ("--ckpt", args.ckpt),
        ("--donor", args.donor),
    ) if not val]
    if blank:
        raise SystemExit(
            "Set paths in configs/chapter4.json or pass " + ", ".join(blank)
        )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    her2 = Path(args.her2)
    out = Path(args.out)
    gen_dir = out / "gen_hard512"
    real_dir = out / "real_ihc512"
    gen_dir.mkdir(parents=True, exist_ok=True)
    real_dir.mkdir(parents=True, exist_ok=True)

    donor_dir = Path(args.donor)
    he_a = torch.from_numpy(np.load(donor_dir / "donor_he_grouped_512.npy")).float()
    ihc_a = torch.from_numpy(np.load(donor_dir / "donor_ihc_grouped_512.npy")).float()
    if he_a.ndim == 3:
        he_a = he_a.permute(2, 0, 1).contiguous() / 255.0
        ihc_a = ihc_a.permute(2, 0, 1).contiguous() / 255.0
    elif he_a.max() > 1.5:
        he_a = he_a / 255.0
        ihc_a = ihc_a / 255.0
    if he_a.ndim == 3:
        he_a = he_a.unsqueeze(0)
        ihc_a = ihc_a.unsqueeze(0)
    he_a = he_a.to(device)
    ihc_a = ihc_a.to(device)
    patches_d = patches_4x4(ihc_a)

    blob = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = ShallowMatchFormer64(use_checkpoint=False).to(device).eval()
    model.load_state_dict(blob["model"], strict=False)
    epoch = int(blob.get("epoch", 197) or 197)
    print(f"ckpt epoch={epoch} device={device} params={sum(p.numel() for p in model.parameters())/1e6:.2f}M", flush=True)

    ds = Her2JpegPairDataset(her2, args.split)
    n_all = len(ds)
    if args.max_tiles > 0:
        ds.names = ds.names[: args.max_tiles]
    if args.shard:
        rank_s, world_s = args.shard.split("/", 1)
        rank, world = int(rank_s), int(world_s)
        if world < 1 or rank < 0 or rank >= world:
            raise SystemExit(f"bad --shard {args.shard}")
        ds.names = ds.names[rank::world]
        print(f"SHARD {rank}/{world} n={len(ds.names)}/{n_all}", flush=True)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=0)

    lpips_fn = None
    if not args.skip_lpips:
        import lpips

        lpips_fn = lpips.LPIPS(net="vgg", verbose=False).to(device).eval()

    sums = {
        "l1": 0.0,
        "ssim": 0.0,
        "psnr": 0.0,
        "lpips": 0.0,
        "dab_l1": 0.0,
        "dab_pos_l1": 0.0,
        "dab_pos_n": 0.0,
        "e_sel": 0.0,
        "e_ora": 0.0,
        "hit1": 0.0,
        "n_cells": 0.0,
    }
    dab_gen_vals: list[torch.Tensor] = []
    n = 0
    t0 = time.time()
    json_path = out / "tables_418_419.json"

    with torch.inference_mode():
        for he, ihc, names in loader:
            he = he.to(device)
            ihc = ihc.to(device)
            B = he.shape[0]
            Fb, Fa = model(he, he_a.expand(B, -1, -1, -1) if he_a.shape[0] == 1 else he_a)
            hard, idx = hard_from_S(Fb, Fa, patches_d.expand(B, -1, -1), tau=0.1)

            # image metrics
            for b in range(B):
                name = names[b] if isinstance(names[b], str) else names[b]
                stem = Path(name).stem
                h = hard[b : b + 1]
                t = ihc[b : b + 1]
                sums["l1"] += float(F.l1_loss(h, t).item())
                sums["ssim"] += ssim_batch(h, t)
                sums["psnr"] += psnr(h, t)
                if lpips_fn is not None:
                    sums["lpips"] += float(
                        lpips_fn(h * 2 - 1, t * 2 - 1).mean().item()
                    )
                dg = dab_hed_ch2(h)
                dt = dab_hed_ch2(t)
                sums["dab_l1"] += float(F.l1_loss(dg, dt).item())
                mask = dt > BF_THR
                if mask.any():
                    sums["dab_pos_l1"] += float((dg - dt).abs()[mask].mean().item())
                    sums["dab_pos_n"] += 1.0
                dab_gen_vals.append(dg.flatten().float().cpu())
                save_png01(gen_dir / f"{stem}.png", h)
                save_png01(real_dir / f"{stem}.png", t)

                # Table 4.19: selected vs oracle under e_qp (one Top-4 call)
                pq = patches_4x4(t)
                pos4, ora_err = compute_global_ihc_topk(pq, patches_d, k=4)
                pd = patches_d[0]
                sel = idx[b].long()
                sel_rgb = pd[sel]
                q_rgb = pq[0]
                e_rgb = (q_rgb - sel_rgb).abs().mean(dim=-1)
                dab_q = dab_hed_ch2(q_rgb.reshape(4096, 3, 8, 8)).reshape(4096, 64)
                dab_s = dab_hed_ch2(sel_rgb.reshape(4096, 3, 8, 8)).reshape(4096, 64)
                e_dab = (dab_q - dab_s).abs().mean(dim=-1)
                sums["e_sel"] += float((e_dab + e_rgb).mean().item())
                sums["e_ora"] += float(ora_err[:, 0].mean().item())
                hit = (sel.unsqueeze(1) == pos4.to(sel.device)).any(dim=1).float().mean()
                sums["hit1"] += float(hit.item())
                sums["n_cells"] += 4096.0
                n += 1

            if n % 25 == 0 or n >= len(ds):
                elapsed = time.time() - t0
                rate = n / max(elapsed, 1e-6)
                eta = (len(ds) - n) / max(rate, 1e-6)
                print(
                    f"[{n}/{len(ds)}] L1={sums['l1']/n:.4f} "
                    f"DAB-L1={sums['dab_l1']/n:.4f} "
                    f"E_sel={sums['e_sel']/n:.4f} E_ora={sums['e_ora']/n:.4f} "
                    f"rate={rate:.2f}/s eta_min={eta/60:.1f}",
                    flush=True,
                )
                partial = {
                    "status": "running",
                    "n": n,
                    "n_val_available": n_all,
                    "epoch": epoch,
                    "means": {k: (sums[k] / n if n else None) for k in sums if k != "n_cells"},
                    "E_selection": (sums["e_sel"] - sums["e_ora"]) / n if n else None,
                    "Hit_P@1_mean_over_tiles": sums["hit1"] / n if n else None,
                    "elapsed_s": elapsed,
                }
                json_path.write_text(json.dumps(partial, indent=2) + "\n", encoding="utf-8")

    means = {k: sums[k] / n for k in sums if k not in ("n_cells", "dab_pos_n")}
    dab_pos_l1 = sums["dab_pos_l1"] / max(sums["dab_pos_n"], 1.0)
    dab_cat = torch.cat(dab_gen_vals).numpy()
    dab_p95 = float(np.percentile(dab_cat, 95))

    fid = None
    if not args.skip_fid:
        from cleanfid import fid as clean_fid

        print("FID_START", flush=True)
        fid = float(
            clean_fid.compute_fid(
                str(gen_dir),
                str(real_dir),
                device=str(device),
                num_workers=0,
            )
        )
        print(f"FID_DONE {fid:.4f}", flush=True)

    result = {
        "status": "ok",
        "checkpoint": str(args.ckpt),
        "epoch": epoch,
        "her2": str(her2),
        "n": n,
        "n_val_available": n_all,
        "resolution": 512,
        "resize": "bilinear 1024->512 (Her2JpegPairDataset)",
        "table_4_18": {
            "FID": fid,
            "L1": means["l1"],
            "SSIM": means["ssim"],
            "PSNR": means["psnr"],
            "LPIPS_VGG": None if args.skip_lpips else means["lpips"],
            "DAB_p95": dab_p95,
            "DAB_L1": means["dab_l1"],
            "DAB_pos_DAB_L1": dab_pos_l1,
            "DAB_pos_thr": BF_THR,
            "DAB_pos_tiles_with_mask": int(sums["dab_pos_n"]),
        },
        "table_4_19": {
            "set": f"HER2Match {args.split} n={n} at 512",
            "E_selected": means["e_sel"],
            "E_oracle": means["e_ora"],
            "E_selection": means["e_sel"] - means["e_ora"],
            "e_qp": "mean|DAB|+mean|RGB| over 8x8 patch",
            "Hit_P@1_mean_over_tiles": means["hit1"],
        },
        "elapsed_s": time.time() - t0,
        "split": args.split,
        "shard": args.shard or None,
        "note": (
            "Native exemplar resolution 512; Table 4.1 direct translators use 256. "
            "FID via cleanfid on 512 PNG folders (Inception resize internal)."
        ),
    }
    json_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    print(f"WROTE {json_path}", flush=True)


if __name__ == "__main__":
    main()
