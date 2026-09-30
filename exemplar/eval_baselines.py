"""Shared exemplar baselines plus K=10 and K=20 learned matchers.

Error is the oracle error: e = mean(|DAB|) + mean(|RGB|) on each 8x8 patch,
against the fixed library of 4,096 donors. Computed once per tile.

Random: mean of that error over all donors.
Constant: one donor index, the lowest mean error on the DAB-enriched training
list, then that same index at every val and test position.
H&E-NN: donor with smallest 8x8 H&E RGB L1; scored with the IHC error above.
Top-4: selected index is among the 4 lowest-error donors.

K=4 learned numbers are not recomputed.

    python -u _scratch/eval_exemplar_baselines.py
    python -u _scratch/eval_exemplar_baselines.py --limit 8 --const-idx 0
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
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT))

from matchformer_pair import (  # noqa: E402
    Her2JpegPairDataset,
    dab_hed_ch2,
    patches_4x4,
    stitch_4x4,
)
from matchformer_shallow64 import (  # noqa: E402
    ShallowMatchFormer64,
    hard_from_S,
)

BF_THR = 0.3
N_DONORS = 4096
N_POS = 4096
TOPK = 4
RANDOM_TOP4 = TOPK / float(N_DONORS)


def load_mosaic(path: Path, device: torch.device) -> torch.Tensor:
    """Match eval_shallow64_tables_418_419.py donor loading."""
    arr = np.load(path)
    t = torch.from_numpy(arr)
    if t.ndim == 3 and t.shape[-1] == 3:
        t = t.permute(2, 0, 1).contiguous()
    t = t.float()
    if float(t.max()) > 1.5:
        t = t / 255.0
    if t.ndim == 3:
        t = t.unsqueeze(0)
    return t.to(device)


def flat_dab(patches: torch.Tensor) -> torch.Tensor:
    """patches [N,192] or [B,N,192] -> DAB flat [...,64]."""
    lead = patches.shape[:-1]
    rgb = patches.reshape(-1, 3, 8, 8)
    dab = dab_hed_ch2(rgb).reshape(-1, 64)
    return dab.reshape(*lead, 64)


def assert_cdist_matches(device: torch.device) -> None:
    q = torch.rand(32, 192, device=device)
    d = torch.rand(64, 192, device=device)
    a = torch.cdist(q, d, p=1) / 192.0
    b = (q[:, None, :] - d[None, :, :]).abs().mean(dim=-1)
    err = (a - b).abs().max().item()
    if err > 1e-5:
        raise SystemExit(f"cdist L1 mismatch max_abs={err}")


class IHCOnly(Dataset):
    """Training pass only needs IHC. Same bilinear 1024 -> 512 as the pair dataset."""

    def __init__(self, root: Path, names: list[str]):
        self.dir = root / "IHC" / "train"
        self.names = list(names)

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, idx: int):
        name = self.names[idx]
        im = Image.open(self.dir / name).convert("RGB")
        im = im.resize((512, 512), Image.Resampling.BILINEAR)
        ihc = torch.from_numpy(np.asarray(im)).float().permute(2, 0, 1) / 255.0
        return ihc, name


def read_train_names(path: Path) -> list[str]:
    names = [
        ln.strip()
        for ln in path.read_text(encoding="utf-8").splitlines()
        if ln.strip().endswith(".jpg")
    ]
    if not names:
        raise SystemExit(f"no jpg names in {path}")
    return names


def load_matcher(path: Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    blob = torch.load(path, map_location=device, weights_only=False)
    model = ShallowMatchFormer64(use_checkpoint=False).to(device).eval()
    state = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
    model.load_state_dict(state, strict=True)
    meta = {
        "path": str(path),
        "epoch": int(blob.get("epoch", -1)) if isinstance(blob, dict) else -1,
        "n_pos": int(blob.get("n_pos", -1)) if isinstance(blob, dict) else -1,
        "step": int(blob.get("step", -1)) if isinstance(blob, dict) else -1,
    }
    return model, meta


def new_acc() -> dict:
    keys = (
        "e_rand",
        "e_const",
        "e_he",
        "e_ora",
        "e_k10",
        "e_k20",
        "hit_const",
        "hit_he",
        "hit_k10",
        "hit_k20",
        "hit_ora",
        "dab_const",
        "dab_he",
        "dab_ora",
        "dab_k10",
        "dab_k20",
    )
    acc = {k: 0.0 for k in keys}
    acc["n"] = 0
    acc["dab_pos_n"] = 0
    for k in ("const", "he", "ora", "k10", "k20"):
        acc[f"dabpos_{k}"] = 0.0
    return acc


def add_dab_pair(acc: dict, key: str, pred: torch.Tensor, dt: torch.Tensor, mask: torch.Tensor) -> None:
    """pred [1,3,H,W], dt/mask [1,H,W]. Target mask is shared across methods."""
    dg = dab_hed_ch2(pred)
    acc[f"dab_{key}"] += float((dg - dt).abs().mean().item())
    if mask.any():
        acc[f"dabpos_{key}"] += float((dg - dt).abs()[mask].mean().item())


@torch.inference_mode()
def score_queries(
    q_rgb: torch.Tensor,
    q_dab: torch.Tensor,
    q_he: torch.Tensor | None,
    d_rgb: torch.Tensor,
    d_dab: torch.Tensor,
    d_he: torch.Tensor | None,
    const_idx: int,
    idx_k10: torch.Tensor | None,
    idx_k20: torch.Tensor | None,
    max_q: int,
) -> dict:
    """q_* are [B,4096,D]. Returns per-tile python floats and index maps [B,4096]."""
    B, N, _ = q_rgb.shape
    device = q_rgb.device
    step = max(1, max_q // B)
    e_rand = torch.zeros(B, dtype=torch.float64, device=device)
    e_const = torch.zeros(B, dtype=torch.float64, device=device)
    e_he = torch.zeros(B, dtype=torch.float64, device=device)
    e_ora = torch.zeros(B, dtype=torch.float64, device=device)
    e_k10 = torch.zeros(B, dtype=torch.float64, device=device)
    e_k20 = torch.zeros(B, dtype=torch.float64, device=device)
    hit_const = torch.zeros(B, dtype=torch.float64, device=device)
    hit_he = torch.zeros(B, dtype=torch.float64, device=device)
    hit_k10 = torch.zeros(B, dtype=torch.float64, device=device)
    hit_k20 = torch.zeros(B, dtype=torch.float64, device=device)
    hit_ora = torch.zeros(B, dtype=torch.float64, device=device)
    nn_idx = torch.empty(B, N, dtype=torch.long, device=device) if q_he is not None else None
    ora_idx = torch.empty(B, N, dtype=torch.long, device=device)
    for row0 in range(0, N, step):
        row1 = min(row0 + step, N)
        rows = row1 - row0
        qr = q_rgb[:, row0:row1].reshape(-1, q_rgb.shape[-1])
        qd = q_dab[:, row0:row1].reshape(-1, q_dab.shape[-1])
        e_rgb = torch.cdist(qr, d_rgb, p=1) / float(q_rgb.shape[-1])
        e_dab = torch.cdist(qd, d_dab, p=1) / float(q_dab.shape[-1])
        e = (e_dab + e_rgb).view(B, rows, N_DONORS)
        top4 = torch.topk(e, k=TOPK, dim=-1, largest=False).indices
        ora = e.argmin(dim=-1)
        ora_idx[:, row0:row1] = ora
        e_rand += e.sum(dim=(1, 2)).double()
        e_const += e[:, :, const_idx].sum(dim=1).double()
        e_ora += e.gather(-1, ora.unsqueeze(-1)).squeeze(-1).sum(dim=1).double()
        hit_const += (top4 == const_idx).any(dim=-1).sum(dim=1).double()
        hit_ora += (top4 == ora.unsqueeze(-1)).any(dim=-1).sum(dim=1).double()
        if q_he is not None and d_he is not None and nn_idx is not None:
            qh = q_he[:, row0:row1].reshape(-1, q_he.shape[-1])
            he = torch.cdist(qh, d_he, p=1).view(B, rows, N_DONORS)
            nn = he.argmin(dim=-1)
            nn_idx[:, row0:row1] = nn
            e_he += e.gather(-1, nn.unsqueeze(-1)).squeeze(-1).sum(dim=1).double()
            hit_he += (top4 == nn.unsqueeze(-1)).any(dim=-1).sum(dim=1).double()
        if idx_k10 is not None:
            sel = idx_k10[:, row0:row1]
            e_k10 += e.gather(-1, sel.unsqueeze(-1)).squeeze(-1).sum(dim=1).double()
            hit_k10 += (top4 == sel.unsqueeze(-1)).any(dim=-1).sum(dim=1).double()
        if idx_k20 is not None:
            sel = idx_k20[:, row0:row1]
            e_k20 += e.gather(-1, sel.unsqueeze(-1)).squeeze(-1).sum(dim=1).double()
            hit_k20 += (top4 == sel.unsqueeze(-1)).any(dim=-1).sum(dim=1).double()
        del e, top4
    denom_all = float(N * N_DONORS)
    denom = float(N)
    out = {
        "e_rand": (e_rand / denom_all).tolist(),
        "e_const": (e_const / denom).tolist(),
        "e_he": (e_he / denom).tolist(),
        "e_ora": (e_ora / denom).tolist(),
        "e_k10": (e_k10 / denom).tolist(),
        "e_k20": (e_k20 / denom).tolist(),
        "hit_const": (hit_const / denom).tolist(),
        "hit_he": (hit_he / denom).tolist(),
        "hit_k10": (hit_k10 / denom).tolist(),
        "hit_k20": (hit_k20 / denom).tolist(),
        "hit_ora": (hit_ora / denom).tolist(),
    }
    return out, nn_idx, ora_idx


def accumulate(acc: dict, scored: dict, b: int, with_learned: bool, with_he: bool) -> None:
    acc["n"] += 1
    for key in ("e_rand", "e_const", "e_ora", "hit_const", "hit_ora"):
        acc[key] += float(scored[key][b])
    if with_he:
        acc["e_he"] += float(scored["e_he"][b])
        acc["hit_he"] += float(scored["hit_he"][b])
    if with_learned:
        acc["e_k10"] += float(scored["e_k10"][b])
        acc["e_k20"] += float(scored["e_k20"][b])
        acc["hit_k10"] += float(scored["hit_k10"][b])
        acc["hit_k20"] += float(scored["hit_k20"][b])


def mean_acc(acc: dict) -> dict:
    n = max(int(acc["n"]), 1)
    np_ = max(int(acc["dab_pos_n"]), 1)
    out = {"n": int(acc["n"]), "dab_pos_n": int(acc["dab_pos_n"])}
    for k, v in acc.items():
        if k in ("n", "dab_pos_n"):
            continue
        if k.startswith("dabpos_"):
            out[k] = v / np_ if acc["dab_pos_n"] else None
        else:
            out[k] = v / n
    out["random_top4"] = RANDOM_TOP4
    return out


def donor_sums_train(
    loader: DataLoader,
    d_rgb: torch.Tensor,
    d_dab: torch.Tensor,
    max_q: int,
    state_path: Path,
    resume_sum: torch.Tensor | None,
    resume_n: int,
) -> tuple[torch.Tensor, int]:
    device = d_rgb.device
    donor_sum = (
        resume_sum.to(device)
        if resume_sum is not None
        else torch.zeros(N_DONORS, dtype=torch.float64, device=device)
    )
    seen = 0
    t0 = time.time()
    n_total = len(loader.dataset)
    for ihc, names in loader:
        ihc = ihc.to(device, non_blocking=True)
        B = ihc.shape[0]
        # skip already finished tiles (loader is sequential, shuffle off)
        if seen + B <= resume_n:
            seen += B
            continue
        q = patches_4x4(ihc)
        q_dab = flat_dab(q)
        step = max(1, max_q // B)
        for row0 in range(0, N_POS, step):
            row1 = min(row0 + step, N_POS)
            qr = q[:, row0:row1].reshape(-1, q.shape[-1])
            qd = q_dab[:, row0:row1].reshape(-1, q_dab.shape[-1])
            e = torch.cdist(qr, d_rgb, p=1) / float(q.shape[-1])
            e = e + torch.cdist(qd, d_dab, p=1) / float(q_dab.shape[-1])
            e = e.view(B, row1 - row0, N_DONORS)
            donor_sum += e.sum(dim=(0, 1)).double()
            del e
        seen += B
        if seen % 100 < B or seen >= n_total:
            rate = seen / max(time.time() - t0, 1e-6)
            eta = (n_total - seen) / max(rate, 1e-6)
            print(
                f"[train {seen}/{n_total}] rate={rate:.2f}/s eta_min={eta/60:.1f}",
                flush=True,
            )
            torch.save(
                {
                    "donor_sum": donor_sum.detach().cpu(),
                    "train_n_done": seen,
                    "n_total": n_total,
                },
                state_path,
            )
        del q, q_dab, ihc
    return donor_sum, seen


def eval_split(
    loader: DataLoader,
    d_rgb: torch.Tensor,
    d_dab: torch.Tensor,
    d_he: torch.Tensor,
    donor_ihc_p: torch.Tensor,
    const_idx: int,
    model10,
    model20,
    he_a: torch.Tensor,
    max_q: int,
    acc: dict,
    skip_n: int,
    split_name: str,
    state_path: Path | None = None,
    save_state: bool = False,
) -> dict:
    device = d_rgb.device
    t0 = time.time()
    n_total = len(loader.dataset)
    seen = 0
    for he, ihc, names in loader:
        B = he.shape[0]
        if seen + B <= skip_n:
            seen += B
            continue
        he = he.to(device, non_blocking=True)
        ihc = ihc.to(device, non_blocking=True)
        with torch.inference_mode():
            Fb, Fa = model10(he, he_a)
            _hard10, idx10 = hard_from_S(Fb, Fa, donor_ihc_p, tau=0.1)
            del Fb, Fa
            Fb, Fa = model20(he, he_a)
            _hard20, idx20 = hard_from_S(Fb, Fa, donor_ihc_p, tau=0.1)
            del Fb, Fa
            q_rgb = patches_4x4(ihc)
            q_he = patches_4x4(he)
            q_dab = flat_dab(q_rgb)
            scored, nn_idx, ora_idx = score_queries(
                q_rgb, q_dab, q_he, d_rgb, d_dab, d_he, const_idx, idx10, idx20, max_q
            )
            dt = dab_hed_ch2(ihc)
            mask = dt > BF_THR
            idx_const = torch.full(
                (B, N_POS), const_idx, dtype=torch.long, device=device
            )
            imgs = {
                "const": stitch_4x4(donor_ihc_p[idx_const]),
                "he": stitch_4x4(donor_ihc_p[nn_idx]),
                "ora": stitch_4x4(donor_ihc_p[ora_idx]),
                "k10": stitch_4x4(donor_ihc_p[idx10]),
                "k20": stitch_4x4(donor_ihc_p[idx20]),
            }
            # hard_from_S already stitched; rebuild from idx so DAB uses the same patches
            del _hard10, _hard20
            for b in range(B):
                accumulate(acc, scored, b, with_learned=True, with_he=True)
                dt_b = dt[b : b + 1]
                mask_b = mask[b : b + 1]
                if mask_b.any():
                    acc["dab_pos_n"] += 1
                for key, img in imgs.items():
                    add_dab_pair(acc, key, img[b : b + 1], dt_b, mask_b)
            seen += B
        if seen % 25 < B or seen >= n_total:
            rate = (seen - skip_n) / max(time.time() - t0, 1e-6)
            eta = (n_total - seen) / max(rate, 1e-6)
            n = max(acc["n"], 1)
            print(
                f"[{split_name} {seen}/{n_total}] "
                f"E_rand={acc['e_rand']/n:.4f} E_const={acc['e_const']/n:.4f} "
                f"E_he={acc['e_he']/n:.4f} E_ora={acc['e_ora']/n:.4f} "
                f"E_k10={acc['e_k10']/n:.4f} E_k20={acc['e_k20']/n:.4f} "
                f"rate={rate:.2f}/s eta_min={eta/60:.1f}",
                flush=True,
            )
            if save_state and state_path is not None:
                blob = {}
                if state_path.is_file():
                    blob = torch.load(state_path, map_location="cpu", weights_only=False)
                blob[split_name] = {"acc": acc, "n_done": int(acc["n"])}
                torch.save(blob, state_path)
        del he, ihc, q_rgb, q_he, q_dab, nn_idx, ora_idx, idx10, idx20, imgs
    return acc


def pack_split(acc: dict) -> dict:
    m = mean_acc(acc)
    def row(err_key, hit_key, dab_key, pos_key, top4_override=None):
        return {
            "selected_error": m[err_key],
            "top4_rate": RANDOM_TOP4 if top4_override == "random" else m[hit_key],
            "dab_l1": None if dab_key is None else m[dab_key],
            "dab_pos_l1": None if pos_key is None else m[pos_key],
        }
    return {
        "n": m["n"],
        "dab_pos_n": m["dab_pos_n"],
        "dab_pos_thr": BF_THR,
        "random": row("e_rand", "hit_const", None, None, top4_override="random"),
        "constant": row("e_const", "hit_const", "dab_const", "dabpos_const"),
        "he_nn": row("e_he", "hit_he", "dab_he", "dabpos_he"),
        "learned_k10": row("e_k10", "hit_k10", "dab_k10", "dabpos_k10"),
        "learned_k20": row("e_k20", "hit_k20", "dab_k20", "dabpos_k20"),
        "oracle": row("e_ora", "hit_ora", "dab_ora", "dabpos_ora"),
    }


def main() -> None:
    from src.config import load_config, resolve

    cfg = load_config(ROOT)
    ap = argparse.ArgumentParser()
    ap.add_argument("--her2", default=resolve(ROOT, cfg.get("her2_root") or ""))
    ap.add_argument("--donor-dir", default=resolve(ROOT, cfg.get("donor_dir") or ""))
    ap.add_argument(
        "--train-list",
        default=resolve(ROOT, cfg.get("train_dab010_list") or ""),
    )
    ap.add_argument("--ckpt10", default=resolve(ROOT, cfg.get("exemplar_k10_ckpt") or ""))
    ap.add_argument("--ckpt20", default=resolve(ROOT, cfg.get("exemplar_k20_ckpt") or ""))
    ap.add_argument("--out", default=str(ROOT / "outputs" / "exemplar_k_eval.json"))
    ap.add_argument("--state", default=str(ROOT / "outputs" / "exemplar_k_eval_state.pt"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--max-q", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0, help="cap tiles per split; 0 = all")
    ap.add_argument("--const-idx", type=int, default=-1, help="skip train pass if >= 0")
    ap.add_argument("--splits", default="val,test")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--shard", default="", help="rank/world, stable sorted-name stride")
    args = ap.parse_args()
    shard_rank, shard_world = 0, 1
    if args.shard:
        rank_s, world_s = args.shard.split("/", 1)
        shard_rank, shard_world = int(rank_s), int(world_s)
        if shard_world < 1 or shard_rank < 0 or shard_rank >= shard_world:
            raise SystemExit(f"bad --shard {args.shard}")

    need = [
        ("--her2", args.her2),
        ("--donor-dir", args.donor_dir),
        ("--ckpt10", args.ckpt10),
        ("--ckpt20", args.ckpt20),
    ]
    if args.const_idx < 0:
        need.append(("--train-list", args.train_list))
    blank = [name for name, val in need if not val]
    if blank:
        raise SystemExit(
            "Set paths in configs/chapter4.json or pass " + ", ".join(blank)
        )
    (ROOT / "outputs").mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise SystemExit("need cuda")
    assert_cdist_matches(device)
    her2 = Path(args.her2)
    donor_dir = Path(args.donor_dir)
    state_path = Path(args.state)
    out_path = Path(args.out)

    he_a = load_mosaic(donor_dir / "donor_he_grouped_512.npy", device)
    ihc_a = load_mosaic(donor_dir / "donor_ihc_grouped_512.npy", device)
    donor_ihc_p = patches_4x4(ihc_a)[0]
    d_rgb = donor_ihc_p
    d_dab = flat_dab(donor_ihc_p)
    d_he = patches_4x4(he_a)[0]
    if d_rgb.shape[0] != N_DONORS:
        raise SystemExit(f"donor count {d_rgb.shape[0]} != {N_DONORS}")
    print(
        f"donor he {tuple(he_a.shape)} ihc {tuple(ihc_a.shape)} device={device}",
        flush=True,
    )

    model10, meta10 = load_matcher(Path(args.ckpt10), device)
    model20, meta20 = load_matcher(Path(args.ckpt20), device)
    print(f"k10 {meta10}", flush=True)
    print(f"k20 {meta20}", flush=True)

    resume = {}
    if state_path.is_file() and args.limit == 0 and args.const_idx < 0:
        resume = torch.load(state_path, map_location="cpu", weights_only=False)
        print(f"resume keys={list(resume.keys())}", flush=True)

    const_idx = int(args.const_idx)
    const_train_mean = None
    train_n = None
    if const_idx < 0:
        names = read_train_names(Path(args.train_list))
        if args.limit > 0:
            names = names[: args.limit]
        train_ds = IHCOnly(her2, names)
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
        )
        prev_sum = resume.get("donor_sum")
        prev_n = int(resume.get("train_n_done", 0)) if prev_sum is not None else 0
        donor_sum, seen = donor_sums_train(
            train_loader, d_rgb, d_dab, args.max_q, state_path, prev_sum, prev_n
        )
        const_idx = int(donor_sum.argmin().item())
        const_train_mean = float(donor_sum[const_idx].item() / (seen * N_POS))
        train_n = seen
        n_tie = int((donor_sum <= donor_sum[const_idx] + 1e-8).sum().item())
        print(
            f"CONSTANT idx={const_idx} train_mean={const_train_mean:.6f} "
            f"n_tiles={seen} n_tie_within_1e-8={n_tie}",
            flush=True,
        )
        torch.save(
            {
                "donor_sum": donor_sum.detach().cpu(),
                "train_n_done": seen,
                "const_idx": const_idx,
                "const_train_mean": const_train_mean,
                "n_tie": n_tie,
                "train_list": args.train_list,
            },
            state_path,
        )
    else:
        print(f"CONSTANT idx={const_idx} (given, train pass skipped)", flush=True)

    splits_out = {}
    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        ds = Her2JpegPairDataset(her2, split)
        if args.limit > 0:
            ds.names = ds.names[: args.limit]
        n_full = len(ds)
        print(f"split {split} n_full={n_full}", flush=True)
        if args.limit == 0 and not args.shard:
            expect = {"val": 3582, "test": 5980}.get(split)
            if expect is not None and n_full != expect:
                raise SystemExit(f"{split} n={n_full} != expected {expect}")
        if args.shard:
            if args.limit == 0:
                expect = {"val": 3582, "test": 5980}.get(split)
                if expect is not None and n_full != expect:
                    raise SystemExit(f"{split} n={n_full} != expected {expect}")
            ds.names = ds.names[shard_rank::shard_world]
            print(
                f"shard {args.shard} n={len(ds)}/{n_full}",
                flush=True,
            )
        loader = DataLoader(
            ds,
            batch_size=args.batch,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
        )
        acc = new_acc()
        skip_n = 0
        if args.limit == 0 and isinstance(resume.get(split), dict):
            acc = resume[split]["acc"]
            skip_n = int(resume[split]["n_done"])
            print(f"resume {split} from {skip_n}", flush=True)
        eval_split(
            loader,
            d_rgb,
            d_dab,
            d_he,
            donor_ihc_p,
            const_idx,
            model10,
            model20,
            he_a,
            args.max_q,
            acc,
            skip_n,
            split,
            state_path=state_path,
            save_state=(args.limit == 0 and not args.shard),
        )
        packed = pack_split(acc)
        splits_out[split] = packed
        if args.limit == 0 and not args.shard:
            blob = torch.load(state_path, map_location="cpu", weights_only=False)
            blob[split] = {"acc": acc, "n_done": acc["n"]}
            blob["const_idx"] = const_idx
            blob["const_train_mean"] = const_train_mean
            torch.save(blob, state_path)
        print(
            f"DONE {split} n={packed['n']} "
            f"ora={packed['oracle']['selected_error']:.6f} "
            f"const={packed['constant']['selected_error']:.6f} "
            f"he={packed['he_nn']['selected_error']:.6f} "
            f"k10={packed['learned_k10']['selected_error']:.6f} "
            f"k20={packed['learned_k20']['selected_error']:.6f}",
            flush=True,
        )
        if split == "val" and args.limit == 0 and not args.shard:
            ora = packed["oracle"]["selected_error"]
            if abs(ora - 0.0771973831527089) > 1e-4:
                raise SystemExit(
                    f"val oracle {ora:.6f} does not match stored 0.077197; "
                    "not writing final json"
                )

    result = {
        "error": "mean(|DAB|)+mean(|RGB|) on 8x8, lambdas 1 and 1, 4096 donors",
        "top4": "selected donor is among the 4 lowest-error donors",
        "random_top4_exact": RANDOM_TOP4,
        "dab_pos_thr": BF_THR,
        "train_list": args.train_list,
        "train_n": train_n,
        "const_idx": const_idx,
        "const_train_mean_error": const_train_mean,
        "shard": args.shard or None,
        "k10_checkpoint": meta10,
        "k20_checkpoint": meta20,
        "k10_requested_epoch_by_ihc_recall_at_1": 192,
        "k20_requested_epoch_by_ihc_recall_at_1": 188,
        "checkpoint_note": (
            "Saved every 25 epochs. Epochs 192 and 188 are not on the volume. "
            "Closest saved file is epoch 200 for both K=10 (|200-192|=8) and "
            "K=20 (|200-188|=12; epoch 175 is 13 away)."
        ),
        "k4_printed_not_recomputed": {
            "selected_error_val": 0.2790,
            "selected_error_test": 0.5477,
            "top4_val": 0.0998,
            "top4_test": 0.0793,
            "dab_l1_val": 0.1989,
            "dab_l1_test": 0.4216,
            "dab_pos_l1_val": 0.6574,
            "dab_pos_l1_test": 0.7902,
            "oracle_val": 0.0772,
            "oracle_test": 0.0624,
        },
        "splits": splits_out,
        "limit": args.limit,
    }
    if args.limit == 0:
        out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"WROTE {out_path}", flush=True)
    else:
        smoke = out_path.with_name("exemplar_k_eval_smoke.json")
        smoke.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"WROTE {smoke}", flush=True)
    print(json.dumps(result["splits"], indent=2), flush=True)


if __name__ == "__main__":
    main()
