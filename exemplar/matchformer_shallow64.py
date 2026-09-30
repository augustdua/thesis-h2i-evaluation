"""Shallow 64x64 MatchFormer: stem + Stage1 (SSC) only + global IHC positives.

Hypothesis from scale diagnostic (last.pt ep43):
  hard64 DAB ~1.87 beats full FPN hard DAB ~2.16; deeper stages corrupt retrieval.
  Old Top-32-mined positives could never discover true IHC donors outside the shortlist.

This model:
  - shared stem + Stage1 SELF/SELF/CROSS at 64x64 (no stage2-4, no FPN)
  - 4096-way multi-positive InfoNCE with GLOBAL IHC positives (precomputed)
  - inference: argmax(S) hard paste of donor IHC 8x8

    python -u bbdm_BCI/matchformer_shallow64.py --mode precompute_global
    python -u bbdm_BCI/matchformer_shallow64.py --mode train --epochs 100
    python -u bbdm_BCI/matchformer_shallow64.py --mode sample_val --ckpt ... --n 8
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# Reuse shared data / IHC helpers from the existing MatchFormer module.
from matchformer_pair import (  # noqa: E402
    CORR_N_POS,
    GLOBAL_POS_DIR,
    GLOBAL_POS_STORE_K,
    GRAD_CLIP,
    HER2_ROOT,
    LAMBDA_DAB,
    LAMBDA_RGB,
    PATCH,
    TAU,
    TILE,
    VAL_PANEL_N,
    Her2JpegPairDataset,
    Stage,
    compute_global_ihc_topk,
    dab_hed_ch2,
    gather_pos_idx_batch,
    load_donor,
    load_global_pos_cache,
    oracle_ihc_target_recall,
    patches_4x4,
    require_cuda,
    save_global_pos_tile,
    stitch_4x4,
    _donor_paths,
    _gpu_mem_str,
    _micro_for,
    _to_u8,
    count_params,
)

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = ROOT / "_scratch"
OUT_DIR = Path(
    os.environ.get(
        "MATCHFORMER_OUT",
        str(SCRATCH / "p25_ddm_v2_x0_id" / "matchformer_shallow64"),
    )
)
C_OUT = 128
GRID = 64
N_CELLS = GRID * GRID  # 4096
LR = float(os.environ.get("MATCHFORMER_LR", "1e-4"))
NUM_EPOCHS = 100
BATCH_SIZE = 64


# ---------------------------------------------------------------------------
# Model: stem + Stage1 only
# ---------------------------------------------------------------------------


class ShallowMatchFormer64(nn.Module):
    """Shared stem + Stage1 (SSC) at 64x64. No deeper pyramid / FPN."""

    def __init__(self, use_checkpoint: bool = True):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.stem = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=7, stride=8, padding=3, bias=False),
            nn.GroupNorm(8, 128),
            nn.GELU(),
        )
        # Same Stage1 pattern as the full MatchFormer baseline.
        self.stage1 = Stage(128, num_heads=4, pattern="SSC")
        # Same-scale projection (replaces the old FPN head at 64-only).
        self.proj = nn.Sequential(
            nn.Conv2d(128, C_OUT, kernel_size=1, bias=False),
            nn.GroupNorm(8, C_OUT),
            nn.GELU(),
            nn.Conv2d(C_OUT, C_OUT, kernel_size=3, padding=1, bias=False),
        )

    def _run_stage(self, fb: torch.Tensor, fa: torch.Tensor):
        if self.use_checkpoint and self.training:
            return checkpoint(self.stage1, fb, fa, use_reentrant=False)
        return self.stage1(fb, fa)

    def encode(self, he_b: torch.Tensor, he_a: torch.Tensor):
        """he_*: B,3,512,512 in [0,1] -> Fb, Fa each B,128,64,64."""
        B = he_b.shape[0]
        Ba = he_a.shape[0]
        fb = self.stem(he_b)
        if Ba == 1 and B > 1:
            fa = self.stem(he_a).expand(B, -1, -1, -1).contiguous()
        else:
            fa = self.stem(he_a)
        fb, fa = self._run_stage(fb, fa)
        Fb = self.proj(fb)
        Fa = self.proj(fa)
        return Fb, Fa

    def forward(self, he_b: torch.Tensor, he_a: torch.Tensor):
        return self.encode(he_b, he_a)

    def features_flat(self, he_b: torch.Tensor, he_a: torch.Tensor):
        """Return L2-normalized [B,4096,128] query and [B,4096,128] donor features."""
        Fb, Fa = self.encode(he_b, he_a)
        fq = F.normalize(Fb.flatten(2).transpose(1, 2).float(), dim=-1)
        fd = F.normalize(Fa.flatten(2).transpose(1, 2).float(), dim=-1)
        return fq, fd


def scores_S(
    Fb: torch.Tensor, Fa: torch.Tensor, tau: float = TAU, chunk: int = 256
) -> torch.Tensor:
    """Cosine/tau matrix [B,4096,4096] from spatial feature maps."""
    fb = F.normalize(Fb.flatten(2).transpose(1, 2).float(), dim=-1)
    fa = F.normalize(Fa.flatten(2).transpose(1, 2).float(), dim=-1)
    B, N, _ = fb.shape
    S = fb.new_empty(B, N, N)
    for i in range(0, N, chunk):
        i1 = min(i + chunk, N)
        S[:, i:i1] = torch.bmm(fb[:, i:i1], fa.transpose(1, 2)) / tau
    return S


def global_contrastive_loss(
    Fb: torch.Tensor,
    Fa: torch.Tensor,
    pos_idx: torch.Tensor,
    tau: float = TAU,
    chunk: int = 256,
) -> torch.Tensor:
    """4096-way multi-positive InfoNCE with stationary global IHC positives.

    L = mean_q ( logsumexp_p S_qp - logsumexp_{p in P_q} S_qp )
    pos_idx: B,N,n_pos
    """
    S = scores_S(Fb, Fa, tau=tau, chunk=chunk)
    B, N, _ = S.shape
    if pos_idx.shape[:2] != (B, N):
        raise ValueError(f"pos_idx {tuple(pos_idx.shape)} vs S {(B, N, N)}")
    pos_idx = pos_idx.long()
    log_den = torch.logsumexp(S, dim=-1)
    pos_scores = torch.gather(S, -1, pos_idx)
    log_num = torch.logsumexp(pos_scores, dim=-1)
    L = (log_den - log_num).mean()
    del S, log_den, log_num, pos_scores
    return L


@torch.no_grad()
def hard_from_S(Fb: torch.Tensor, Fa: torch.Tensor, patches_donor: torch.Tensor, tau: float = TAU):
    """Inference: argmax(S) -> hard paste of real donor IHC 8x8 patches."""
    S = scores_S(Fb, Fa, tau=tau)
    idx = S.argmax(dim=-1)  # B,N
    B, N = idx.shape
    if patches_donor.dim() == 2:
        patches_donor = patches_donor.unsqueeze(0)
    if patches_donor.shape[0] == 1 and B > 1:
        patches_donor = patches_donor.expand(B, -1, -1)
    expand = idx.unsqueeze(-1).expand(-1, -1, patches_donor.shape[-1])
    hard_p = torch.gather(patches_donor, 1, expand)
    return stitch_4x4(hard_p), idx


def save_recon_panel(
    path: Path,
    he_b: torch.Tensor,
    true_b: torch.Tensor,
    hard_b: torch.Tensor,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = he_b.shape[0]
    fig, axs = plt.subplots(n, 3, figsize=(9, 3.0 * n))
    if n == 1:
        axs = np.array([axs])
    titles = ["HE", "true IHC", "hard argmax(S)"]
    for i in range(n):
        imgs = [_to_u8(he_b[i : i + 1]), _to_u8(true_b[i : i + 1]), _to_u8(hard_b[i : i + 1])]
        for j, (im, title) in enumerate(zip(imgs, titles)):
            axs[i, j].imshow(im)
            if i == 0:
                axs[i, j].set_title(title, fontsize=10)
            axs[i, j].axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Precompute / train
# ---------------------------------------------------------------------------


def precompute_global(args):
    """Same global-IHC Top-K labels as Experiment A (reusable cache)."""
    device = require_cuda(getattr(args, "device", "cuda"))
    store_k = int(
        getattr(args, "store_k", None)
        or os.environ.get("MATCHFORMER_GLOBAL_POS_STORE_K", GLOBAL_POS_STORE_K)
    )
    her2 = Path(os.environ.get("MATCHFORMER_HER2", str(HER2_ROOT)))
    train_list = os.environ.get("MATCHFORMER_TRAIN_LIST", "").strip()
    if train_list:
        train_names = [
            ln.strip()
            for ln in Path(train_list).read_text(encoding="utf-8").splitlines()
            if ln.strip().endswith(".jpg")
        ]
    else:
        train_names = Her2JpegPairDataset(her2, "train").names
    val_list = os.environ.get("MATCHFORMER_VAL_LIST", "").strip()
    val_names = []
    if val_list:
        val_names = [
            ln.strip()
            for ln in Path(val_list).read_text(encoding="utf-8").splitlines()
            if ln.strip().endswith(".jpg")
        ]
    names = list(dict.fromkeys([*train_names, *val_names]))
    GLOBAL_POS_DIR.mkdir(parents=True, exist_ok=True)
    _he_a, ihc_a = load_donor()
    patches_d = patches_4x4(ihc_a.to(device))
    he_train, he_val = her2 / "HE" / "train", her2 / "HE" / "val"
    print(
        f"shallow64 precompute_global n={len(names)} out={GLOBAL_POS_DIR} "
        f"k={store_k}",
        flush=True,
    )
    t0 = time.time()
    done = skipped = 0
    for name in names:
        stem = Path(name).stem
        out_p = GLOBAL_POS_DIR / f"{stem}.pt"
        if out_p.is_file() and not getattr(args, "force", False):
            try:
                blob = torch.load(out_p, map_location="cpu", weights_only=False)
                have_k = int(blob.get("k", blob["idx"].shape[-1]))
            except Exception:
                have_k = 0
            if have_k >= store_k:
                skipped += 1
                done += 1
                continue
        if (he_train / name).is_file():
            split = "train"
        elif (he_val / name).is_file():
            split = "val"
        else:
            raise FileNotFoundError(name)
        _he, ihc, _n = Her2JpegPairDataset(her2, split, names=[name])[0]
        pq = patches_4x4(ihc.unsqueeze(0).to(device))
        idx, err = compute_global_ihc_topk(pq, patches_d, k=store_k)
        save_global_pos_tile(out_p, idx, err)
        done += 1
        if done % 50 == 0 or done == len(names):
            print(
                f"  {done}/{len(names)} skipped={skipped} "
                f"elapsed_s={time.time()-t0:.1f}",
                flush=True,
            )
    print(f"DONE precompute wrote/kept={done} skipped={skipped}", flush=True)


def train(args):
    device = require_cuda(getattr(args, "device", "cuda"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "ckpts").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "panels").mkdir(parents=True, exist_ok=True)
    ckpt_dir = OUT_DIR / "ckpts"
    panel_dir = OUT_DIR / "panels"
    ckpt_stem = os.environ.get("MATCHFORMER_CKPT_STEM", "matchformer_shallow64")
    dtype_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)

    her2 = Path(os.environ.get("MATCHFORMER_HER2", str(HER2_ROOT)))
    train_list = os.environ.get("MATCHFORMER_TRAIN_LIST", "").strip()
    if train_list:
        train_names = [
            ln.strip()
            for ln in Path(train_list).read_text(encoding="utf-8").splitlines()
            if ln.strip().endswith(".jpg")
        ]
        train_ds = Her2JpegPairDataset(her2, "train", names=train_names)
    else:
        train_ds = Her2JpegPairDataset(her2, "train")
    val_list = os.environ.get("MATCHFORMER_VAL_LIST", "").strip()
    if val_list:
        panel_names = [
            ln.strip()
            for ln in Path(val_list).read_text(encoding="utf-8").splitlines()
            if ln.strip().endswith(".jpg")
        ]
    else:
        panel_names = Her2JpegPairDataset(her2, "val").names[:VAL_PANEL_N]
    panel_ds = Her2JpegPairDataset(her2, "val", names=panel_names)

    batch_size = int(getattr(args, "batch_size", BATCH_SIZE))
    epochs = int(getattr(args, "epochs", NUM_EPOCHS))
    n_pos = int(
        getattr(args, "n_pos", None)
        or os.environ.get("MATCHFORMER_N_POS", CORR_N_POS)
    )
    store_k_need = max(n_pos, GLOBAL_POS_STORE_K)
    ckpt_every = int(os.environ.get("MATCHFORMER_CKPT_EVERY", "1"))
    micro = _micro_for(batch_size)
    n_workers = int(os.environ.get("MATCHFORMER_WORKERS", "4"))

    from torch.utils.data import DataLoader

    loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=n_workers > 0,
    )
    panel_loader = DataLoader(panel_ds, batch_size=len(panel_names), shuffle=False)

    he_a, ihc_a = load_donor()
    he_a = he_a.to(device)
    patches_a = patches_4x4(ihc_a.to(device))  # 1,4096,192
    donor_he_path, _ = _donor_paths()

    he_panel, ihc_panel, _ = next(iter(panel_loader))
    he_panel = he_panel.to(device)
    ihc_panel = ihc_panel.to(device)

    print(
        f"GLOBAL_POS_DIR={GLOBAL_POS_DIR} store_k>={store_k_need} "
        f"train_n_pos={n_pos} ckpt_every={ckpt_every}",
        flush=True,
    )
    pos_cache = load_global_pos_cache(GLOBAL_POS_DIR, train_ds.names, n_pos=n_pos)
    panel_pos = gather_pos_idx_batch(
        load_global_pos_cache(GLOBAL_POS_DIR, panel_names, n_pos=n_pos),
        panel_names,
        device,
    )

    use_ckpt = os.environ.get("MATCHFORMER_GRAD_CKPT", "0").strip() not in ("0", "false", "False")
    model = ShallowMatchFormer64(use_checkpoint=use_ckpt).to(device)
    init_ckpt = os.environ.get("MATCHFORMER_INIT_CKPT", "").strip()
    init_epoch = 0
    if init_ckpt:
        blob = torch.load(init_ckpt, map_location=device, weights_only=False)
        state = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
        # Warm-start stem (+ stage1 if shapes match) from full MatchFormer.
        missing, unexpected = model.load_state_dict(state, strict=False)
        if isinstance(blob, dict):
            init_epoch = int(blob.get("epoch", 0))
        print(
            f"warm start {init_ckpt} source_epoch={init_epoch} strict=False "
            f"missing={len(missing)} unexpected={len(unexpected)}",
            flush=True,
        )

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    console_path = OUT_DIR / "console.txt"
    epoch_offset = int(os.environ.get("MATCHFORMER_EPOCH_OFFSET", "0"))
    header = (
        f"MatchFormer SHALLOW64 train\n"
        f"her2={her2} n_train={len(train_ds)} n_val_panel={len(panel_ds)} "
        f"lr={LR} panel={panel_names}\n"
        f"donor={donor_he_path}\n"
        f"batch={batch_size} micro={micro} epochs={epochs} "
        f"epoch_offset={epoch_offset} "
        f"tau={TAU} n_pos={n_pos} loss=4096-way global IHC InfoNCE\n"
        f"arch=stem+stage1(SSC)+proj64  NO stage2-4/FPN/GAN\n"
        f"device={device} bf16=True params={count_params(model)/1e6:.2f}M "
        f"ckpt_stem={ckpt_stem} init_ckpt={init_ckpt or 'none'} "
        f"source_epoch={init_epoch}\n"
    )
    print(header, end="", flush=True)
    append_console = os.environ.get("MATCHFORMER_CONSOLE_APPEND", "0").strip() not in (
        "0",
        "false",
        "False",
        "",
    )
    if append_console and console_path.is_file():
        with console_path.open("a", encoding="utf-8") as f:
            f.write("\n--- continue ---\n" + header)
        lines = [header]
    else:
        console_path.write_text(header, encoding="utf-8")
        lines = [header]

    def log_line(s: str):
        print(s, flush=True)
        with console_path.open("a", encoding="utf-8") as f:
            f.write(s + "\n")
        lines.append(s + "\n")

    global_step = 0
    if init_ckpt:
        # Re-read step from the warm-start blob already loaded above when possible.
        try:
            _b = torch.load(init_ckpt, map_location="cpu", weights_only=False)
            if isinstance(_b, dict) and "step" in _b:
                global_step = int(_b["step"])
        except Exception:
            pass
    t0 = time.time()
    for epoch_i in range(1, epochs + 1):
        epoch = epoch_offset + epoch_i
        model.train()
        L_acc = L_hard_acc = L_dab_acc = 0.0
        n_seen = 0
        for he_b, ihc_b, names in loader:
            he_b = he_b.to(device, non_blocking=True)
            ihc_b = ihc_b.to(device, non_blocking=True)
            B = he_b.shape[0]
            pos_b = gather_pos_idx_batch(pos_cache, list(names), device)
            opt.zero_grad(set_to_none=True)
            for s in range(0, B, micro):
                e = min(s + micro, B)
                mb = e - s
                with dtype_ctx:
                    Fb, Fa = model(he_b[s:e], he_a)
                loss = global_contrastive_loss(Fb, Fa, pos_b[s:e], tau=TAU) * (mb / B)
                loss.backward()
                L_v = float(loss.detach()) * (B / mb)
                L_acc += L_v * mb
                with torch.no_grad():
                    hard, _ = hard_from_S(Fb.detach(), Fa.detach(), patches_a)
                    L_hard_acc += F.l1_loss(hard, ihc_b[s:e]).item() * mb
                    L_dab_acc += (
                        F.l1_loss(dab_hed_ch2(hard), dab_hed_ch2(ihc_b[s:e])).item() * mb
                    )
                    del hard
                del Fb, Fa, loss
                n_seen += mb
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            global_step += 1
            del he_b, ihc_b, pos_b

        L_v = L_acc / max(n_seen, 1)
        L_hard = L_hard_acc / max(n_seen, 1)
        L_dab = L_dab_acc / max(n_seen, 1)
        msg = (
            f"epoch {epoch:04d}  step {global_step}  "
            f"L_contrast={L_v:.4f}  L_hard={L_hard:.4f}  L_dab_hard={L_dab:.4f}  "
            f"{_gpu_mem_str()}"
        )
        log_line(msg)

        save_ckpt = (
            ckpt_every <= 1
            or (epoch_i % ckpt_every == 0)
            or (epoch_i == epochs)
        )
        if save_ckpt:
            ckpt_path = ckpt_dir / f"{ckpt_stem}_epoch{epoch:04d}.pt"
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "step": global_step,
                    "n_pos": n_pos,
                },
                ckpt_path,
            )
            # also write last.pt
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "step": global_step,
                    "n_pos": n_pos,
                },
                ckpt_dir / "last.pt",
            )
            log_line(f"  saved ckpt {ckpt_path.name}")
        else:
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "step": global_step,
                    "n_pos": n_pos,
                },
                ckpt_dir / "last.pt",
            )

        model.eval()
        with torch.no_grad(), dtype_ctx:
            Fb_p, Fa_p = model(he_panel, he_a)
        hard_p, _ = hard_from_S(Fb_p, Fa_p, patches_a)
        png = panel_dir / f"recon_epoch{epoch:04d}.png"
        save_recon_panel(png, he_panel.cpu(), ihc_panel.cpu(), hard_p.cpu())
        log_line(f"  saved panel {png.name}")
        recalls = oracle_ihc_target_recall(Fb_p, Fa_p, panel_pos, tau=TAU)
        # also hard DAB on panel
        dab_h = float(F.l1_loss(dab_hed_ch2(hard_p), dab_hed_ch2(ihc_panel)).item())
        rgb_h = float(F.l1_loss(hard_p, ihc_panel).item())
        rec_s = "  ".join(f"{k}={v:.4f}" for k, v in recalls.items())
        log_line(f"  panel hard_RGB={rgb_h:.4f} hard_DAB={dab_h:.4f}  {rec_s}")
        del Fb_p, Fa_p, hard_p
        model.train()

        vol_name = os.environ.get("MATCHFORMER_VOL_COMMIT", "").strip()
        commit_every = int(os.environ.get("MATCHFORMER_VOL_COMMIT_EVERY", "5"))
        if vol_name and (epoch_i % commit_every == 0 or epoch_i == epochs):
            try:
                import modal as _modal

                _modal.Volume.from_name(vol_name).commit()
                log_line(f"  volume commit {vol_name}")
            except Exception as exc:
                log_line(f"  volume commit failed {type(exc).__name__}: {exc}")
        gc.collect()
        torch.cuda.empty_cache()

    elapsed = time.time() - t0
    log_line(
        f"DONE epochs={epochs} abs_epoch={epoch_offset + epochs} "
        f"steps={global_step} elapsed_s={elapsed:.1f}"
    )
    (OUT_DIR / "train_log.txt").write_text("".join(lines), encoding="utf-8")
    vol_name = os.environ.get("MATCHFORMER_VOL_COMMIT", "").strip()
    if vol_name:
        try:
            import modal as _modal

            _modal.Volume.from_name(vol_name).commit()
            log_line(f"  volume commit {vol_name} (final)")
        except Exception as exc:
            log_line(f"  volume commit failed {type(exc).__name__}: {exc}")


def _read_name_list(path: str | Path) -> list[str]:
    p = Path(path)
    if not p.is_file():
        return []
    return [
        ln.strip()
        for ln in p.read_text(encoding="utf-8").splitlines()
        if ln.strip().endswith(".jpg")
    ]


def _pick_sample_names(
    her2: Path,
    n: int,
    offset: int,
    skip_panel: bool,
    split: str,
) -> list[tuple[str, str]]:
    """Return [(split, name), ...] for sample_val.

    On this volume val has only 4 tiles (the training panel). ``auto`` therefore
    fills remaining slots from train, preferring tiles not in MATCHFORMER_TRAIN_LIST.
    """
    val_list = os.environ.get("MATCHFORMER_VAL_LIST", "").strip()
    panel_fixed = (
        _read_name_list(val_list)
        if val_list
        else Her2JpegPairDataset(her2, "val").names[:VAL_PANEL_N]
    )
    panel_set = set(panel_fixed)
    train_list = os.environ.get("MATCHFORMER_TRAIN_LIST", "").strip()
    train_used = set(_read_name_list(train_list)) if train_list else set()

    all_val = Her2JpegPairDataset(her2, "val").names
    all_train = Her2JpegPairDataset(her2, "train").names

    def take_val() -> list[tuple[str, str]]:
        names = [nm for nm in all_val if (not skip_panel) or nm not in panel_set]
        return [("val", nm) for nm in names]

    def take_train(prefer_held_out: bool) -> list[tuple[str, str]]:
        if prefer_held_out and train_used:
            held = [nm for nm in all_train if nm not in train_used]
            if held:
                return [("train", nm) for nm in held]
        # Diverse stride through train so we do not just get the first WSI.
        if len(all_train) <= n + offset:
            pool = all_train
        else:
            step = max(1, len(all_train) // max(n + offset, 1))
            pool = all_train[::step]
            if len(pool) < n + offset:
                pool = all_train
        return [("train", nm) for nm in pool]

    split = (split or "auto").strip().lower()
    if split == "val":
        candidates = take_val()
    elif split == "train":
        candidates = take_train(prefer_held_out=True)
    else:
        # auto: all usable val first, then train fill
        candidates = take_val()
        if len(candidates) < offset + n:
            candidates = candidates + take_train(prefer_held_out=True)

    picked = candidates[offset : offset + n]
    if not picked:
        raise RuntimeError(
            f"no sample names left (val={len(all_val)} train={len(all_train)} "
            f"skip_panel={skip_panel} split={split} offset={offset} n={n})"
        )
    return picked


@torch.no_grad()
def sample_val(args):
    """Hard-argmax recon panels for several tiles from a checkpoint.

    Default ``auto`` split: residual val tiles (this volume has only 4), then
    fill from train for extra qualitative coverage beyond the fixed panel.
    """
    device = require_cuda(getattr(args, "device", "cuda"))
    her2 = Path(os.environ.get("MATCHFORMER_HER2", str(HER2_ROOT)))
    ckpt_path = Path(
        getattr(args, "ckpt", "")
        or os.environ.get("MATCHFORMER_INIT_CKPT", "")
    ).expanduser()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"ckpt missing: {ckpt_path}")

    out_dir = Path(
        getattr(args, "out_dir", "")
        or os.environ.get(
            "MATCHFORMER_SAMPLE_OUT",
            str(OUT_DIR / "sample_val"),
        )
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    n = int(getattr(args, "n", 8) or 8)
    offset = int(getattr(args, "offset", 0) or 0)
    skip_panel = not bool(getattr(args, "include_panel", False))
    split = str(getattr(args, "split", "auto") or "auto")

    picked = _pick_sample_names(
        her2, n=n, offset=offset, skip_panel=skip_panel, split=split
    )

    blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = ShallowMatchFormer64(use_checkpoint=False).to(device).eval()
    missing, unexpected = model.load_state_dict(blob["model"], strict=False)
    epoch_tag = int(blob.get("epoch", 0) or 0)
    print(
        f"sample_val ckpt={ckpt_path.name} epoch={epoch_tag} "
        f"n={len(picked)} skip_panel={skip_panel} split={split} out={out_dir} "
        f"missing={len(missing)} unexpected={len(unexpected)}",
        flush=True,
    )
    for sp, nm in picked:
        print(f"  [{sp}] {nm}", flush=True)

    he_a, ihc_a = load_donor()
    he_a = he_a.to(device)
    patches_a = patches_4x4(ihc_a.to(device))

    he_rows, true_rows, hard_rows = [], [], []
    metrics = []
    for sp, nm in picked:
        he, ihc, _name = Her2JpegPairDataset(her2, sp, names=[nm])[0]
        he = he.unsqueeze(0).to(device)
        ihc = ihc.unsqueeze(0).to(device)
        Fb, Fa = model(he, he_a)
        hard, _idx = hard_from_S(Fb, Fa, patches_a, tau=TAU)
        rgb = float(F.l1_loss(hard, ihc).item())
        dab = float(F.l1_loss(dab_hed_ch2(hard), dab_hed_ch2(ihc)).item())
        stem = Path(nm).stem
        metrics.append((sp, stem, rgb, dab))
        he_rows.append(he.cpu())
        true_rows.append(ihc.cpu())
        hard_rows.append(hard.cpu())
        save_recon_panel(
            out_dir / f"recon_{sp}_{stem}.png",
            he.cpu(),
            ihc.cpu(),
            hard.cpu(),
        )
        print(f"  [{sp}] {stem} hard_RGB={rgb:.4f} hard_DAB={dab:.4f}", flush=True)

    he_b = torch.cat(he_rows, dim=0)
    true_b = torch.cat(true_rows, dim=0)
    hard_b = torch.cat(hard_rows, dim=0)
    grid_path = out_dir / f"recon_grid_ep{epoch_tag:04d}_n{len(picked)}.png"
    save_recon_panel(grid_path, he_b, true_b, hard_b)

    mean_rgb = float(np.mean([m[2] for m in metrics]))
    mean_dab = float(np.mean([m[3] for m in metrics]))
    summary = out_dir / "sample_val_summary.txt"
    lines = [
        f"ckpt={ckpt_path}",
        f"epoch={epoch_tag}",
        f"n={len(picked)} skip_panel={skip_panel} split={split} offset={offset}",
        f"mean_hard_RGB={mean_rgb:.4f} mean_hard_DAB={mean_dab:.4f}",
        f"grid={grid_path.name}",
        "",
    ]
    for sp, stem, rgb, dab in metrics:
        lines.append(f"[{sp}] {stem}  hard_RGB={rgb:.4f}  hard_DAB={dab:.4f}")
    summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        f"DONE sample_val mean_RGB={mean_rgb:.4f} mean_DAB={mean_dab:.4f} "
        f"grid={grid_path.name}",
        flush=True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mode",
        choices=("precompute_global", "train", "sample_val"),
        default="train",
    )
    ap.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument(
        "--n-pos",
        type=int,
        default=None,
        help="Multi-positive InfoNCE set size K (default 4 / MATCHFORMER_N_POS)",
    )
    ap.add_argument(
        "--store-k",
        type=int,
        default=None,
        help="Top-K stored by precompute_global (must be >= --n-pos)",
    )
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--ckpt", type=str, default="")
    ap.add_argument("--n", type=int, default=8, help="number of tiles to sample")
    ap.add_argument("--offset", type=int, default=0, help="skip first N candidates")
    ap.add_argument(
        "--split",
        type=str,
        default="auto",
        choices=("auto", "val", "train"),
        help="auto fills train after the tiny val set",
    )
    ap.add_argument(
        "--include-panel",
        action="store_true",
        help="also include the fixed training val-panel names",
    )
    ap.add_argument("--out-dir", type=str, default="")
    args = ap.parse_args()
    if args.n_pos is not None:
        os.environ["MATCHFORMER_N_POS"] = str(int(args.n_pos))
    if args.store_k is not None:
        os.environ["MATCHFORMER_GLOBAL_POS_STORE_K"] = str(int(args.store_k))
    if args.mode == "precompute_global":
        precompute_global(args)
    elif args.mode == "sample_val":
        sample_val(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
