# Test A: one fixed H&E/IHC pair, no aug, frozen data.
# MatchFormer-style hierarchical self/cross + CoCosNet-v2 soft correspondence.
# Local only. Do not use for Dual ViT / TUM / Thunder jobs.
"""MatchFormer-pair correspondence overfit (Test A)."""

from __future__ import annotations

import argparse
import gc
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = ROOT / "_scratch"
OUT_DIR = Path(os.environ.get("MATCHFORMER_OUT", str(SCRATCH / "p25_ddm_v2_x0_id" / "matchformer_pair")))
CORR_DIR = OUT_DIR / "corr_scales"
CKPT_PATH = OUT_DIR / "matchformer_pair_step0400.pt"
MD_PATH = SCRATCH / "p25_ddm_v2_x0_id" / "matchformer_pair.md"

B_SLOT = 58  # gate-cells val slot
# A chosen by max mean Ruifrok HED channel-2 (DAB) over train IHC
A_INDEX = 4413
A_NAME = "wsi010_id3287_x61440_y97280.jpg"
B_NAME = "wsi013_id517_x35840_y50176.jpg"

TAU = 0.1
TOPK = 16  # soft-transfer Top-K for logging / inference panels (unchanged)
CORR_CAND_TOPK = 32  # legacy: C_q top H&E candidates when POS_MODE=topk
CORR_N_POS = int(os.environ.get("MATCHFORMER_N_POS", "4"))  # P_q size in InfoNCE
GLOBAL_POS_STORE_K = int(
    os.environ.get("MATCHFORMER_GLOBAL_POS_STORE_K", "16")
)  # precompute Top-K; train uses first CORR_N_POS
# Positives: "topk" = old S Top-32 then IHC mine; "global" = precomputed global IHC Top-K.
POS_MODE = os.environ.get("MATCHFORMER_POS_MODE", "topk").strip().lower()
GLOBAL_POS_DIR = Path(
    os.environ.get(
        "MATCHFORMER_GLOBAL_POS_DIR",
        str(SCRATCH / "p25_ddm_v2_x0_id" / "matchformer_global_pos"),
    )
)
# Donor encode: "shared" = full shared hierarchy on 512 mosaic (baseline / Exp A);
# "independent" = stem+local SELF per 32x32 medoid, scatter to mosaic (Exp B).
DONOR_MODE = os.environ.get("MATCHFORMER_DONOR_MODE", "shared").strip().lower()
DONOR_LOCAL_DEPTH = int(os.environ.get("MATCHFORMER_DONOR_LOCAL_DEPTH", "2"))
# LSGAN on the soft reconstruction. 0 disables it. Same weight as BBDM Exp 16.
LAMBDA_ADV = float(os.environ.get("MATCHFORMER_LAMBDA_ADV", "0.01"))
LAMBDA_RGB = 1.0  # lambda_R: RGB L1 weight in IHC error e_qp
LAMBDA_DAB = 1.0  # lambda_D: DAB L1 weight in IHC error e_qp
C_OUT = 128
TILE = 512
GRID = 64
PATCH = TILE // GRID  # 8 px; stem stride 8 keeps the 64x64 correspondence
MEDOID = 32  # Ward medoid side in donor mosaic
CELLS_PER_MEDOID = MEDOID // PATCH  # 4
N_MEDOIDS_SIDE = TILE // MEDOID  # 16
N_MEDOIDS = N_MEDOIDS_SIDE * N_MEDOIDS_SIDE  # 256
DONOR_PANEL = (
    SCRATCH / "p25_ddm_v2_x0_id" / "he_patch32_kmeans256_panel.png"
)
TILES512 = Path(os.environ.get("MATCHFORMER_TILES", str(SCRATCH / "p25_ddm_v2_x0_id" / "tiles512")))
NUM_STEPS = 400
PRINT_EVERY = 20
PNG_EVERY = 100
LR = float(os.environ.get("MATCHFORMER_LR", "1e-4"))
GRAD_CLIP = 1.0
NUM_EPOCHS_FULL = 30
BATCH_SIZE_FULL = 16
VAL_PANEL_N = 4
HER2_ROOT = Path(
    os.environ.get("MATCHFORMER_HER2", "/home/duaa/c9_knockouts/her2match_full")
)

# Same 4 cells on the 64-grid for every scale (mapped by integer downsample).
# Labels: brown gland, blue nucleus, pale stroma, tissue edge on B.
QUERY_CELLS_64 = [
    (2, 59, "brown_gland"),
    (57, 6, "blue_nucleus"),
    (34, 19, "pale_stroma"),
    (26, 13, "tissue_edge"),
]

# Ruifrok HED (rows = H, E, DAB OD vectors). Channel 2 = DAB.
_HED_FROM_RGB = torch.tensor(
    [
        [0.650, 0.704, 0.286],
        [0.072, 0.990, 0.105],
        [0.268, 0.570, 0.776],
    ],
    dtype=torch.float32,
)


def dab_hed_ch2(rgb01: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """rgb01: (B,3,H,W) in [0,1] -> DAB concentration (B,H,W), HED channel 2."""
    # dens = OD @ inv(HED); HED rows are stain vectors in OD space
    M_inv = torch.linalg.inv(_HED_FROM_RGB.to(device=rgb01.device, dtype=rgb01.dtype))
    od = -torch.log(rgb01.clamp(min=0) + eps)  # B,3,H,W
    # einsum: for each pixel, dens = M_inv @ od  if dens = M^{-1} od with M rows stains
    # skimage: stains = od.T @ rgb_from_hed -> dens_c = sum_s od_s * rgb_from_hed[s,c]
    # rgb_from_hed = inv(hed_from_rgb)
    dens = torch.einsum("bchw,cs->bshw", od, M_inv)
    return dens[:, 2].clamp(min=0)


# ---------------------------------------------------------------------------
# Blocks
# ---------------------------------------------------------------------------


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        hid = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hid)
        self.fc2 = nn.Linear(hid, dim)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class SelfBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.mlp = MLP(dim)
        self.scale = (dim // num_heads) ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: B,N,C
        B, N, C = x.shape
        h = self.heads
        dh = C // h
        y = self.norm1(x)
        qkv = self.qkv(y).reshape(B, N, 3, h, dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        # memory-safe SDPA (no full attn materialization when backend allows)
        attn_out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        attn_out = attn_out.transpose(1, 2).reshape(B, N, C)
        x = x + self.proj(attn_out)
        x = x + self.mlp(self.norm2(x))
        return x


class CrossBlock(nn.Module):
    """Bidirectional cross-attention: B->A and A->B in one block."""

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.heads = num_heads
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim)
        self.norm_mlp = nn.LayerNorm(dim)
        self.mlp = MLP(dim)

    def _cross(self, q_tok: torch.Tensor, kv_tok: torch.Tensor) -> torch.Tensor:
        B, Nq, C = q_tok.shape
        Nk = kv_tok.shape[1]
        h = self.heads
        dh = C // h
        q = self.q_proj(self.norm_q(q_tok)).reshape(B, Nq, h, dh).transpose(1, 2)
        k = self.k_proj(self.norm_kv(kv_tok)).reshape(B, Nk, h, dh).transpose(1, 2)
        v = self.v_proj(self.norm_kv(kv_tok)).reshape(B, Nk, h, dh).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        out = out.transpose(1, 2).reshape(B, Nq, C)
        return self.out_proj(out)

    def forward(self, feat_b: torch.Tensor, feat_a: torch.Tensor):
        # feat_*: B,N,C
        feat_b = feat_b + self._cross(feat_b, feat_a)
        feat_a = feat_a + self._cross(feat_a, feat_b)
        feat_b = feat_b + self.mlp(self.norm_mlp(feat_b))
        feat_a = feat_a + self.mlp(self.norm_mlp(feat_a))
        return feat_b, feat_a


def _to_tokens(x: torch.Tensor) -> torch.Tensor:
    # B,C,H,W -> B,HW,C
    B, C, H, W = x.shape
    return x.flatten(2).transpose(1, 2).contiguous()


def _from_tokens(t: torch.Tensor, H: int, W: int) -> torch.Tensor:
    B, N, C = t.shape
    return t.transpose(1, 2).reshape(B, C, H, W).contiguous()


class Stage(nn.Module):
    def __init__(self, dim: int, num_heads: int, pattern: str, down_in: int | None = None):
        """pattern: sequence of 'S' and 'C', e.g. 'SSC' or 'SCC'. Query-only: all 'S'."""
        super().__init__()
        self.down = None
        if down_in is not None:
            self.down = nn.Sequential(
                nn.Conv2d(down_in, dim, 3, stride=2, padding=1),
                nn.GELU(),
            )
        self.blocks = nn.ModuleList()
        self.kinds = list(pattern)
        self.has_cross = "C" in self.kinds
        for k in self.kinds:
            if k == "S":
                self.blocks.append(SelfBlock(dim, num_heads))
            elif k == "C":
                self.blocks.append(CrossBlock(dim, num_heads))
            else:
                raise ValueError(k)
        self.dim = dim

    def forward(self, fb: torch.Tensor, fa: torch.Tensor | None = None):
        if self.down is not None:
            fb = self.down(fb)
            if fa is not None:
                fa = self.down(fa)
        H, W = fb.shape[-2:]
        tb = _to_tokens(fb)
        ta = _to_tokens(fa) if fa is not None else None
        for kind, blk in zip(self.kinds, self.blocks):
            if kind == "S":
                tb = blk(tb)
                if ta is not None:
                    ta = blk(ta)
            else:
                if ta is None:
                    raise RuntimeError("CROSS block requires donor features fa")
                tb, ta = blk(tb, ta)
        out_b = _from_tokens(tb, H, W)
        if ta is None:
            return out_b
        return out_b, _from_tokens(ta, H, W)


def mosaic_to_medoids(he: torch.Tensor) -> torch.Tensor:
    """B,3,512,512 mosaic -> (B*256),3,32,32 medoids (row-major mr, mc)."""
    B, C, H, W = he.shape
    if H != TILE or W != TILE or C != 3:
        raise ValueError(f"expected B,3,{TILE},{TILE} got {tuple(he.shape)}")
    x = he.reshape(B, C, N_MEDOIDS_SIDE, MEDOID, N_MEDOIDS_SIDE, MEDOID)
    x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
    return x.reshape(B * N_MEDOIDS, C, MEDOID, MEDOID)


def medoid_feats_to_mosaic(feats: torch.Tensor, batch: int) -> torch.Tensor:
    """(B*256),128,4,4 -> B,128,64,64 in patches_4x4 mosaic row-major order."""
    c = feats.shape[1]
    x = feats.reshape(
        batch, N_MEDOIDS_SIDE, N_MEDOIDS_SIDE, c, CELLS_PER_MEDOID, CELLS_PER_MEDOID
    )
    # out[b,c, mr*4+lr, mc*4+lc] = x[b,mr,mc,c,lr,lc]
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
    return x.reshape(batch, c, GRID, GRID)


def assert_medoid_scatter_roundtrip() -> None:
    """Unique per-cell codes survive pack→scatter; mosaic_to_medoids places (17,42)."""
    assert_medoid_index_alignment()
    mosaic = torch.zeros(1, C_OUT, GRID, GRID)
    for r in range(GRID):
        for c in range(GRID):
            mosaic[0, 0, r, c] = float(medoid_global_index(r, c))
    packed = mosaic.reshape(
        1, C_OUT, N_MEDOIDS_SIDE, CELLS_PER_MEDOID, N_MEDOIDS_SIDE, CELLS_PER_MEDOID
    )
    packed = packed.permute(0, 2, 4, 1, 3, 5).contiguous()
    packed = packed.reshape(N_MEDOIDS, C_OUT, CELLS_PER_MEDOID, CELLS_PER_MEDOID)
    back = medoid_feats_to_mosaic(packed, 1)
    if not torch.allclose(back, mosaic):
        raise AssertionError("medoid scatter roundtrip failed")
    he = torch.zeros(1, 3, TILE, TILE)
    he[0, 0, 17 * PATCH : 17 * PATCH + PATCH, 42 * PATCH : 42 * PATCH + PATCH] = 1.0
    med = mosaic_to_medoids(he)
    mr, mc, lr, lc = medoid_local_coords(17, 42)
    mid = mr * N_MEDOIDS_SIDE + mc
    y0, x0 = lr * PATCH, lc * PATCH
    if float(med[mid, 0, y0 : y0 + PATCH, x0 : x0 + PATCH].mean()) < 0.99:
        raise AssertionError("mosaic_to_medoids misplaced cell (17,42)")


class FPN(nn.Module):
    def __init__(self, dims: tuple[int, ...], out_ch: int = 128):
        super().__init__()
        # dims: stage1..4 channel counts at 64,32,16,8
        self.laterals = nn.ModuleList([nn.Conv2d(d, out_ch, 1) for d in dims])
        self.smooth = nn.ModuleList(
            [nn.Conv2d(out_ch, out_ch, 3, padding=1) for _ in dims[:-1]]
        )
        self.out = nn.Conv2d(out_ch, out_ch, 3, padding=1)

    def forward(self, feats: list[torch.Tensor]) -> torch.Tensor:
        # feats[0]=64, [1]=32, [2]=16, [3]=8
        laterals = [lat(f) for lat, f in zip(self.laterals, feats)]
        x = laterals[-1]
        for i in range(len(laterals) - 2, -1, -1):
            x = F.interpolate(x, size=laterals[i].shape[-2:], mode="nearest")
            x = self.smooth[i](x + laterals[i])
        return self.out(x)


class MatchFormerPair(nn.Module):
    def __init__(
        self,
        use_checkpoint: bool = True,
        donor_mode: str | None = None,
        donor_local_depth: int | None = None,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.donor_mode = (donor_mode or DONOR_MODE).strip().lower()
        if self.donor_mode not in ("shared", "independent"):
            raise ValueError(f"donor_mode must be shared|independent, got {self.donor_mode}")
        depth = DONOR_LOCAL_DEPTH if donor_local_depth is None else int(donor_local_depth)
        self.stem = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=7, stride=8, padding=3, bias=False),
            nn.GroupNorm(8, 128),
            nn.GELU(),
        )
        if self.donor_mode == "shared":
            # Shared hierarchy + CROSS on the artificial mosaic (baseline / Exp A).
            self.stage1 = Stage(128, num_heads=4, pattern="SSC")
            self.stage2 = Stage(192, num_heads=6, pattern="SSC", down_in=128)
            self.stage3 = Stage(256, num_heads=8, pattern="SCC", down_in=192)
            self.stage4 = Stage(512, num_heads=8, pattern="SCC", down_in=256)
            self.donor_local = None
        else:
            # Exp B: query hierarchy SELF-only (no query↔donor CROSS); donor = local medoids.
            self.stage1 = Stage(128, num_heads=4, pattern="SSS")
            self.stage2 = Stage(192, num_heads=6, pattern="SSS", down_in=128)
            self.stage3 = Stage(256, num_heads=8, pattern="SSS", down_in=192)
            self.stage4 = Stage(512, num_heads=8, pattern="SSS", down_in=256)
            self.donor_local = nn.ModuleList(
                [SelfBlock(128, num_heads=4) for _ in range(depth)]
            )
        self.fpn = FPN((128, 192, 256, 512), out_ch=C_OUT)

    def _run_stage_pair(self, stage: Stage, fb: torch.Tensor, fa: torch.Tensor):
        if self.use_checkpoint and self.training:
            return checkpoint(stage, fb, fa, use_reentrant=False)
        return stage(fb, fa)

    def _run_stage_query(self, stage: Stage, fb: torch.Tensor):
        if self.use_checkpoint and self.training:
            return checkpoint(stage, fb, use_reentrant=False)
        return stage(fb)

    def _encode_donor_independent(self, he_a: torch.Tensor) -> torch.Tensor:
        """he_a: Ba,3,512,512 -> Fa: Ba,128,64,64 (mosaic-aligned)."""
        Ba = he_a.shape[0]
        med = mosaic_to_medoids(he_a)  # Ba*256,3,32,32
        x = self.stem(med)  # Ba*256,128,4,4
        t = x.flatten(2).transpose(1, 2).contiguous()  # Ba*256,16,128
        for blk in self.donor_local:
            if self.use_checkpoint and self.training:
                t = checkpoint(blk, t, use_reentrant=False)
            else:
                t = blk(t)
        x = t.transpose(1, 2).reshape(Ba * N_MEDOIDS, C_OUT, CELLS_PER_MEDOID, CELLS_PER_MEDOID)
        return medoid_feats_to_mosaic(x, Ba)

    def encode(self, he_b: torch.Tensor, he_a: torch.Tensor, return_stages: bool = False):
        # he_*: B,3,512,512 in [0,1]. Stem stride 8 -> 64x64.
        if self.donor_mode == "shared":
            fb = self.stem(he_b)
            fa = self.stem(he_a)
            f1b, f1a = self._run_stage_pair(self.stage1, fb, fa)
            f2b, f2a = self._run_stage_pair(self.stage2, f1b, f1a)
            f3b, f3a = self._run_stage_pair(self.stage3, f2b, f2a)
            f4b, f4a = self._run_stage_pair(self.stage4, f3b, f3a)
            Fb = self.fpn([f1b, f2b, f3b, f4b])
            Fa = self.fpn([f1a, f2a, f3a, f4a])
            if not return_stages:
                return Fb, Fa
            stages = {
                "stage1": (f1b, f1a),
                "stage2": (f2b, f2a),
                "stage3": (f3b, f3a),
                "stage4": (f4b, f4a),
                "FPN": (Fb, Fa),
            }
            return Fb, Fa, stages

        # Independent donor (Exp B): no encoder-side CROSS.
        B = he_b.shape[0]
        Ba = he_a.shape[0]
        fb = self.stem(he_b)
        f1b = self._run_stage_query(self.stage1, fb)
        f2b = self._run_stage_query(self.stage2, f1b)
        f3b = self._run_stage_query(self.stage3, f2b)
        f4b = self._run_stage_query(self.stage4, f3b)
        Fb = self.fpn([f1b, f2b, f3b, f4b])
        Fa = self._encode_donor_independent(he_a)
        if Ba == 1 and B > 1:
            Fa = Fa.expand(B, -1, -1, -1).contiguous()
        elif Ba != B:
            raise ValueError(f"he_a batch {Ba} incompatible with he_b batch {B}")
        if not return_stages:
            return Fb, Fa
        stages = {
            "stage1": (f1b, None),
            "stage2": (f2b, None),
            "stage3": (f3b, None),
            "stage4": (f4b, None),
            "FPN": (Fb, Fa),
        }
        return Fb, Fa, stages

    def forward(self, he_b: torch.Tensor, he_a: torch.Tensor, return_stages: bool = False):
        # Alias for DataParallel / DDP (they wrap forward, not encode).
        return self.encode(he_b, he_a, return_stages=return_stages)


# ---------------------------------------------------------------------------
# Correspondence + transfer
# ---------------------------------------------------------------------------


def correspondence(Fb: torch.Tensor, Fa: torch.Tensor, tau: float = TAU, chunk: int = 256):
    """Fb,Fa: B,C,64,64 -> W: B,4096,4096 (row-stochastic over A).

    Chunked matmul/softmax with periodic sync to avoid Windows TDR on 4096^2.
    """
    fb = F.normalize(Fb.flatten(2).transpose(1, 2).float(), dim=-1)  # B,N,C
    fa = F.normalize(Fa.flatten(2).transpose(1, 2).float(), dim=-1)
    B, N, _C = fb.shape
    S = fb.new_empty(B, N, N)
    for i in range(0, N, chunk):
        i1 = min(i + chunk, N)
        S[:, i:i1] = torch.bmm(fb[:, i:i1], fa.transpose(1, 2)) / tau
        if fb.is_cuda and (i // chunk) % 4 == 0:
            torch.cuda.synchronize()
    del fb, fa
    R = S.new_empty(B, N, N)
    for i in range(0, N, chunk):
        i1 = min(i + chunk, N)
        R[:, i:i1] = torch.softmax(S[:, i:i1], dim=-1)
    Cmat = S.new_empty(B, N, N)
    for j in range(0, N, chunk):
        j1 = min(j + chunk, N)
        Cmat[:, :, j:j1] = torch.softmax(S[:, :, j:j1], dim=-2)
        if S.is_cuda and (j // chunk) % 4 == 0:
            torch.cuda.synchronize()
    del S
    M = R * Cmat
    del R, Cmat
    W = M / M.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    del M
    return W


def patches_4x4(img: torch.Tensor) -> torch.Tensor:
    """img B,3,512,512 -> B,4096,192 (row-major 64x64 of 8x8 patches)."""
    p = F.unfold(img, kernel_size=PATCH, stride=PATCH)
    return p.transpose(1, 2).contiguous()


def stitch_4x4(patches: torch.Tensor) -> torch.Tensor:
    """patches B,4096,192 -> B,3,512,512."""
    p = patches.transpose(1, 2)
    return F.fold(p, output_size=(TILE, TILE), kernel_size=PATCH, stride=PATCH)


def soft_transfer(W: torch.Tensor, patches_a: torch.Tensor, topk: int = TOPK):
    """W: B,Q,P; patches_a: B,P,192 -> soft image B,3,512,512 and topk W.

    Gradients flow only through the selected Top-K weights (idx has no grad).
    """
    B, Q, _P = W.shape
    vals, idx = torch.topk(W, k=topk, dim=-1)  # B,Q,K
    vals = vals / vals.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    K = topk
    # Stop grad into non-topk mass of W by rebuilding a sparse-ish gather from vals only.
    flat_idx = idx.reshape(B, Q * K)
    gathered = torch.gather(
        patches_a, 1, flat_idx.unsqueeze(-1).expand(-1, -1, patches_a.shape[-1])
    ).reshape(B, Q, K, patches_a.shape[-1])
    soft_p = (vals.unsqueeze(-1) * gathered).sum(dim=2)
    return stitch_4x4(soft_p), vals, idx


class PatchGANDiscriminator(nn.Module):
    """70x70 PatchGAN. Input is concat(HE, IHC or soft recon), 6 channels.

    Same stack as bbdm_BCI/model.py PatchGANDiscriminator. Images are mapped
    from [0, 1] to [-1, 1] before this module.
    """

    def __init__(self, in_channels=6, base_channels=64, n_layers=3):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, base_channels, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        ch = base_channels
        for _i in range(1, n_layers):
            out_ch = min(ch * 2, 512)
            layers += [
                nn.Conv2d(ch, out_ch, 4, stride=2, padding=1),
                nn.InstanceNorm2d(out_ch),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            ch = out_ch
        out_ch = min(ch * 2, 512)
        layers += [
            nn.Conv2d(ch, out_ch, 4, stride=1, padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, 1, 4, stride=1, padding=1),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, source, target):
        return self.model(torch.cat([source, target], dim=1))


def _lsgan_on_soft(disc, disc_opt, he, ihc, soft, mb, B, dtype_ctx):
    """One LSGAN step on the soft IHC. Returns (g_adv, d_loss) as floats.

    Discriminator sees detached fakes. Generator step freezes disc weights
    so the adversarial gradient updates the matcher only.
    """
    he_m = he * 2 - 1
    real_m = ihc * 2 - 1
    fake_m = soft * 2 - 1
    disc_opt.zero_grad(set_to_none=True)
    with dtype_ctx:
        d_real = disc(he_m, real_m)
        d_fake = disc(he_m, fake_m.detach())
        d_loss = 0.5 * (
            F.mse_loss(d_real, torch.ones_like(d_real))
            + F.mse_loss(d_fake, torch.zeros_like(d_fake))
        )
    d_loss.backward()
    disc_opt.step()
    for p in disc.parameters():
        p.requires_grad_(False)
    with dtype_ctx:
        d_fake_g = disc(he_m, fake_m)
        g_adv = F.mse_loss(d_fake_g, torch.ones_like(d_fake_g))
    (LAMBDA_ADV * g_adv * (mb / float(B))).backward()
    for p in disc.parameters():
        p.requires_grad_(True)
    return float(g_adv.detach()), float(d_loss.detach())


@torch.no_grad()
def hard_transfer(W: torch.Tensor, patches_a: torch.Tensor):
    idx = W.argmax(dim=-1)  # B,Q
    B, Q = idx.shape
    expand_idx = idx.unsqueeze(-1).expand(-1, -1, patches_a.shape[-1])
    hard_p = torch.gather(patches_a, 1, expand_idx)
    return stitch_4x4(hard_p)


def _patches_dab_flat(patches: torch.Tensor) -> torch.Tensor:
    """patches B,N,192 (unfold 8x8 RGB) -> DAB B,N,64 (flattened 8x8)."""
    B, N, _ = patches.shape
    rgb = patches.reshape(B * N, 3, PATCH, PATCH)
    dab = dab_hed_ch2(rgb)
    return dab.reshape(B, N, PATCH * PATCH)


def medoid_global_index(cell_r: int, cell_c: int) -> int:
    """Map 64-grid (r,c) to flat donor index; matches patches_4x4 row-major."""
    return int(cell_r) * GRID + int(cell_c)


def medoid_local_coords(cell_r: int, cell_c: int) -> tuple[int, int, int, int]:
    """Return (medoid_r, medoid_c, local_r, local_c) for a 64-grid cell."""
    return (
        int(cell_r) // CELLS_PER_MEDOID,
        int(cell_c) // CELLS_PER_MEDOID,
        int(cell_r) % CELLS_PER_MEDOID,
        int(cell_c) % CELLS_PER_MEDOID,
    )


def medoid_bank_to_mosaic_index(medoid_r: int, medoid_c: int, local_r: int, local_c: int) -> int:
    """Scatter a cell from an independent-medoid encoder back to mosaic row-major index.

    Independent medoid encode yields [256,4,4,C]. Naive reshape to 4096 is
    medoid-major and does NOT match patches_4x4. Always scatter:
      r = mr*4+lr, c = mc*4+lc, index = r*64+c
    before comparing to donor IHC patches.
    """
    r = int(medoid_r) * CELLS_PER_MEDOID + int(local_r)
    c = int(medoid_c) * CELLS_PER_MEDOID + int(local_c)
    return medoid_global_index(r, c)


def assert_medoid_index_alignment(n_checks: int = 64) -> None:
    """Unit check: mosaic row-major <-> medoid (mr,mc,lr,lc) scatter is bijective."""
    rng = np.random.default_rng(0)
    for _ in range(n_checks):
        r = int(rng.integers(0, GRID))
        c = int(rng.integers(0, GRID))
        g = medoid_global_index(r, c)
        mr, mc, lr, lc = medoid_local_coords(r, c)
        f = medoid_bank_to_mosaic_index(mr, mc, lr, lc)
        if g != f:
            raise AssertionError(
                f"index mismatch cell=({r},{c}) mosaic={g} scatter={f} "
                f"medoid=({mr},{mc}) local=({lr},{lc})"
            )
    r, c = 17, 42
    assert medoid_global_index(r, c) == 17 * 64 + 42
    mr, mc, lr, lc = medoid_local_coords(r, c)
    assert (mr, mc, lr, lc) == (4, 10, 1, 2)
    assert medoid_bank_to_mosaic_index(mr, mc, lr, lc) == medoid_global_index(r, c)


def compute_global_ihc_topk(
    patches_q: torch.Tensor,
    patches_d: torch.Tensor,
    k: int = GLOBAL_POS_STORE_K,
    q_chunk: int = 128,
    lambda_rgb: float = LAMBDA_RGB,
    lambda_dab: float = LAMBDA_DAB,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Global IHC Top-k donors per query cell (no H&E shortlist).

    patches_q / patches_d: [N,192] or [1,N,192].
    Returns idx [N,k] long, err [N,k] float (lowest error first).
    """
    pq = patches_q[0] if patches_q.dim() == 3 else patches_q
    pd = patches_d[0] if patches_d.dim() == 3 else patches_d
    N = pq.shape[0]
    if pd.shape[0] != N:
        raise ValueError(f"donor patches N={pd.shape[0]} != query N={N}")
    dab_q = _patches_dab_flat(pq.unsqueeze(0))[0]
    dab_d = _patches_dab_flat(pd.unsqueeze(0))[0]
    best_idx = []
    best_err = []
    for q0 in range(0, N, q_chunk):
        q1 = min(q0 + q_chunk, N)
        e_rgb = (pq[q0:q1].unsqueeze(1) - pd.unsqueeze(0)).abs().mean(dim=-1)
        e_dab = (dab_q[q0:q1].unsqueeze(1) - dab_d.unsqueeze(0)).abs().mean(dim=-1)
        err = lambda_dab * e_dab + lambda_rgb * e_rgb
        vals, indices = torch.topk(err, k=k, dim=1, largest=False)
        best_idx.append(indices)
        best_err.append(vals)
    return torch.cat(best_idx, 0), torch.cat(best_err, 0)


def save_global_pos_tile(path: Path, idx: torch.Tensor, err: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "idx": idx.detach().cpu().to(torch.int32),
            "err": err.detach().cpu().to(torch.float16),
            "k": int(idx.shape[-1]),
            "lambda_rgb": LAMBDA_RGB,
            "lambda_dab": LAMBDA_DAB,
        },
        path,
    )


def load_global_pos_tile(path: Path, n_pos: int = CORR_N_POS) -> torch.Tensor:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    idx = blob["idx"]
    if idx.shape[-1] < n_pos:
        raise ValueError(f"{path} has k={idx.shape[-1]} < n_pos={n_pos}")
    return idx[:, :n_pos].long()


def load_global_pos_cache(pos_dir: Path, names: list[str], n_pos: int = CORR_N_POS) -> dict[str, torch.Tensor]:
    """Load [4096,n_pos] long tensors keyed by JPEG name (with .jpg)."""
    cache: dict[str, torch.Tensor] = {}
    missing = []
    for name in names:
        stem = Path(name).stem
        p = pos_dir / f"{stem}.pt"
        if not p.is_file():
            missing.append(name)
            continue
        cache[name] = load_global_pos_tile(p, n_pos=n_pos)
    if missing:
        preview = ", ".join(missing[:5])
        raise FileNotFoundError(
            f"missing {len(missing)} global-pos files under {pos_dir} "
            f"(e.g. {preview}). Run --mode precompute_global first."
        )
    return cache


def gather_pos_idx_batch(
    cache: dict[str, torch.Tensor], names: list[str], device: torch.device
) -> torch.Tensor:
    return torch.stack([cache[n] for n in names], 0).to(device, non_blocking=True)


@torch.no_grad()
def oracle_ihc_target_recall(
    Fb: torch.Tensor,
    Fa: torch.Tensor,
    pos_idx: torch.Tensor,
    tau: float = TAU,
    ks: tuple[int, ...] = (1, 4, 16, 32),
    n_true: int = CORR_N_POS,
    chunk: int = 256,
) -> dict[str, float]:
    """Fraction of query cells whose predicted Top-k contains ≥1 global Top-n_true IHC donor.

    Fb/Fa: B,C,H,W. pos_idx: B,N,Kstore (uses first n_true).
    """
    fb = F.normalize(Fb.flatten(2).transpose(1, 2).float(), dim=-1)
    fa = F.normalize(Fa.flatten(2).transpose(1, 2).float(), dim=-1)
    B, N, _ = fb.shape
    true = pos_idx[..., :n_true].to(device=fb.device)  # B,N,n_true
    max_k = max(ks)
    # scores only need topk indices; compute S in query chunks
    hit = {k: 0.0 for k in ks}
    n_cells = float(B * N)
    for i in range(0, N, chunk):
        i1 = min(i + chunk, N)
        S = torch.bmm(fb[:, i:i1], fa.transpose(1, 2)) / tau  # B,Qc,N
        _v, pred = torch.topk(S, k=max_k, dim=-1)  # B,Qc,max_k
        del _v, S
        t = true[:, i:i1]  # B,Qc,n_true
        for k in ks:
            pk = pred[..., :k]
            # any true in pred: for each cell, compare
            # (B,Qc,n_true,1) == (B,Qc,1,k) -> any over n_true and k
            match = (t.unsqueeze(-1) == pk.unsqueeze(-2)).any(dim=-1).any(dim=-1)
            hit[k] += float(match.float().sum().item())
        del pred, t
    return {f"ihc_recall@{k}": hit[k] / n_cells for k in ks}


def contrastive_correspondence_loss(
    Fb: torch.Tensor,
    Fa: torch.Tensor,
    ihc_q: torch.Tensor | None = None,
    patches_donor: torch.Tensor | None = None,
    tau: float = TAU,
    chunk: int = 256,
    cand_topk: int = CORR_CAND_TOPK,
    n_pos: int = CORR_N_POS,
    lambda_rgb: float = LAMBDA_RGB,
    lambda_dab: float = LAMBDA_DAB,
    pos_idx: torch.Tensor | None = None,
) -> torch.Tensor:
    """Multi-positive InfoNCE on cosine/tau scores S (before W).

    If ``pos_idx`` is given (B,N,n_pos), positives are those indices (Experiment A:
    stationary global IHC labels). Otherwise legacy Top-32 H&E shortlist then IHC mine.

    L(q) = logsumexp_p S_qp - logsumexp_{p in P_q} S_qp.
    """
    fb = F.normalize(Fb.flatten(2).transpose(1, 2).float(), dim=-1)  # B,N,C
    fa = F.normalize(Fa.flatten(2).transpose(1, 2).float(), dim=-1)
    B, N, _C = fb.shape
    S = fb.new_empty(B, N, N)
    for i in range(0, N, chunk):
        i1 = min(i + chunk, N)
        S[:, i:i1] = torch.bmm(fb[:, i:i1], fa.transpose(1, 2)) / tau
        if fb.is_cuda and (i // chunk) % 4 == 0:
            torch.cuda.synchronize()
    del fb, fa

    log_den = torch.logsumexp(S, dim=-1)  # B,N

    if pos_idx is None:
        if ihc_q is None or patches_donor is None:
            raise ValueError("legacy topk mining needs ihc_q and patches_donor")
        with torch.no_grad():
            _vals, cand_idx = torch.topk(S, k=cand_topk, dim=-1)  # B,N,K
            del _vals
            patches_q = patches_4x4(ihc_q.float())  # B,N,192
            if patches_donor.shape[0] == 1 and B > 1:
                patches_d = patches_donor.expand(B, -1, -1)
            else:
                patches_d = patches_donor
            dab_q = _patches_dab_flat(patches_q)
            dab_d = _patches_dab_flat(patches_d)

            K = cand_topk
            flat = cand_idx.reshape(B, N * K)
            don_rgb = torch.gather(
                patches_d, 1, flat.unsqueeze(-1).expand(-1, -1, patches_d.shape[-1])
            ).reshape(B, N, K, patches_d.shape[-1])
            e_rgb = (patches_q.unsqueeze(2) - don_rgb).abs().mean(dim=-1)
            del don_rgb
            don_dab = torch.gather(
                dab_d, 1, flat.unsqueeze(-1).expand(-1, -1, dab_d.shape[-1])
            ).reshape(B, N, K, dab_d.shape[-1])
            e_dab = (dab_q.unsqueeze(2) - don_dab).abs().mean(dim=-1)
            del don_dab, dab_q, dab_d, patches_q, patches_d, flat
            e_qp = lambda_dab * e_dab + lambda_rgb * e_rgb
            del e_dab, e_rgb
            _e_pos, pos_local = torch.topk(e_qp, k=n_pos, dim=-1, largest=False)
            del _e_pos, e_qp
            pos_idx = torch.gather(cand_idx, -1, pos_local)  # B,N,n_pos
            del cand_idx, pos_local
    else:
        if pos_idx.shape[0] != B or pos_idx.shape[1] != N:
            raise ValueError(
                f"pos_idx shape {tuple(pos_idx.shape)} != (B={B}, N={N}, n_pos)"
            )
        pos_idx = pos_idx[..., :n_pos].long()

    pos_scores = torch.gather(S, -1, pos_idx)  # B,N,n_pos (grad through S)
    log_num = torch.logsumexp(pos_scores, dim=-1)
    L = (log_den - log_num).mean()
    del S, log_den, log_num, pos_scores, pos_idx
    return L


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def donor_mosaic_u8(panel_path: Path = DONOR_PANEL) -> np.ndarray:
    """Pack the 256 panel cells into one 512x512 H&E tile.

    The PNG is a 16x16 grid of 32px patches drawn at scale 4 with a 2px gap.
    Each cell is nearest-downsampled back to 32 and placed with no gap.
    """
    from PIL import Image

    im = np.asarray(Image.open(panel_path).convert("RGB"))
    cell, gap, cols, patch = 128, 2, 16, 32
    canvas = np.empty((cols * patch, cols * patch, 3), dtype=np.uint8)
    for i in range(cols * cols):
        r, c = divmod(i, cols)
        y0 = gap + r * (cell + gap)
        x0 = gap + c * (cell + gap)
        crop = im[y0 : y0 + cell, x0 : x0 + cell]
        small = np.asarray(
            Image.fromarray(crop).resize((patch, patch), Image.Resampling.NEAREST)
        )
        canvas[r * patch : (r + 1) * patch, c * patch : (c + 1) * patch] = small
    return canvas


def save_donor_512(panel_path: Path = DONOR_PANEL) -> Path:
    """Write the fixed donor as a 512x512 uint8 tile."""
    TILES512.mkdir(parents=True, exist_ok=True)
    mosaic = donor_mosaic_u8(panel_path)
    out = TILES512 / "donor_he_kmeans256_512.npy"
    np.save(out, mosaic)
    from PIL import Image

    Image.fromarray(mosaic).save(TILES512 / "donor_he_kmeans256_512.png")
    return out


def _stack_u8(arr: np.ndarray) -> torch.Tensor:
    t = torch.from_numpy(np.ascontiguousarray(arr)).float() / 255.0
    if t.ndim == 3:
        t = t.unsqueeze(0)
    return t.permute(0, 3, 1, 2)


def _query_npy_pair() -> tuple[Path, Path] | None:
    """Prefer query16, then query8. MATCHFORMER_QUERY=8|16 forces one."""
    force = os.environ.get("MATCHFORMER_QUERY", "").strip()
    q16 = (TILES512 / "he_query16_512.npy", TILES512 / "ihc_query16_512.npy")
    q8 = (TILES512 / "he_query8_512.npy", TILES512 / "ihc_query8_512.npy")
    if force == "16":
        return q16 if q16[0].is_file() and q16[1].is_file() else None
    if force == "8":
        return q8 if q8[0].is_file() and q8[1].is_file() else None
    if q16[0].is_file() and q16[1].is_file():
        return q16
    if q8[0].is_file() and q8[1].is_file():
        return q8
    return None


def load_pair():
    """Queries are 512 tiles. Donor H&E and IHC are the shuffled codebook mosaics."""
    pair = _query_npy_pair()
    if pair is not None:
        he_b = np.load(pair[0])
        ihc_b = np.load(pair[1])
    else:
        he_b = np.load(TILES512 / "he_val_512.npy", mmap_mode="r")[B_SLOT]
        ihc_b = np.load(TILES512 / "ihc_val_512.npy", mmap_mode="r")[B_SLOT]
    donor_path = TILES512 / "donor_he_shuf_512.npy"
    ihc_path = TILES512 / "donor_ihc_shuf_512.npy"
    if not donor_path.is_file() or not ihc_path.is_file():
        raise SystemExit(f"Missing shuffled donor mosaics in {TILES512}")
    he_a = np.load(donor_path)
    ihc_a = np.load(ihc_path)
    return _stack_u8(he_b), _stack_u8(ihc_b), _stack_u8(he_a), _stack_u8(ihc_a)


def load_donor():
    """Prefer Ward-grouped mosaics; else shuffled codebook mosaics."""
    grouped_he = TILES512 / "donor_he_grouped_512.npy"
    grouped_ihc = TILES512 / "donor_ihc_grouped_512.npy"
    if grouped_he.is_file() and grouped_ihc.is_file():
        return _stack_u8(np.load(grouped_he)), _stack_u8(np.load(grouped_ihc))
    donor_path = TILES512 / "donor_he_shuf_512.npy"
    ihc_path = TILES512 / "donor_ihc_shuf_512.npy"
    if not donor_path.is_file() or not ihc_path.is_file():
        raise SystemExit(
            f"Missing donor mosaics in {TILES512} "
            f"(need grouped or shuffled donor_he/ihc_*_512.npy)"
        )
    return _stack_u8(np.load(donor_path)), _stack_u8(np.load(ihc_path))


def _donor_paths() -> tuple[Path, Path]:
    grouped_he = TILES512 / "donor_he_grouped_512.npy"
    grouped_ihc = TILES512 / "donor_ihc_grouped_512.npy"
    if grouped_he.is_file() and grouped_ihc.is_file():
        return grouped_he, grouped_ihc
    return TILES512 / "donor_he_shuf_512.npy", TILES512 / "donor_ihc_shuf_512.npy"


def _gpu_mem_str() -> str:
    """Per-visible-GPU used MiB (nvidia-style: total - free)."""
    parts = []
    n = torch.cuda.device_count()
    for i in range(n):
        try:
            free, total = torch.cuda.mem_get_info(i)
            used = (total - free) / (1024.0 * 1024.0)
        except Exception:
            used = torch.cuda.memory_reserved(i) / (1024.0 * 1024.0)
        parts.append(f"gpu{i}_mem={used:.0f}MiB")
    return " ".join(parts) if parts else "gpu_mem=n/a"


def _unwrap(m: nn.Module) -> nn.Module:
    return m.module if isinstance(m, (nn.DataParallel, nn.parallel.DistributedDataParallel)) else m


class Her2JpegPairDataset(torch.utils.data.Dataset):
    """Stream paired HE/IHC JPEGs; bilinear 1024 -> 512. Never materializes all tiles."""

    def __init__(self, root: Path, split: str, names: list[str] | None = None):
        self.he_dir = root / "HE" / split
        self.ihc_dir = root / "IHC" / split
        if not self.he_dir.is_dir() or not self.ihc_dir.is_dir():
            raise SystemExit(f"Missing HER2 dirs: {self.he_dir} / {self.ihc_dir}")
        if names is None:
            he_names = {p.name for p in self.he_dir.glob("*.jpg")}
            ihc_names = {p.name for p in self.ihc_dir.glob("*.jpg")}
            self.names = sorted(he_names & ihc_names)
        else:
            self.names = list(names)
        if not self.names:
            raise SystemExit(f"No paired JPEGs in {self.he_dir} and {self.ihc_dir}")

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, idx: int):
        from PIL import Image

        name = self.names[idx]
        him = Image.open(self.he_dir / name).convert("RGB")
        iim = Image.open(self.ihc_dir / name).convert("RGB")
        him = him.resize((TILE, TILE), Image.Resampling.BILINEAR)
        iim = iim.resize((TILE, TILE), Image.Resampling.BILINEAR)
        he = torch.from_numpy(np.asarray(him)).float().permute(2, 0, 1) / 255.0
        ihc = torch.from_numpy(np.asarray(iim)).float().permute(2, 0, 1) / 255.0
        return he, ihc, name


# ---------------------------------------------------------------------------
# Viz
# ---------------------------------------------------------------------------


def _to_u8(img01: torch.Tensor) -> np.ndarray:
    x = img01.detach().float().clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
    return (x * 255.0 + 0.5).astype(np.uint8)


def save_recon_panel(path: Path, true_b, soft_b, hard_b):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 3, figsize=(9, 3.2))
    for ax, im, title in zip(
        axs,
        [_to_u8(true_b), _to_u8(soft_b), _to_u8(hard_b)],
        ["true B_IHC", "soft", "hard"],
    ):
        ax.imshow(im)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_recon_panel_4(
    path: Path,
    he_b: torch.Tensor,
    true_b: torch.Tensor,
    soft_b: torch.Tensor,
    hard_b: torch.Tensor,
):
    """4 val rows: HE | true IHC | soft | hard."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = he_b.shape[0]
    fig, axs = plt.subplots(n, 4, figsize=(12, 3.0 * n))
    if n == 1:
        axs = np.array([axs])
    titles = ["HE", "true IHC", "soft", "hard"]
    for i in range(n):
        imgs = [
            _to_u8(he_b[i : i + 1]),
            _to_u8(true_b[i : i + 1]),
            _to_u8(soft_b[i : i + 1]),
            _to_u8(hard_b[i : i + 1]),
        ]
        for j, (im, title) in enumerate(zip(imgs, titles)):
            axs[i, j].imshow(im)
            if i == 0:
                axs[i, j].set_title(title, fontsize=10)
            axs[i, j].axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_W_queries(path: Path, W: torch.Tensor, query_cells: list[tuple[int, int]]):
    """W: 1,4096,4096. query_cells as (row,col) on 64x64 grid."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    W0 = W[0].detach().float().cpu()
    fig, axs = plt.subplots(1, len(query_cells), figsize=(3.2 * len(query_cells), 3.0))
    if len(query_cells) == 1:
        axs = [axs]
    for ax, (r, c) in zip(axs, query_cells):
        q = r * 64 + c
        heat = W0[q].reshape(64, 64).numpy()
        ax.imshow(heat, cmap="magma", interpolation="nearest", vmin=0, vmax=heat.max() + 1e-8)
        ax.set_title(f"q=({r},{c})", fontsize=9)
        ax.axis("off")
    fig.suptitle("W rows (64x64, nearest)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _nn_up_u8(img_hwc: np.ndarray, out: int = 256) -> np.ndarray:
    """Nearest-neighbor upsample HxWxC to out x out (uint8)."""
    t = torch.from_numpy(np.ascontiguousarray(img_hwc)).float()
    if t.max() > 1.5:
        t = t / 255.0
    if t.ndim == 2:
        t = t.unsqueeze(-1)
    t = t.permute(2, 0, 1).unsqueeze(0)
    t = F.interpolate(t, size=(out, out), mode="nearest")
    arr = t.squeeze(0).permute(1, 2, 0).clamp(0, 1).numpy()
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    return (arr * 255.0 + 0.5).astype(np.uint8)


@torch.no_grad()
def softmax_corr_row(Fb: torch.Tensor, Fa: torch.Tensor, q_rc: tuple[int, int], tau: float = TAU):
    """L2-normalize, S/tau, softmax over A for one B query. Returns heat, argmax (r,c), max_p."""
    _, _C, H, W = Fb.shape
    fb = F.normalize(Fb.flatten(2).transpose(1, 2).float(), dim=-1)
    fa = F.normalize(Fa.flatten(2).transpose(1, 2).float(), dim=-1)
    r, c = q_rc
    q = r * W + c
    S = torch.matmul(fb[:, q : q + 1], fa.transpose(1, 2)) / tau
    prob = torch.softmax(S, dim=-1).squeeze(0).squeeze(0)
    heat = prob.reshape(H, W).cpu().numpy()
    amax = int(prob.argmax().item())
    return heat, (amax // W, amax % W), float(prob.max().item())


def _scale_query(r64: int, c64: int, grid: int) -> tuple[int, int]:
    factor = 64 // grid
    return r64 // factor, c64 // factor


@torch.no_grad()
def visualize_scale_correspondence(
    model: nn.Module,
    he_b: torch.Tensor,
    he_a: torch.Tensor,
    out_dir: Path,
    device: torch.device,
    query_cells_64: list[tuple[int, int, str]] | None = None,
):
    """Per-scale softmax correspondence panels + contact sheet + FPN panel."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    out_dir.mkdir(parents=True, exist_ok=True)
    if query_cells_64 is None:
        query_cells_64 = QUERY_CELLS_64

    model.eval()
    use_bf16 = device.type == "cuda"
    dtype_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_bf16 else nullcontext()
    )
    with dtype_ctx:
        _Fb, _Fa, stages = model.encode(he_b, he_a, return_stages=True)

    he_a_u8 = _to_u8(he_a)
    side = int(he_b.shape[-1])
    scale_meta = [
        ("stage1", 64, stages["stage1"]),
        ("stage2", 32, stages["stage2"]),
        ("stage3", 16, stages["stage3"]),
        ("stage4", 8, stages["stage4"]),
        ("FPN", 64, stages["FPN"]),
    ]

    readings: dict[str, dict] = {}

    for name, grid, (Fb_s, Fa_s) in scale_meta:
        Fb_s = Fb_s.float()
        Fa_s = Fa_s.float()
        b_tile = (
            F.adaptive_avg_pool2d(he_b.float(), (grid, grid))
            .squeeze(0)
            .permute(1, 2, 0)
            .clamp(0, 1)
            .cpu()
            .numpy()
        )
        b_disp = _nn_up_u8(b_tile, side)
        cell = side / grid
        px = side // grid

        fig, axs = plt.subplots(len(query_cells_64), 4, figsize=(12, 3.0 * len(query_cells_64)))
        maxes = []
        row_notes = []
        for i, (r64, c64, label) in enumerate(query_cells_64):
            rq, cq = _scale_query(r64, c64, grid)
            heat, (ar, ac), max_p = softmax_corr_row(Fb_s, Fa_s, (rq, cq), tau=TAU)
            maxes.append(max_p)
            row_notes.append((label, (rq, cq), (ar, ac), max_p))

            axs[i, 0].imshow(b_disp)
            axs[i, 0].add_patch(
                Rectangle((cq * cell, rq * cell), cell, cell, fill=False, edgecolor="lime", lw=2)
            )
            axs[i, 0].set_title(f"B {name} q={label}\n({rq},{cq})/{grid}", fontsize=8)
            axs[i, 0].axis("off")

            axs[i, 1].imshow(he_a_u8)
            axs[i, 1].add_patch(
                Rectangle((ac * cell, ar * cell), cell, cell, fill=False, edgecolor="cyan", lw=2)
            )
            axs[i, 1].set_title(f"A HE  argmax=({ar},{ac})", fontsize=8)
            axs[i, 1].axis("off")

            axs[i, 2].imshow(
                heat, cmap="magma", interpolation="nearest", vmin=0, vmax=max(float(heat.max()), 1e-8)
            )
            axs[i, 2].set_title(f"softmax heat  max={max_p:.3f}", fontsize=8)
            axs[i, 2].axis("off")

            y0, y1 = ar * px, (ar + 1) * px
            x0, x1 = ac * px, (ac + 1) * px
            patch = he_a_u8[y0:y1, x0:x1]
            axs[i, 3].imshow(_nn_up_u8(patch, side))
            axs[i, 3].set_title("argmax patch on A", fontsize=8)
            axs[i, 3].axis("off")

        mean_max = float(np.mean(maxes))
        readings[name] = {"mean_max_softmax": mean_max, "rows": row_notes, "grid": grid}
        fig.suptitle(f"{name} ({grid}x{grid})  mean max-softmax={mean_max:.4f}", fontsize=11)
        fig.tight_layout()
        png = out_dir / ("corr_FPN64.png" if name == "FPN" else f"corr_{name}.png")
        fig.savefig(png, dpi=120)
        plt.close(fig)
        print(f"  saved {png.name}  mean_max_softmax={mean_max:.4f}", flush=True)

    # Contact sheet: rows=scales, cols=queries; B (NN) | heat side by side
    fig, axs = plt.subplots(len(scale_meta), 4, figsize=(10, 2.4 * len(scale_meta)))
    for row, (name, grid, (Fb_s, Fa_s)) in enumerate(scale_meta):
        Fb_s = Fb_s.float()
        Fa_s = Fa_s.float()
        b_tile = (
            F.adaptive_avg_pool2d(he_b.float(), (grid, grid))
            .squeeze(0)
            .permute(1, 2, 0)
            .clamp(0, 1)
            .cpu()
            .numpy()
        )
        b_disp = _nn_up_u8(b_tile, side)
        cell = side / grid
        for col, (r64, c64, label) in enumerate(query_cells_64):
            rq, cq = _scale_query(r64, c64, grid)
            heat, _arc, max_p = softmax_corr_row(Fb_s, Fa_s, (rq, cq), tau=TAU)
            heat_rgb = plt.cm.magma(heat / (heat.max() + 1e-8))[:, :, :3]
            heat_u8 = _nn_up_u8(heat_rgb, side)
            combo = np.concatenate([b_disp, heat_u8], axis=1)
            ax = axs[row, col]
            ax.imshow(combo)
            ax.add_patch(
                Rectangle((cq * cell, rq * cell), cell, cell, fill=False, edgecolor="lime", lw=1.5)
            )
            ax.set_title(f"{name}/{label} max={max_p:.2f}", fontsize=7)
            ax.axis("off")
    fig.suptitle("Contact sheet: B (NN) | softmax heat per scale/query", fontsize=11)
    fig.tight_layout()
    sheet = out_dir / "corr_contact_sheet.png"
    fig.savefig(sheet, dpi=130)
    plt.close(fig)
    print(f"  saved {sheet.name}", flush=True)

    summary_path = out_dir / "corr_readings.txt"
    lines = []
    for name, meta in readings.items():
        lines.append(f"{name} grid={meta['grid']} mean_max_softmax={meta['mean_max_softmax']:.4f}")
        for label, qrc, arc, mx in meta["rows"]:
            lines.append(f"  {label} q={qrc} argmax={arc} max={mx:.4f}")
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return readings, sheet


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def require_cuda(device_name: str = "cuda") -> torch.device:
    """CUDA-only. Abort if missing; never fall back to CPU."""
    if device_name != "cuda":
        raise SystemExit(
            f"Refused: device={device_name!r}. MatchFormer Test A must run on CUDA only."
        )
    if not torch.cuda.is_available():
        raise SystemExit("Refused: torch.cuda.is_available() is False. No CPU fallback.")
    return torch.device("cuda")


def train(args):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = require_cuda(getattr(args, "device", "cuda"))
    use_bf16 = True
    dtype_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)

    he_b, ihc_b, he_a, ihc_a = load_pair()
    he_b = he_b.to(device)
    ihc_b = ihc_b.to(device)
    he_a = he_a.to(device)
    ihc_a = ihc_a.to(device)
    if he_a.shape[0] == 1 and he_b.shape[0] > 1:
        he_a = he_a.expand(he_b.shape[0], -1, -1, -1).contiguous()
        ihc_a = ihc_a.expand(he_b.shape[0], -1, -1, -1).contiguous()

    assert he_a.shape[0] == he_b.shape[0]
    B = he_b.shape[0]
    # Microbatch correspondence when B is large (16x 4096^2 W can OOM).
    micro = int(os.environ.get("MATCHFORMER_MICROBATCH", "0"))
    if micro <= 0:
        micro = 4 if B > 8 else B
    micro = min(micro, B)

    # Gradient checkpointing on stages (spec-allowed) to fit 4060 8GB.
    model = MatchFormerPair(use_checkpoint=True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    log_path = OUT_DIR / "train_log.txt"
    lines = []
    gpu_name = torch.cuda.get_device_name(0)
    header = (
        f"Test A MatchFormer-pair overfit\n"
        f"donor={DONOR_PANEL.name} tile={TILE}\n"
        f"B_slot={B_SLOT} B_name={B_NAME}\n"
        f"n_query={B} microbatch={micro}\n"
        f"range=[0,1] tau={TAU} topk={TOPK} steps={args.steps} "
        f"device={device} gpu={gpu_name} bf16={use_bf16}\n"
        f"params={count_params(model)/1e6:.2f}M\n"
    )
    print(header, end="")
    lines.append(header)

    query_cells = [(r, c) for r, c, _ in QUERY_CELLS_64]
    t0 = time.time()
    last_print = ""
    first_print = ""

    patches_a = patches_4x4(ihc_a)
    console_path = OUT_DIR / "console.txt"
    console_path.write_text(header, encoding="utf-8")

    def log_line(s: str):
        print(s, flush=True)
        with console_path.open("a", encoding="utf-8") as f:
            f.write(s + "\n")

    def save_ckpt(step: int) -> Path:
        path = OUT_DIR / f"matchformer_pair_step{step:04d}.pt"
        torch.save(model.state_dict(), path)
        log_line(f"  saved ckpt {path.name}")
        return path

    for step in range(1, args.steps + 1):
        try:
            model.train()
            opt.zero_grad(set_to_none=True)
            do_print = step % PRINT_EVERY == 0 or step == 1
            do_png = (step % PNG_EVERY == 0) or (step == args.steps)

            L_contrast_acc = 0.0
            L_soft_acc = 0.0
            L_dab_acc = 0.0
            L_hard_acc = 0.0
            L_dab_hard_acc = 0.0
            mean_max_w_acc = 0.0
            soft_vis = hard_vis = W_vis = None
            n_seen = 0

            for s in range(0, B, micro):
                e = min(s + micro, B)
                mb = e - s
                with dtype_ctx:
                    Fb, Fa = model.encode(he_b[s:e], he_a[s:e])
                L_contrast = contrastive_correspondence_loss(
                    Fb, Fa, ihc_b[s:e], patches_a[s:e], tau=TAU
                )
                loss = L_contrast * (mb / B)
                loss.backward()
                L_contrast_v_mb = float(L_contrast.detach())
                # Free S graph before allocating W (do not hold both).
                del L_contrast, loss

                with torch.no_grad():
                    L_contrast_acc += L_contrast_v_mb * mb
                    W = correspondence(Fb.detach(), Fa.detach(), tau=TAU)
                    soft, _, _ = soft_transfer(W, patches_a[s:e], topk=TOPK)
                    L_soft = F.l1_loss(soft, ihc_b[s:e])
                    L_dab_soft = F.l1_loss(dab_hed_ch2(soft), dab_hed_ch2(ihc_b[s:e]))
                    L_soft_acc += float(L_soft) * mb
                    L_dab_acc += float(L_dab_soft) * mb
                    if do_print or do_png:
                        hard = hard_transfer(W, patches_a[s:e])
                        L_hard_acc += F.l1_loss(hard, ihc_b[s:e]).item() * mb
                        L_dab_hard_acc += (
                            F.l1_loss(dab_hed_ch2(hard), dab_hed_ch2(ihc_b[s:e])).item() * mb
                        )
                        mean_max_w_acc += W.max(dim=-1).values.mean().item() * mb
                        if do_png and s == 0:
                            soft_vis = soft.float().cpu()
                            hard_vis = hard.float().cpu()
                            W_vis = W.float().cpu()
                        del hard
                    del W, soft, L_soft, L_dab_soft
                n_seen += mb
                del Fb, Fa

            L_contrast_v = L_contrast_acc / n_seen
            L_soft_v = L_soft_acc / n_seen
            L_dab_v = L_dab_acc / n_seen
            L_hard = L_hard_acc / n_seen if do_print or do_png else float("nan")
            L_dab_hard = L_dab_hard_acc / n_seen if do_print or do_png else float("nan")
            mean_max_w = mean_max_w_acc / n_seen if do_print or do_png else float("nan")

            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            if step % 20 == 0:
                gc.collect()
                torch.cuda.empty_cache()

            if do_print:
                msg = (
                    f"step {step:04d}  L_contrast={L_contrast_v:.4f}  "
                    f"L_soft={L_soft_v:.4f}  L_dab_soft={L_dab_v:.4f}  "
                    f"L_hard={L_hard:.4f}  L_dab_hard={L_dab_hard:.4f}  "
                    f"mean_maxW={mean_max_w:.4f}"
                )
                log_line(msg)
                lines.append(msg + "\n")
                last_print = msg
                if step == 1:
                    first_print = msg

            if do_png and soft_vis is not None:
                png = OUT_DIR / f"recon_step{step:04d}.png"
                save_recon_panel(png, ihc_b[:1].cpu(), soft_vis[:1], hard_vis[:1])
                wpng = OUT_DIR / f"W_queries_step{step:04d}.png"
                save_W_queries(wpng, W_vis, query_cells)
                log_line(f"  saved {png.name} {wpng.name}")
                del soft_vis, hard_vis, W_vis

            # Checkpoint at step 400 only (plus normal completion below).
            if step == 400:
                save_ckpt(step)
        except RuntimeError as e:
            log_line(f"FAIL at step {step}: {e}")
            try:
                gc.collect()
                torch.cuda.empty_cache()
            except Exception:
                pass
            raise

    elapsed = time.time() - t0
    final_png = OUT_DIR / f"recon_step{args.steps:04d}.png"
    # Normal completion save (same path when steps==400).
    ckpt = save_ckpt(args.steps)
    footer = (
        f"DONE steps={args.steps} elapsed_s={elapsed:.1f} "
        f"final_png={final_png} ckpt={ckpt}\n"
    )
    print(footer, end="")
    lines.append(footer)
    log_path.write_text("".join(lines), encoding="utf-8")

    if MD_PATH.exists():
        md = MD_PATH.read_text(encoding="utf-8")
        appendix = (
            f"\n\n### Run result ({time.strftime('%Y-%m-%d %H:%M')}) retrain+ckpt\n"
            f"- First print: `{first_print or 'n/a'}`\n"
            f"- Last print: `{last_print}`\n"
            f"- Final PNG: `{final_png.as_posix()}`\n"
            f"- Ckpt: `{ckpt.as_posix()}`\n"
            f"- Log: `{log_path.as_posix()}`\n"
        )
        MD_PATH.write_text(md.rstrip() + appendix, encoding="utf-8")

    return last_print, final_png, ckpt


def _micro_for(batch_size: int) -> int:
    micro = int(os.environ.get("MATCHFORMER_MICROBATCH", "0"))
    if micro <= 0:
        micro = 4 if batch_size > 8 else batch_size
    return max(1, min(micro, batch_size))


@torch.no_grad()
def _val_panel_forward(
    model: nn.Module,
    he_b: torch.Tensor,
    ihc_b: torch.Tensor,
    he_a: torch.Tensor,
    patches_a: torch.Tensor,
    dtype_ctx,
):
    """Soft/hard recons for a small fixed val batch (logging / PNG only)."""
    B = he_b.shape[0]
    raw = _unwrap(model)
    independent = getattr(raw, "donor_mode", "shared") == "independent"
    soft_list, hard_list = [], []
    micro = _micro_for(B)
    for s in range(0, B, micro):
        e = min(s + micro, B)
        with dtype_ctx:
            if independent:
                Fb, Fa = raw.encode(he_b[s:e], he_a)
            else:
                Fb, Fa = raw.encode(he_b[s:e], he_a.expand(e - s, -1, -1, -1))
        W = correspondence(Fb, Fa, tau=TAU)
        soft, _, _ = soft_transfer(W, patches_a.expand(e - s, -1, -1), topk=TOPK)
        hard = hard_transfer(W, patches_a.expand(e - s, -1, -1))
        soft_list.append(soft.float().cpu())
        hard_list.append(hard.float().cpu())
        del Fb, Fa, W, soft, hard
    return torch.cat(soft_list, 0), torch.cat(hard_list, 0)


def train_full(args):
    """Full HER2 train-split MatchFormer (same model/loss as overfit)."""
    from torch.utils.data import DataLoader, DistributedSampler

    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    use_ddp = local_rank >= 0 and world_size > 1
    if use_ddp:
        import torch.distributed as dist

        torch.cuda.set_device(local_rank)
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        device = torch.device("cuda", local_rank)
        is_main = local_rank == 0
        rank = dist.get_rank()
    else:
        device = require_cuda(getattr(args, "device", "cuda"))
        is_main = True
        rank = 0
        local_rank = 0

    if is_main:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "ckpts").mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "panels").mkdir(parents=True, exist_ok=True)
    if use_ddp:
        import torch.distributed as dist

        dist.barrier()

    ckpt_dir = OUT_DIR / "ckpts"
    panel_dir = OUT_DIR / "panels"
    ckpt_stem = os.environ.get("MATCHFORMER_CKPT_STEM", "matchformer_full")
    dtype_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)

    her2 = Path(os.environ.get("MATCHFORMER_HER2", str(HER2_ROOT)))
    train_list = os.environ.get("MATCHFORMER_TRAIN_LIST", "").strip()
    if train_list:
        train_names = [
            ln.strip()
            for ln in Path(train_list).read_text(encoding="utf-8").splitlines()
            if ln.strip().endswith(".jpg")
        ]
        if not train_names:
            raise SystemExit(f"empty train list {train_list}")
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
        val_ds = Her2JpegPairDataset(her2, "val")
        panel_names = val_ds.names[:VAL_PANEL_N]
    if len(panel_names) < 1:
        raise SystemExit("val panel list is empty")
    panel_ds = Her2JpegPairDataset(her2, "val", names=panel_names)

    batch_size = int(getattr(args, "batch_size", BATCH_SIZE_FULL))
    epochs = int(getattr(args, "epochs", NUM_EPOCHS_FULL))
    micro = _micro_for(batch_size)
    n_workers = int(os.environ.get("MATCHFORMER_WORKERS", "4"))

    sampler = None
    if use_ddp:
        sampler = DistributedSampler(
            train_ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False
        )
    loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=n_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=n_workers > 0,
    )
    panel_loader = DataLoader(panel_ds, batch_size=len(panel_names), shuffle=False)

    he_a, ihc_a = load_donor()
    he_a = he_a.to(device)
    ihc_a = ihc_a.to(device)
    patches_a = patches_4x4(ihc_a)  # 1,4096,192
    donor_he_path, _donor_ihc_path = _donor_paths()

    he_panel, ihc_panel, _panel_batch_names = next(iter(panel_loader))
    he_panel = he_panel.to(device)
    ihc_panel = ihc_panel.to(device)

    use_global_pos = POS_MODE == "global"
    donor_mode = DONOR_MODE
    pos_cache: dict[str, torch.Tensor] | None = None
    panel_pos: torch.Tensor | None = None
    if donor_mode == "independent":
        assert_medoid_scatter_roundtrip()
        if is_main:
            print(
                f"DONOR_MODE=independent local_depth={DONOR_LOCAL_DEPTH} "
                f"(query SELF-only hierarchy; no encoder CROSS)",
                flush=True,
            )
    if use_global_pos:
        assert_medoid_index_alignment()
        if is_main:
            print(
                f"POS_MODE=global dir={GLOBAL_POS_DIR} store_k={GLOBAL_POS_STORE_K} "
                f"train_n_pos={CORR_N_POS}",
                flush=True,
            )
        pos_cache = load_global_pos_cache(GLOBAL_POS_DIR, train_ds.names, n_pos=CORR_N_POS)
        panel_pos = gather_pos_idx_batch(
            load_global_pos_cache(GLOBAL_POS_DIR, panel_names, n_pos=GLOBAL_POS_STORE_K),
            panel_names,
            device,
        )
    elif is_main:
        print(f"POS_MODE={POS_MODE} (legacy Top-{CORR_CAND_TOPK} then IHC mine)", flush=True)

    model = MatchFormerPair(use_checkpoint=True, donor_mode=donor_mode).to(device)
    init_ckpt = os.environ.get("MATCHFORMER_INIT_CKPT", "").strip()
    init_epoch = 0
    if init_ckpt:
        blob = torch.load(init_ckpt, map_location=device, weights_only=False)
        state = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
        # Exp B architecture differs (SSS + donor_local); load overlapping keys (stem/FPN).
        strict = donor_mode == "shared"
        incompatible = model.load_state_dict(state, strict=strict)
        if isinstance(blob, dict):
            init_epoch = int(blob.get("epoch", 0))
        if is_main:
            if strict:
                print(f"warm start {init_ckpt} source_epoch={init_epoch}", flush=True)
            else:
                miss = list(incompatible.missing_keys)
                unexp = list(incompatible.unexpected_keys)
                print(
                    f"warm start {init_ckpt} source_epoch={init_epoch} strict=False "
                    f"missing={len(miss)} unexpected={len(unexp)} "
                    f"(stem/partial query stages; donor_local fresh)",
                    flush=True,
                )
    n_vis = torch.cuda.device_count()
    use_dp = (not use_ddp) and n_vis > 1
    if use_ddp:
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False
        )
    elif use_dp:
        model = nn.DataParallel(model)
    opt = torch.optim.AdamW(_unwrap(model).parameters(), lr=LR, weight_decay=0.01)
    disc = None
    disc_opt = None
    if LAMBDA_ADV > 0:
        disc = PatchGANDiscriminator().to(device)
        disc_opt = torch.optim.AdamW(
            disc.parameters(), lr=LR * 0.5, betas=(0.5, 0.999), weight_decay=0.0
        )
        if init_ckpt:
            blob_d = torch.load(init_ckpt, map_location=device, weights_only=False)
            if isinstance(blob_d, dict) and "disc" in blob_d:
                disc.load_state_dict(blob_d["disc"])
                if is_main:
                    print(f"warm start disc from {init_ckpt}", flush=True)
        if is_main:
            print(
                f"PatchGAN lambda_adv={LAMBDA_ADV} disc_params={count_params(disc)/1e6:.2f}M",
                flush=True,
            )

    console_path = OUT_DIR / "console.txt"
    log_path = OUT_DIR / "train_log.txt"
    gpu_names = [torch.cuda.get_device_name(i) for i in range(n_vis)]
    eff_batch = batch_size * (world_size if use_ddp else 1)
    parallel = (
        f"ddp world={world_size} local_rank={local_rank}"
        if use_ddp
        else (f"DataParallel n={n_vis}" if use_dp else "single_gpu")
    )
    header = (
        f"MatchFormer-pair FULL train\n"
        f"her2={her2} n_train={len(train_ds)} n_val_panel={len(panel_ds)} "
        f"warm_start_epoch={init_epoch} lr={LR} "
        f"panel={panel_names}\n"
        f"donor={donor_he_path}\n"
        f"batch_per_rank={batch_size} effective_batch={eff_batch} "
        f"microbatch={micro} epochs={epochs} "
        f"tile={TILE} tau={TAU} topk={TOPK} lambda_adv={LAMBDA_ADV} "
        f"pos_mode={POS_MODE} n_pos={CORR_N_POS} donor_mode={donor_mode}\n"
        f"device={device} gpus={gpu_names} parallel={parallel} "
        f"bf16=True params={count_params(_unwrap(model))/1e6:.2f}M\n"
        f"ckpt_stem={ckpt_stem}\n"
    )
    if is_main:
        print(header, end="", flush=True)
        console_path.write_text(header, encoding="utf-8")
    lines = [header]

    def log_line(s: str):
        if not is_main:
            return
        print(s, flush=True)
        with console_path.open("a", encoding="utf-8") as f:
            f.write(s + "\n")
        lines.append(s + "\n")

    global_step = 0
    t0 = time.time()
    first_epoch_msg = ""
    last_epoch_msg = ""

    for epoch in range(1, epochs + 1):
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
        L_contrast_acc = L_soft_acc = L_dab_acc = 0.0
        L_hard_acc = L_dab_hard_acc = mean_max_w_acc = 0.0
        L_gan_acc = L_d_acc = 0.0
        n_seen = 0
        for he_b, ihc_b, names in loader:
            he_b = he_b.to(device, non_blocking=True)
            ihc_b = ihc_b.to(device, non_blocking=True)
            B = he_b.shape[0]
            if donor_mode == "independent":
                he_a_b = he_a  # encode once; Fa expands inside model
            else:
                he_a_b = he_a.expand(B, -1, -1, -1).contiguous()
            patches_a_b = patches_a.expand(B, -1, -1).contiguous()
            pos_b = (
                gather_pos_idx_batch(pos_cache, list(names), device)
                if use_global_pos
                else None
            )

            opt.zero_grad(set_to_none=True)
            for s in range(0, B, micro):
                e = min(s + micro, B)
                mb = e - s
                with dtype_ctx:
                    if donor_mode == "independent":
                        Fb, Fa = model(he_b[s:e], he_a_b)
                    else:
                        Fb, Fa = model(he_b[s:e], he_a_b[s:e])
                if use_global_pos:
                    L_contrast = contrastive_correspondence_loss(
                        Fb, Fa, tau=TAU, pos_idx=pos_b[s:e]
                    )
                else:
                    L_contrast = contrastive_correspondence_loss(
                        Fb, Fa, ihc_b[s:e], patches_a_b[s:e], tau=TAU
                    )
                loss = L_contrast * (mb / B)
                loss.backward()
                L_contrast_v_mb = float(L_contrast.detach())
                # Free S graph before allocating W (do not hold both).
                del L_contrast, loss, Fb, Fa

                L_contrast_acc += L_contrast_v_mb * mb
                if disc is not None:
                    with dtype_ctx:
                        if donor_mode == "independent":
                            Fb, Fa = model(he_b[s:e], he_a_b)
                        else:
                            Fb, Fa = model(he_b[s:e], he_a_b[s:e])
                    W = correspondence(Fb, Fa, tau=TAU)
                    soft, _, _ = soft_transfer(W, patches_a_b[s:e], topk=TOPK)
                    g_adv, d_loss = _lsgan_on_soft(
                        disc, disc_opt, he_b[s:e], ihc_b[s:e], soft, mb, B, dtype_ctx
                    )
                    L_gan_acc += g_adv * mb
                    L_d_acc += d_loss * mb
                    W = W.detach()
                    soft = soft.detach()
                    del Fb, Fa
                else:
                    with torch.no_grad():
                        if donor_mode == "independent":
                            Fb, Fa = model(he_b[s:e], he_a_b)
                        else:
                            Fb, Fa = model(he_b[s:e], he_a_b[s:e])
                    W = correspondence(Fb, Fa, tau=TAU)
                    soft, _, _ = soft_transfer(W, patches_a_b[s:e], topk=TOPK)
                    del Fb, Fa

                with torch.no_grad():
                    L_soft = F.l1_loss(soft, ihc_b[s:e])
                    L_dab_soft = F.l1_loss(dab_hed_ch2(soft), dab_hed_ch2(ihc_b[s:e]))
                    L_soft_acc += float(L_soft) * mb
                    L_dab_acc += float(L_dab_soft) * mb
                    hard = hard_transfer(W, patches_a_b[s:e])
                    L_hard_acc += F.l1_loss(hard, ihc_b[s:e]).item() * mb
                    L_dab_hard_acc += (
                        F.l1_loss(dab_hed_ch2(hard), dab_hed_ch2(ihc_b[s:e])).item() * mb
                    )
                    mean_max_w_acc += W.max(dim=-1).values.mean().item() * mb
                    del hard, W, soft, L_soft, L_dab_soft
                n_seen += mb

            torch.nn.utils.clip_grad_norm_(_unwrap(model).parameters(), GRAD_CLIP)
            opt.step()
            global_step += 1
            del he_b, ihc_b, he_a_b, patches_a_b
            if pos_b is not None:
                del pos_b

        # Average metrics across ranks when DDP.
        stats = torch.tensor(
            [
                L_contrast_acc,
                L_soft_acc,
                L_dab_acc,
                L_hard_acc,
                L_dab_hard_acc,
                mean_max_w_acc,
                L_gan_acc,
                L_d_acc,
                float(n_seen),
            ],
            device=device,
            dtype=torch.float64,
        )
        if use_ddp:
            import torch.distributed as dist

            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        n_tot = max(float(stats[8].item()), 1.0)
        L_contrast_v = float(stats[0].item()) / n_tot
        L_soft_v = float(stats[1].item()) / n_tot
        L_dab_v = float(stats[2].item()) / n_tot
        L_hard = float(stats[3].item()) / n_tot
        L_dab_hard = float(stats[4].item()) / n_tot
        mean_max_w = float(stats[5].item()) / n_tot
        L_gan_v = float(stats[6].item()) / n_tot
        L_d_v = float(stats[7].item()) / n_tot
        mem_s = _gpu_mem_str()
        msg = (
            f"epoch {epoch:04d}  step {global_step}  "
            f"L_contrast={L_contrast_v:.4f}  "
            f"L_gan={L_gan_v:.4f}  L_d={L_d_v:.4f}  "
            f"L_soft={L_soft_v:.4f}  L_dab_soft={L_dab_v:.4f}  "
            f"L_hard={L_hard:.4f}  L_dab_hard={L_dab_hard:.4f}  "
            f"mean_maxW={mean_max_w:.4f}  {mem_s}"
        )
        log_line(msg)
        last_epoch_msg = msg
        if epoch == 1:
            first_epoch_msg = msg

        if is_main:
            ckpt_path = ckpt_dir / f"{ckpt_stem}_epoch{epoch:04d}.pt"
            payload = {
                "model": _unwrap(model).state_dict(),
                "step": global_step,
                "epoch": epoch,
            }
            if disc is not None:
                payload["disc"] = disc.state_dict()
            torch.save(payload, ckpt_path)
            log_line(f"  saved ckpt {ckpt_path.name}")

            raw = _unwrap(model)
            raw.eval()
            soft_p, hard_p = _val_panel_forward(
                raw, he_panel, ihc_panel, he_a, patches_a, dtype_ctx
            )
            png = panel_dir / f"recon_epoch{epoch:04d}.png"
            save_recon_panel_4(png, he_panel.cpu(), ihc_panel.cpu(), soft_p, hard_p)
            log_line(f"  saved panel {png.name}")
            if use_global_pos and panel_pos is not None:
                with torch.no_grad(), dtype_ctx:
                    Fb_p, Fa_p = raw(he_panel, he_a.expand(he_panel.shape[0], -1, -1, -1))
                recalls = oracle_ihc_target_recall(Fb_p, Fa_p, panel_pos, tau=TAU)
                del Fb_p, Fa_p
                rec_s = "  ".join(f"{k}={v:.4f}" for k, v in recalls.items())
                log_line(f"  panel {rec_s}")
            del soft_p, hard_p
            raw.train()
            vol_name = os.environ.get("MATCHFORMER_VOL_COMMIT", "").strip()
            if vol_name:
                try:
                    import modal as _modal

                    _modal.Volume.from_name(vol_name).commit()
                    log_line(f"  volume commit {vol_name}")
                except Exception as exc:
                    log_line(f"  volume commit failed {type(exc).__name__}: {exc}")
        gc.collect()
        torch.cuda.empty_cache()
        if use_ddp:
            import torch.distributed as dist

            dist.barrier()

    elapsed = time.time() - t0
    footer = (
        f"DONE epochs={epochs} steps={global_step} elapsed_s={elapsed:.1f} "
        f"first={first_epoch_msg!r} last={last_epoch_msg!r}\n"
    )
    log_line(footer.rstrip())
    if is_main:
        log_path.write_text("".join(lines), encoding="utf-8")
    if use_ddp:
        import torch.distributed as dist

        dist.barrier()
        dist.destroy_process_group()
    return last_epoch_msg, ckpt_dir


def precompute_global_positives(args):
    """Write per-tile global IHC Top-K indices under GLOBAL_POS_DIR (Experiment A)."""
    assert_medoid_index_alignment()
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
    if val_list:
        val_names = [
            ln.strip()
            for ln in Path(val_list).read_text(encoding="utf-8").splitlines()
            if ln.strip().endswith(".jpg")
        ]
    else:
        val_names = []
    names = list(dict.fromkeys([*train_names, *val_names]))
    GLOBAL_POS_DIR.mkdir(parents=True, exist_ok=True)
    _he_a, ihc_a = load_donor()
    patches_d = patches_4x4(ihc_a.to(device))  # 1,4096,192
    print(
        f"precompute_global n={len(names)} out={GLOBAL_POS_DIR} "
        f"k={store_k} device={device}",
        flush=True,
    )
    he_train = her2 / "HE" / "train"
    he_val = her2 / "HE" / "val"

    t0 = time.time()
    done = 0
    skipped = 0
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
            raise FileNotFoundError(f"tile not in train or val: {name}")
        tile_ds = Her2JpegPairDataset(her2, split, names=[name])
        _he, ihc, _n = tile_ds[0]
        ihc = ihc.unsqueeze(0).to(device)
        pq = patches_4x4(ihc)
        idx, err = compute_global_ihc_topk(
            pq, patches_d, k=store_k, q_chunk=128
        )
        save_global_pos_tile(out_p, idx, err)
        done += 1
        if done % 50 == 0 or done == len(names):
            print(
                f"  {done}/{len(names)} skipped={skipped} "
                f"elapsed_s={time.time()-t0:.1f} last={stem}",
                flush=True,
            )
    print(
        f"DONE precompute_global wrote/kept={done} skipped_existing={skipped} "
        f"dir={GLOBAL_POS_DIR}",
        flush=True,
    )


def run_viz(ckpt_path: Path | None = None):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = require_cuda("cuda")
    ckpt_path = ckpt_path or CKPT_PATH
    if not ckpt_path.exists():
        raise FileNotFoundError(f"missing ckpt: {ckpt_path}")

    he_b, _ihc_b, he_a, _ihc_a = load_pair()
    he_b = he_b[:1].to(device)
    he_a = he_a[:1].to(device)

    model = MatchFormerPair(use_checkpoint=False).to(device)
    blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
    model.load_state_dict(state)
    print(f"loaded {ckpt_path} on {device}", flush=True)
    return visualize_scale_correspondence(model, he_b, he_a, CORR_DIR, device)


class nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *args):
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=NUM_STEPS)
    ap.add_argument(
        "--mode",
        type=str,
        default="overfit",
        choices=("overfit", "full", "precompute_global"),
        help="overfit: fixed pair; full: HER2 train; precompute_global: Experiment A labels",
    )
    ap.add_argument("--epochs", type=int, default=NUM_EPOCHS_FULL)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE_FULL)
    ap.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Must be cuda. CPU is refused.",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Recompute global-pos tiles even if .pt already exists",
    )
    ap.add_argument(
        "--store-k",
        type=int,
        default=None,
        help="Top-K to store in precompute_global (default: MATCHFORMER_GLOBAL_POS_STORE_K or 16)",
    )
    ap.add_argument(
        "--n-pos",
        type=int,
        default=None,
        help="InfoNCE positive-set size K (default: MATCHFORMER_N_POS or 4)",
    )
    ap.add_argument(
        "--viz-only",
        action="store_true",
        help="Load matchformer_pair_step0400.pt and write corr_scales/ PNGs",
    )
    ap.add_argument("--ckpt", type=str, default="", help="Override ckpt path for --viz-only")
    args = ap.parse_args()
    if args.viz_only:
        ckpt = Path(args.ckpt) if args.ckpt else CKPT_PATH
        run_viz(ckpt)
    elif args.mode == "precompute_global":
        precompute_global_positives(args)
    elif args.mode == "full":
        train_full(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
