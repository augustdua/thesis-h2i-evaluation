"""Image DAB-L1 used for saved translator outputs.

StainDeconvDAB and the tissue mask are copied from
_scratch/compute_thesis_diag_a5_vib.py (Ruifrok H and DAB, two rows).

Exemplar patch error uses a different function, dab_hed_ch2 in
exemplar/matchformer_pair.py (full 3-stain HED inverse, channel 2).
This module imports that function. It does not redefine it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

_EX = Path(__file__).resolve().parents[1] / "exemplar"
if str(_EX) not in sys.path:
    sys.path.insert(0, str(_EX))

from matchformer_pair import dab_hed_ch2  # noqa: E402

BF_THR = 0.3
TISSUE_OD = 0.15


class StainDeconvDAB(nn.Module):
    _W_RUIFROK = ((0.650, 0.704, 0.286), (0.268, 0.570, 0.776))

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        W = torch.tensor(self._W_RUIFROK, dtype=torch.float32)
        W = W / W.norm(dim=1, keepdim=True)
        self.register_buffer("W_pinv", torch.linalg.pinv(W))
        self.eps = eps

    def forward(self, rgb_neg1_pos1: torch.Tensor) -> torch.Tensor:
        rgb01 = (rgb_neg1_pos1 + 1.0) * 0.5
        od = -torch.log(rgb01 + self.eps)
        c = torch.einsum("bchw,cs->bshw", od, self.W_pinv).clamp(min=0)
        return c[:, 1]


def load_rgb01(path: Path, tile: int = 256) -> torch.Tensor:
    im = Image.open(path).convert("RGB").resize((tile, tile), Image.BICUBIC)
    t = torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0)
    return t.permute(2, 0, 1)


def tissue_mask_he(he01: torch.Tensor) -> torch.Tensor:
    """he01: [3,H,W] in [0,1] -> [H,W] bool."""
    od = -torch.log(he01.clamp(min=0) + 1e-6)
    return od.mean(0) > TISSUE_OD


def score_dir(
    gen_dir: Path,
    data_root: Path,
    split: str,
    dab: StainDeconvDAB,
    max_n: int | None = None,
) -> dict:
    if gen_dir is None or not Path(gen_dir).is_dir():
        return {"status": "not_computed", "reason": f"gen dir missing: {gen_dir}"}
    gen_dir = Path(gen_dir)
    gens = {p.stem: p for p in gen_dir.glob("*.png")}
    if not gens:
        gens = {p.stem: p for p in gen_dir.glob("*.jpg")}
    stems = sorted(gens)
    if max_n is not None:
        stems = stems[:max_n]
    if not stems:
        return {"status": "not_computed", "reason": f"no images in {gen_dir}"}
    tissue_l1 = []
    dabpos_l1 = []
    for s in stems:
        gp = gens[s]
        real = data_root / "IHC" / split / f"{s}.jpg"
        he_p = data_root / "HE" / split / f"{s}.jpg"
        if not real.is_file() or not he_p.is_file():
            continue
        g01 = load_rgb01(gp)
        t01 = load_rgb01(real)
        h01 = load_rgb01(he_p)
        g = g01 * 2 - 1
        t = t01 * 2 - 1
        dg = dab(g.unsqueeze(0))[0]
        dt = dab(t.unsqueeze(0))[0]
        m = tissue_mask_he(h01)
        if m.any():
            tissue_l1.append(float((dg[m] - dt[m]).abs().mean().item()))
        pos = dt > BF_THR
        if pos.any():
            dabpos_l1.append(float((dg[pos] - dt[pos]).abs().mean().item()))
    if not tissue_l1:
        return {
            "status": "not_computed",
            "reason": "no overlapping stems with HE/IHC",
            "gen_dir": str(gen_dir),
        }
    return {
        "status": "computed",
        "n": len(tissue_l1),
        "n_dab_pos_tiles": len(dabpos_l1),
        "gen_dir": str(gen_dir),
        "dab_l1_tissue_mask": float(np.mean(tissue_l1)),
        "dab_l1_dab_positive": float(np.mean(dabpos_l1)) if dabpos_l1 else None,
        "BF_THR": BF_THR,
        "TISSUE_OD": TISSUE_OD,
    }
