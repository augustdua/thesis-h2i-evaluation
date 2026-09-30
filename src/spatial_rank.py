"""Spatial effective rank. Copied from _scratch/eval_c9_spatial_reff.py."""
from __future__ import annotations

import numpy as np
import torch


def spatial_reff(z: torch.Tensor) -> float:
    """z: [C,H,W] float."""
    C, H, W = z.shape
    X = z.reshape(C, H * W).T.float().cpu().numpy()  # HW x C
    Xc = X - X.mean(axis=0, keepdims=True)
    hw = Xc.shape[0]
    if hw < 2:
        return float("nan")
    cov = (Xc.T @ Xc) / float(hw - 1)
    evals = np.linalg.eigvalsh(cov)
    evals = np.clip(evals[::-1], 0.0, None)
    s = float(evals.sum())
    if s <= 0:
        return 0.0
    p = evals / s
    pos = p[p > 0]
    return float(np.exp(-(pos * np.log(pos)).sum()))
