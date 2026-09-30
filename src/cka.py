"""Linear CKA. Copied from _scratch/eval_c9_fused_panel.py (same text as compute_thesis_diag_a12_a4.py)."""
from __future__ import annotations

import numpy as np


def center_cols(A: np.ndarray) -> np.ndarray:
    return A - A.mean(axis=0, keepdims=True)


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    Xc = center_cols(X)
    Yc = center_cols(Y)
    ytx = Yc.T @ Xc
    num = float(np.sum(ytx * ytx))
    den = float(np.linalg.norm(Xc.T @ Xc, "fro") * np.linalg.norm(Yc.T @ Yc, "fro"))
    return num / den if den > 0 else float("nan")
