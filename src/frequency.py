"""Wavelet energy, half-Nyquist power, and the numpy DAB used with them.

Copied from _scratch/ch4_figs/hf_study.py. This is the DAB inside the frequency
study (Tables 4.7 and 4.9). It is not dab_hed_ch2 and not StainDeconvDAB.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pywt
from PIL import Image

LEVELS = 4
RUIFROK = np.array([[0.650, 0.704, 0.286], [0.268, 0.570, 0.776], [0.0, 0.0, 0.0]])


def dab_matrix() -> np.ndarray:
    w = RUIFROK[:2] / np.linalg.norm(RUIFROK[:2], axis=1, keepdims=True)
    return np.linalg.pinv(w)


W_PINV = dab_matrix()


def dab(rgb01: np.ndarray) -> np.ndarray:
    od = -np.log(np.clip(rgb01, 0, 1) + 1e-6)
    return np.clip(od @ W_PINV, 0, None)[..., 1]


def load(path: Path) -> np.ndarray:
    im = Image.open(path).convert("RGB")
    if im.size != (256, 256):
        im = im.resize((256, 256), Image.BICUBIC)
    return np.asarray(im, dtype=np.float64) / 255.0


def luma(rgb: np.ndarray) -> np.ndarray:
    return rgb @ np.array([0.299, 0.587, 0.114])


def band_fractions(y: np.ndarray) -> np.ndarray:
    y = y - y.mean()
    coeffs = pywt.wavedec2(y, "db4", mode="periodization", level=LEVELS)
    tot = float((y ** 2).sum()) + 1e-12
    # coeffs[1] is the coarsest detail level (LEVELS), coeffs[-1] the finest (level 1)
    e = [sum(float((d ** 2).sum()) for d in coeffs[-k]) / tot for k in range(1, LEVELS + 1)]
    return np.array(e)


def i_hf(y: np.ndarray) -> float:
    y = y - y.mean()
    p = np.abs(np.fft.fftshift(np.fft.fft2(y))) ** 2
    n = y.shape[0]
    f = np.fft.fftshift(np.fft.fftfreq(n))
    fx, fy = np.meshgrid(f, f)
    r = np.sqrt(fx ** 2 + fy ** 2) / 0.5
    keep = r > 0
    return float(p[keep & (r > 0.5)].sum() / p[keep].sum())


def lowpass_rgb(rgb: np.ndarray, k: int) -> np.ndarray:
    out = np.empty_like(rgb)
    for c in range(3):
        co = pywt.wavedec2(rgb[..., c], "db4", mode="periodization", level=LEVELS)
        for j in range(1, k + 1):
            co[-j] = tuple(np.zeros_like(d) for d in co[-j])
        out[..., c] = pywt.waverec2(co, "db4", mode="periodization")
    return np.clip(out, 0, 1)


def summarise(rows: list[dict]) -> dict:
    b = np.array([r["bands"] for r in rows])
    h = np.array([r["i_hf"] for r in rows])
    return {
        "n": len(rows),
        "band_frac_mean": b.mean(0).tolist(),
        "i_hf_mean": float(h.mean()),
        "i_hf_std": float(h.std()),
    }
