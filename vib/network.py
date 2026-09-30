"""VIB network entry point. No weights.

The variational bottleneck is C9Paired(vib=True) together with GaussianBottleneckHead.
Those classes are defined once in direct_translators/c9_paired.py and imported here.
"""
from __future__ import annotations

import sys
from pathlib import Path

_DT = Path(__file__).resolve().parents[1] / "direct_translators"
if str(_DT) not in sys.path:
    sys.path.insert(0, str(_DT))

from c9_paired import C9Paired, GaussianBottleneckHead  # noqa: E402


def build_vib(uni_module, vib_z_ch: int = 64, z_hw: int = 128, **kwargs) -> C9Paired:
    """Construct the HE-to-IHC VIB translator. uni_module is a frozen UNI trunk."""
    return C9Paired(
        uni_module=uni_module,
        vib=True,
        vib_z_ch=vib_z_ch,
        z_hw=z_hw,
        **kwargs,
    )


__all__ = ["C9Paired", "GaussianBottleneckHead", "build_vib"]
