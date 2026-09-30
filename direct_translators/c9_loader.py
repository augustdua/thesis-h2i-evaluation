"""Import C9Paired from a local BBDM checkout.

bbdm_BCI/train_c9_paired.py is not in this repository. On import it reads a
Hugging Face token from a local file. Point bbdm_root at a checkout that
contains bbdm_BCI/train_c9_paired.py and bbdm_BCI/src_ds_her2_thunder.py.
"""
from __future__ import annotations

import sys
from pathlib import Path


def import_train_c9(bbdm_root: Path):
    bci = Path(bbdm_root) / "bbdm_BCI"
    model_py = bci / "train_c9_paired.py"
    helper_py = bci / "src_ds_her2_thunder.py"
    if not model_py.is_file() or not helper_py.is_file():
        raise SystemExit(
            "CKA, spatial effective rank, and VIB encode need C9Paired from "
            f"{model_py} and {helper_py}. Set bbdm_root in configs/chapter4.json. "
            "Those files are not vendored here because train_c9_paired.py reads a token file on import."
        )
    bci_s = str(bci)
    if bci_s not in sys.path:
        sys.path.insert(0, bci_s)
    import train_c9_paired

    return train_c9_paired
