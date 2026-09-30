"""Load the vendored C9 network.

direct_translators/c9_paired.py is the architecture. It does not read a token file.
bbdm_root is accepted and ignored so older call sites keep working.
"""
from __future__ import annotations

from pathlib import Path


def import_train_c9(bbdm_root: Path | None = None):
    import c9_paired

    return c9_paired
