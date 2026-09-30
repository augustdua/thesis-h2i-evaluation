"""Load configs/chapter4.json. No tokens."""
from __future__ import annotations

import json
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]


def load_config(root: Path | None = None) -> dict:
    root = PKG if root is None else Path(root)
    return json.loads((root / "configs" / "chapter4.json").read_text(encoding="utf-8"))


def resolve(root: Path, raw: str) -> str:
    if not raw:
        return ""
    path = Path(raw)
    if not path.is_absolute():
        path = Path(root) / path
    return str(path)
