"""Score saved VIB images with the same DAB-L1 as the other translators.

No VIB weights are in this repository. The chapter's working notes did not
keep full validation or test image sets for the spatial VIB grids (32, 64, 256),
for VIB 128 at beta 1e-4 beyond a 240-tile validation subset, or for the other
beta values. The frequency study can still score the 240-tile subset when
gen_dirs['vib_c64_b1e-4'] is set. A step-15500 full_vib_c9_z64 checkpoint was
used for representation checks; it has to be downloaded separately, and encode
goes through bbdm_root (see direct_translators/c9_loader.py).

  python -u vib/score_vib.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config, resolve  # noqa: E402
from src.dab import StainDeconvDAB, score_dir  # noqa: E402

NOT_RETAINED = [
    "VIB spatial grids 32, 64, and 256: full validation and test images were not kept locally.",
    "VIB 128, beta 1e-4: full validation and test generated images were not kept. A 240-tile validation subset was used for the frequency table (cell vib_c64_b1e-4).",
    "Other VIB beta values: full image sets and released weight files are not in this repository.",
    "full_vib_c9_z64 step 15500 was the checkpoint the fused-panel script looked for. It is not committed here.",
]


def main() -> None:
    cfg = load_config(ROOT)
    her2 = cfg.get("her2_root") or ""
    if not her2:
        raise SystemExit("Set her2_root in configs/chapter4.json")
    dab = StainDeconvDAB()
    data = Path(resolve(ROOT, her2))
    rows = {}
    for name, raw in (cfg.get("gen_dirs") or {}).items():
        if "vib" not in name.lower():
            continue
        if not raw:
            rows[name] = {"status": "not_computed", "reason": "gen dir not set"}
            continue
        rows[name] = score_dir(Path(resolve(ROOT, raw)), data, "val", dab)
    out = ROOT / "outputs" / "vib_dab_l1.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "not_retained": NOT_RETAINED,
        "rows": rows,
        "note": "Same StainDeconvDAB, tissue OD 0.15, and DAB threshold 0.3 as direct_translators/dab_l1.py.",
    }
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    print(f"WROTE {out}", flush=True)


if __name__ == "__main__":
    main()
