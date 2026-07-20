"""Promote a reduced layer subset of each fitted J-lens to git-LFS.

Per the signed-off design plan (Decision 4,
`research/arcs/jspace/plans/2026-07-18-jspace-design.md`): the FULL fitted lens
tensor set (27 layers x d_model^2 fp16, ~127 MB at 1.5B / ~694 MB at 7B) stays
cache-only under `data/cache/` (gitignored, regenerable via
`examples/jspace_fit_lens.py`), while a reduced layer subset is committed via
git-LFS under `research/arcs/jspace/data/` so a clean clone can inspect
representative depths without the multi-GB fit.

**Layer-subset adjustment.** Decision 4 names layers {0,5,10,15,20,25,27}. The
fitted lenses have `source_layers` = 0..26 (27 layers; index 27 does not
exist), so this script uses **{0,5,10,15,20,25,26}** — the plan's cadence with
the trailing 27 clamped to the last valid layer index (26). Every other layer
is verbatim from the plan; this is the only deviation.

For each cached lens it loads the native dict
(`{"J": {layer:int -> (d,d) fp16}, "n_prompts", "source_layers", "d_model"}`),
restricts `J`/`source_layers` to the subset, adds `"subset_of"` (full lens
filename) and `"subset_layers"` provenance keys, and writes
`data/jlens_<stem>_layer-subset.pt` plus a `.config.json` sidecar (a copy of
the cache sidecar with `"subset_layers"` and `"full_set_location"` added).

Modes:
    python examples/jspace_promote_lens_subset.py           # write subsets (idempotent)
    python examples/jspace_promote_lens_subset.py --check    # verify vs cache, no write
    python examples/jspace_promote_lens_subset.py --stems jlens_qwen2.5-7b_nf4_n100

`--check` reloads each on-disk subset and asserts every subset J_l is
`torch.allclose` with the corresponding cache-source layer (and that the
subset's metadata matches), exiting 1 on any drift without writing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
ARC_DATA = _REPO_ROOT / "research" / "arcs" / "jspace" / "data"
CACHE_DIR = ARC_DATA / "cache"

# Decision 4 layers, with the trailing 27 clamped to the last valid index (26).
SUBSET_LAYERS: list[int] = [0, 5, 10, 15, 20, 25, 26]

DEFAULT_STEMS: list[str] = [
    "jlens_qwen2.5-7b_nf4_n100",
    "jlens_qwen2.5-1.5b_bf16_n100",
]

FULL_SET_LOCATION = (
    "data/cache/ (gitignored; regenerate via examples/jspace_fit_lens.py)"
)


def _subset_dict(full: dict[str, Any], full_filename: str) -> dict[str, Any]:
    """Build the subset lens dict from a full lens dict, in the same format."""
    jfull: dict[int, torch.Tensor] = full["J"]
    source_layers: list[int] = list(full["source_layers"])
    missing = [layer for layer in SUBSET_LAYERS if layer not in source_layers]
    if missing:
        raise ValueError(
            f"{full_filename}: subset layers {missing} absent from "
            f"source_layers {source_layers}"
        )
    return {
        "J": {layer: jfull[layer] for layer in SUBSET_LAYERS},
        "n_prompts": full["n_prompts"],
        "source_layers": list(SUBSET_LAYERS),
        "d_model": full["d_model"],
        "subset_of": full_filename,
        "subset_layers": list(SUBSET_LAYERS),
    }


def _subset_sidecar(cache_sidecar: dict[str, Any]) -> dict[str, Any]:
    out = dict(cache_sidecar)
    out["subset_layers"] = list(SUBSET_LAYERS)
    out["full_set_location"] = FULL_SET_LOCATION
    return out


def promote_one(stem: str) -> int:
    src_pt = CACHE_DIR / f"{stem}.pt"
    src_cfg = CACHE_DIR / f"{stem}.config.json"
    if not src_pt.exists():
        print(f"FAIL: cache lens absent: {src_pt}")
        return 1
    if not src_cfg.exists():
        print(f"FAIL: cache sidecar absent: {src_cfg}")
        return 1
    full = torch.load(src_pt, weights_only=False)
    subset = _subset_dict(full, src_pt.name)
    sidecar = _subset_sidecar(json.loads(src_cfg.read_text()))

    out_pt = ARC_DATA / f"{stem}_layer-subset.pt"
    out_cfg = ARC_DATA / f"{stem}_layer-subset.config.json"
    torch.save(subset, out_pt)
    out_cfg.write_text(json.dumps(sidecar, indent=2) + "\n")
    size_mb = out_pt.stat().st_size / 1e6
    print(
        f"wrote {out_pt.relative_to(_REPO_ROOT)}  "
        f"(layers {SUBSET_LAYERS}, {size_mb:.1f} MB) + config sidecar"
    )
    return 0


def check_one(stem: str) -> int:
    src_pt = CACHE_DIR / f"{stem}.pt"
    out_pt = ARC_DATA / f"{stem}_layer-subset.pt"
    out_cfg = ARC_DATA / f"{stem}_layer-subset.config.json"
    problems: list[str] = []
    if not out_pt.exists():
        print(f"CHECK {stem}: FAIL — subset missing: {out_pt}")
        return 1
    if not out_cfg.exists():
        problems.append(f"subset sidecar missing: {out_cfg}")
    if not src_pt.exists():
        print(f"CHECK {stem}: FAIL — cache source absent, cannot verify: {src_pt}")
        return 1

    full = torch.load(src_pt, weights_only=False)
    subset = torch.load(out_pt, weights_only=False)
    if list(subset["source_layers"]) != list(SUBSET_LAYERS):
        problems.append(f"source_layers {subset['source_layers']} != {SUBSET_LAYERS}")
    if list(subset.get("subset_layers", [])) != list(SUBSET_LAYERS):
        problems.append(
            f"subset_layers {subset.get('subset_layers')} != {SUBSET_LAYERS}"
        )
    if subset.get("subset_of") != src_pt.name:
        problems.append(f"subset_of {subset.get('subset_of')!r} != {src_pt.name!r}")
    if int(subset["n_prompts"]) != int(full["n_prompts"]):
        problems.append("n_prompts drift vs cache source")
    if int(subset["d_model"]) != int(full["d_model"]):
        problems.append("d_model drift vs cache source")
    for layer in SUBSET_LAYERS:
        if not subset["J"][layer].allclose(full["J"][layer]):
            problems.append(f"J[{layer}] not allclose to cache source")

    if problems:
        print(f"CHECK {stem}: FAIL")
        for p in problems:
            print(f"  - {p}")
        return 1
    print(f"CHECK {stem}: OK  (subset J allclose cache source; metadata matches)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--stems",
        nargs="+",
        default=DEFAULT_STEMS,
        help="cache lens stems (without .pt) to promote/verify",
    )
    ap.add_argument(
        "--check",
        action="store_true",
        help="verify on-disk subsets match the cache source (allclose); no write",
    )
    args = ap.parse_args()

    rc = 0
    for stem in args.stems:
        rc |= check_one(stem) if args.check else promote_one(stem)
    return rc


if __name__ == "__main__":
    sys.exit(main())
