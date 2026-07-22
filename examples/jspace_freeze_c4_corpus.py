#!/usr/bin/env python3
"""Freeze a seeded C4-en fitting corpus for the jspace corpus-sensitivity check.

The primary jspace fitting corpus
(`research/arcs/jspace/data/fitting_prompts_wikitext103_n1000.json`) mirrors
the companion repo's `jlens.examples.load_wikitext_prompts` selection, but
wikitext-103 is English-Wikipedia register — narrower than Qwen2.5's
multilingual pretraining mix. This script freezes an alternative corpus drawn
from allenai/c4 (config `en`) so the 1.5B lens can be refit on a broader
web-text distribution and the stage-3/4 metric suite re-run to test corpus
sensitivity (README "Deferred / follow-up directions" item 2).

Selection (mirrors the wikitext freeze's shape, plus a seeded shuffle):
  1. Stream allenai/c4, config `en`, split `train` (streaming=True).
  2. `shuffle(seed=42, buffer_size=10_000)` on the streaming dataset — a
     buffered reservoir shuffle so the first 1000 kept records are not just
     the head of the shard order.
  3. Keep records with `len(text.strip()) >= 600`.
  4. Take the first 1000 in post-shuffle order.

Deterministic given the seed: `datasets` derives its shuffle RNG from the
fixed seed (42) and the fixed buffer_size (10_000) over an immutable public
dataset, so re-running reproduces the same 1000 records byte-for-byte. The
frozen JSON is committed so downstream fits never re-stream.

Usage:
    python examples/jspace_freeze_c4_corpus.py            # write the JSON
    python examples/jspace_freeze_c4_corpus.py --force    # overwrite existing

Network: streaming needs huggingface.co. If the sandbox blocks it, re-run
with the sandbox disabled (read-only public-dataset download).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ARC_DATA = Path("research/arcs/jspace/data")
OUT_PATH = ARC_DATA / "fitting_prompts_c4en_n1000.json"

DATASET = "allenai/c4"
CONFIG = "en"
SPLIT = "train"
SEED = 42
BUFFER_SIZE = 10_000
MIN_CHARS = 600
N_PROMPTS = 1000


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=OUT_PATH)
    p.add_argument("--n-prompts", type=int, default=N_PROMPTS)
    p.add_argument("--min-chars", type=int, default=MIN_CHARS)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--buffer-size", type=int, default=BUFFER_SIZE)
    p.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Skip the first N ACCEPTED (post-filter) records before collecting. "
        "The disjoint-held-out set uses --offset 1000 to take records 1001+ of "
        "the SAME seeded stream the n=1000 fitting corpus consumed 1-1000.",
    )
    p.add_argument(
        "--dedup-against",
        type=Path,
        default=None,
        help="A frozen corpus JSON ({'prompts': [...]}) whose texts are EXCLUDED "
        "from collection — belt-and-suspenders disjointness guarantee independent "
        "of streaming-shuffle byte-reproducibility across datasets versions.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing frozen corpus (default: refuse if present).",
    )
    return p.parse_args()


def freeze_c4_prompts(
    n_prompts: int,
    *,
    min_chars: int,
    seed: int,
    buffer_size: int,
    offset: int = 0,
    exclude: set[str] | None = None,
) -> list[str]:
    """Return ``n_prompts`` C4-en records of at least ``min_chars`` characters, in
    seeded-shuffle order, after skipping the first ``offset`` accepted records and
    any whose text is in ``exclude``. Deterministic given ``seed`` and
    ``buffer_size`` over the immutable allenai/c4 `en` train split — so
    ``offset=1000`` yields exactly the records the n=1000 fitting freeze skipped."""
    if n_prompts <= 0:
        return []
    from datasets import load_dataset

    exclude = exclude or set()
    dataset = load_dataset(DATASET, CONFIG, split=SPLIT, streaming=True)
    dataset = dataset.shuffle(seed=seed, buffer_size=buffer_size)
    prompts: list[str] = []
    n_accepted = 0  # accepted-by-filter count, for the offset skip
    for record in dataset:
        text = record["text"]
        if len(text.strip()) < min_chars:
            continue
        n_accepted += 1
        if n_accepted <= offset:
            continue  # part of the earlier (fitting) slice of the same stream
        if text in exclude:
            continue  # already in the dedup-against corpus (disjointness guard)
        prompts.append(text)
        if len(prompts) == n_prompts:
            break
    return prompts


def main() -> int:
    args = parse_args()
    if args.out.exists() and not args.force:
        print(f"REFUSE: {args.out} exists (pass --force to overwrite)")
        return 1

    exclude: set[str] = set()
    if args.dedup_against is not None:
        dd = json.loads(Path(args.dedup_against).read_text(encoding="utf-8"))
        exclude = set(dd["prompts"] if isinstance(dd, dict) else dd)
        print(f"[dedup] excluding {len(exclude)} texts from {args.dedup_against}")

    prompts = freeze_c4_prompts(
        args.n_prompts,
        min_chars=args.min_chars,
        seed=args.seed,
        buffer_size=args.buffer_size,
        offset=args.offset,
        exclude=exclude,
    )
    if len(prompts) < args.n_prompts:
        print(
            f"WARNING: only collected {len(prompts)} prompts, "
            f"requested {args.n_prompts}"
        )
    if exclude:
        overlap = len(set(prompts) & exclude)
        print(
            f"[dedup] overlap of collected set with dedup-against: {overlap} (want 0)"
        )

    selection = (
        f"shuffle(seed={args.seed}, buffer_size={args.buffer_size}), skip first "
        f"{args.offset} accepted, then first {args.n_prompts} records with "
        f"len(text.strip()) >= {args.min_chars}, in post-shuffle order"
    )
    doc = {
        # Flat top-level provenance, mirroring the wikitext freeze's shape.
        "source": f"{DATASET} {CONFIG} {SPLIT}",
        "selection": selection,
        # Structured meta block: dataset/config/split/seed/buffer_size/filter.
        "dataset": DATASET,
        "config": CONFIG,
        "split": SPLIT,
        "seed": args.seed,
        "buffer_size": args.buffer_size,
        "offset": args.offset,
        "dedup_against": (str(args.dedup_against) if args.dedup_against else None),
        "filter": f"len(text.strip()) >= {args.min_chars}",
        "n": len(prompts),
        "prompts": prompts,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=2))
    print(
        f"wrote {args.out}  ({len(prompts)} prompts, "
        f"{args.out.stat().st_size / 1e6:.2f} MB)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
