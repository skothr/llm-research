#!/usr/bin/env python3
"""Stage 3b of the jspace arc: intermediate-concept lens evaluation.

The stage-3 readout scan measured how well each lens *predicts the model's
final output* — a metric the Jacobian lens is explicitly NOT optimized for
([gurnee2026-workspace §2.4]: a tuned/affine lens "skips ahead" to the answer,
while the J-lens surfaces the *currently-held intermediate concept*). This
script runs the fair test of the earlier-layers claim: the rank of the
intermediate concept's token in each lens readout.

It consumes the companion repo's lens-quality eval sets
(``jacobian-lens/data/evaluations/lens-eval-<slug>.json``). Per that dir's
README, every item carries ``prompt`` + ``intermediates``; ``target`` (when
present) only *defines the readout position* and is not scored. For the target
evals here (multihop, association, multilingual, order-ops, typo) the scored
position is the FINAL prompt token — for multihop the token immediately
preceding ``target``; for association the closing period. (poetry uses the
line-1 newline token — not implemented; pass --evals only for final-token sets.)

Metric (README): a concept is a *hit* at threshold k if its min-over-layers
lens rank <= k. Here we additionally split the min over three layer bands
(early 0-8, mid 9-18, late 19-26) and report top-10 / top-50 hit rates per band
for both the Jacobian lens (use_jacobian=True) and the logit-lens baseline.

Concept-token resolution: each intermediate word is scored by the min rank over
its single-token forms among ``{word, " "+word}`` (the space-prefixed form is
typically the single meaningful token for a mid-text concept); if neither form
is single-token, the first token of ``" "+word`` is used and the item flagged.

Usage:
    python examples/jspace_lens_eval.py --evals multihop association
    python examples/jspace_lens_eval.py --evals multihop --n-items 40

Run CPU-only via CUDA_VISIBLE_DEVICES="" while a GPU job holds the device.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import time
from typing import Any, cast

import numpy as np
import torch
from torch import Tensor

BANDS: dict[str, tuple[int, int]] = {"early": (0, 8), "mid": (9, 18), "late": (19, 26)}
THRESHOLDS = (10, 50)
DEFAULT_EVAL_DIR = "/home/ai/ai-projects/jacobian-lens/data/evaluations"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--mode", default="bf16", choices=["nf4", "int8", "bf16", "fp16"])
    p.add_argument("--device", default="cpu", choices=["cuda", "cpu"])
    p.add_argument(
        "--lens",
        default="research/arcs/04_jspace/data/cache/jlens_qwen2.5-1.5b_bf16_n100.pt",
    )
    p.add_argument("--eval-dir", default=DEFAULT_EVAL_DIR)
    p.add_argument(
        "--evals",
        nargs="+",
        default=["multihop", "association"],
        help="Eval slugs (files lens-eval-<slug>.json). Final-token sets only.",
    )
    p.add_argument(
        "--n-items", type=int, default=0, help="Subsample items per eval (0 = all)."
    )
    p.add_argument("--max-seq-len", type=int, default=512)
    p.add_argument("--examples", type=int, default=3, help="Example items to print.")
    p.add_argument(
        "--out",
        default=None,
        help="Output .pt path. Default derives from the lens stem "
        "(lens_eval_<lensstem>.pt in the lens's dir), so runs on different "
        "lenses never clobber each other.",
    )
    return p.parse_args()


def token_rank(readout: Tensor, token_id: int) -> int:
    """0-based rank of ``token_id`` in ``readout`` (0 == top-1)."""
    return int((readout > readout[token_id]).sum().item())


def concept_token_ids(tok: Any, word: str) -> tuple[list[int], bool]:
    """Single-token forms of ``word`` among ``{word, " "+word}``.

    Returns ``(ids, exact)`` where ``exact`` is True if at least one form was a
    single token; when neither is, falls back to the first token of ``" "+word``
    and returns ``exact=False``.
    """
    cands: list[int] = []
    for form in (word, " " + word):
        ids = tok(form, add_special_tokens=False).input_ids
        if len(ids) == 1:
            cands.append(int(ids[0]))
    if cands:
        return sorted(set(cands)), True
    fallback = tok(" " + word, add_special_tokens=False).input_ids
    return [int(fallback[0])], False


def build_model(args: argparse.Namespace) -> Any:
    from llm_surgeon import surgery

    from jlens import from_hf
    from jlens.protocol import LensModel

    device_map: dict[str, int | str] = {"": 0} if args.device == "cuda" else {"": "cpu"}
    model, tok = surgery.load_model(args.model, mode=args.mode, device_map=device_map)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    # HFLensModel.layers is an nn.ModuleList (fails the protocol's invariant
    # Sequence attribute check); cast is the sanctioned narrowing here.
    return cast(LensModel, from_hf(model, tok))


def band_rates(ranks: np.ndarray, layers: list[int]) -> dict[str, dict[int, float]]:
    """``ranks`` is ``[n_instances, n_layers]`` (0-based ranks).

    Returns ``{band: {k: hit_rate}}`` where hit_rate is the fraction of
    instances whose min rank over the band's layers is < k.
    """
    out: dict[str, dict[int, float]] = {}
    for band, (lo, hi) in {**BANDS, "overall": (0, max(layers))}.items():
        cols = [i for i, l in enumerate(layers) if lo <= l <= hi]
        if not cols:
            out[band] = {k: float("nan") for k in THRESHOLDS}
            continue
        best = ranks[:, cols].min(axis=1)
        out[band] = {k: float((best < k).mean()) for k in THRESHOLDS}
    return out


def run_eval(
    slug: str, args: argparse.Namespace, m: Any, lens: Any, layers: list[int]
) -> dict[str, Any]:
    path = Path(args.eval_dir) / f"lens-eval-{slug}.json"
    items = json.loads(path.read_text(encoding="utf-8"))["items"]
    if args.n_items > 0:
        items = items[: args.n_items]
    tok = m.tokenizer

    rank_j_rows: list[np.ndarray] = []
    rank_l_rows: list[np.ndarray] = []
    instances: list[dict[str, Any]] = []
    n_inexact = 0
    t0 = time()
    for it in items:
        prompt = it["prompt"]
        seq_len = int(m.encode(prompt, max_length=args.max_seq_len).shape[1])
        pos = [seq_len - 1]  # final prompt token
        ll_j, _, _ = lens.apply(
            m,
            prompt,
            layers=layers,
            positions=pos,
            max_seq_len=args.max_seq_len,
            use_jacobian=True,
        )
        ll_l, _, _ = lens.apply(
            m,
            prompt,
            layers=layers,
            positions=pos,
            max_seq_len=args.max_seq_len,
            use_jacobian=False,
        )
        for word in it["intermediates"]:
            ids, exact = concept_token_ids(tok, word)
            if not exact:
                n_inexact += 1
            rj = np.array([min(token_rank(ll_j[l][0], i) for i in ids) for l in layers])
            rl = np.array([min(token_rank(ll_l[l][0], i) for i in ids) for l in layers])
            rank_j_rows.append(rj)
            rank_l_rows.append(rl)
            best_layer_j = int(layers[int(rj.argmin())])
            best_layer_l = int(layers[int(rl.argmin())])
            # Top-10 decoded readout strings at each lens's best layer (additive).
            topk_strs_j = [
                tok.decode([int(t)]) for t in ll_j[best_layer_j][0].topk(10).indices
            ]
            topk_strs_l = [
                tok.decode([int(t)]) for t in ll_l[best_layer_l][0].topk(10).indices
            ]
            instances.append(
                {
                    "name": it.get("name"),
                    "word": word,
                    "token_ids": ids,
                    "exact_single_token": exact,
                    "best_layer_j": best_layer_j,
                    "best_rank_j": int(rj.min()),
                    "best_layer_l": best_layer_l,
                    "best_rank_l": int(rl.min()),
                    "best_layer_topk_strs_j": topk_strs_j,
                    "best_layer_topk_strs_l": topk_strs_l,
                }
            )
    ranks_j = np.stack(rank_j_rows)
    ranks_l = np.stack(rank_l_rows)
    return {
        "slug": slug,
        "n_items": len(items),
        "n_instances": len(instances),
        "n_inexact_token": n_inexact,
        "rates_j": band_rates(ranks_j, layers),
        "rates_l": band_rates(ranks_l, layers),
        "ranks_j": ranks_j,
        "ranks_l": ranks_l,
        "instances": instances,
        "seconds": time() - t0,
    }


def print_eval_table(res: dict[str, Any]) -> None:
    print(
        f"\n=== {res['slug']}  (items={res['n_items']}, "
        f"instances={res['n_instances']}, inexact-token={res['n_inexact_token']}, "
        f"{res['seconds']:.1f}s) ==="
    )
    print(
        f"{'band':<8} {'J@10':>7} {'J@50':>7} {'logit@10':>9} {'logit@50':>9} "
        f"{'ΔJ-L@10':>8} {'ΔJ-L@50':>8}"
    )
    for band in ("early", "mid", "late", "overall"):
        j = res["rates_j"][band]
        ll = res["rates_l"][band]
        print(
            f"{band:<8} {j[10]:>7.2f} {j[50]:>7.2f} {ll[10]:>9.2f} {ll[50]:>9.2f} "
            f"{j[10] - ll[10]:>8.2f} {j[50] - ll[50]:>8.2f}"
        )


def print_examples(res: dict[str, Any], tok: Any, n: int) -> None:
    print(f"  examples ({res['slug']}): best (layer, rank) per lens")
    for inst in res["instances"][:n]:
        jt = tok.decode([inst["token_ids"][0]])
        print(
            f"    {inst['name']!r:<22} concept={inst['word']!r} tok={jt!r} | "
            f"J: L{inst['best_layer_j']} rank {inst['best_rank_j']}  | "
            f"logit: L{inst['best_layer_l']} rank {inst['best_rank_l']}"
        )


def main() -> None:
    args = parse_args()
    from jlens import JacobianLens

    t_load = time()
    m = build_model(args)
    lens = JacobianLens.load(args.lens)
    layers = list(lens.source_layers)
    print(f"[load] model + lens in {time() - t_load:.1f}s; {lens!r}")
    print(f"[bands] early={BANDS['early']} mid={BANDS['mid']} late={BANDS['late']}")

    results: dict[str, Any] = {}
    for slug in args.evals:
        res = run_eval(slug, args, m, lens, layers)
        results[slug] = res
        print_eval_table(res)
        print_examples(res, m.tokenizer, args.examples)

    summary = {
        "model": args.model,
        "mode": args.mode,
        "lens": args.lens,
        "layers": layers,
        "bands": BANDS,
        "thresholds": list(THRESHOLDS),
        "per_eval": {
            slug: {
                "n_items": r["n_items"],
                "n_instances": r["n_instances"],
                "n_inexact_token": r["n_inexact_token"],
                "rates_j": r["rates_j"],
                "rates_l": r["rates_l"],
            }
            for slug, r in results.items()
        },
    }
    lens_stem = Path(args.lens).stem
    out = (
        Path(args.out)
        if args.out
        else Path(args.lens).with_name(
            f"lens_eval_{lens_stem.replace('jlens_', '')}.pt"
        )
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"summary": summary, "per_eval": results}, out)
    print(f"\n[save] {out}")


if __name__ == "__main__":
    main()
