"""Cross-script pairing check (post-hoc, 2026-07-21): is ' the'/'的' a
similarity pair or only a shared-block pair?

Finding #6 (README) lists cross-script pairs for the 21-dim entangled block.
This check quantifies the '的' side: its nearest-cosine English token, and
how the ordering shifts when the block dims are ablated (distinguishing
shared glue-block loading from distributed/semantic alignment).

Candidate token ids are NOT taken from a tokenizer at run time — they are
constants cross-validated against ids already recorded in committed
artifacts (emb_structural_block.pt top_token_ids, emb_fullvocab_analysis.pt
handle_census), originally resolved via the Qwen2 BPE table (shared by
Qwen2.5; llama.cpp ggml-vocab-qwen2.gguf). The script asserts every
validatable binding before computing.

Inputs : cached emb_WE_bf16.pt (not committed; regenerate via the pinned
         HF revision), emb_structural_block.pt, emb_fullvocab_analysis.pt.
Output : emb_de_cosine_check.pt — the 11 candidate embedding rows plus
         derived cosines and '的''s top-20 full-vocab neighbors, so the
         audit re-derives the cited numbers from committed data alone.
"""

from __future__ import annotations

import torch

from _emb_artifacts import load_artifact, write_artifact

# label -> Qwen2.5 token id (see docstring for provenance)
CANDIDATES: dict[str, int] = {
    " the": 279,
    "的": 9370,
    " of": 315,
    "'s": 594,
    "’s": 748,
    " de": 409,
    " a": 264,
    " to": 311,
    ",": 11,
    "，": 3837,
    "。": 1773,
}
TOPK = 20


def main() -> None:
    sb = load_artifact("emb_structural_block.pt")
    census = load_artifact("emb_fullvocab_analysis.pt")["handle_census"]

    # cross-validate the id constants against committed artifacts
    block_bindings = dict(zip(sb["top_tokens"], (int(i) for i in sb["top_token_ids"])))
    prep = next(e for e in census if e["class"] == "preposition")
    census_bindings = dict(
        zip(prep["top_tokens"], (int(i) for i in prep["top_token_ids"]))
    )
    for label, tid in CANDIDATES.items():
        for src in (block_bindings, census_bindings):
            if label in src:
                assert src[label] == tid, (label, tid, src[label])
    print(
        f"id bindings validated against artifacts for "
        f"{sum(1 for l in CANDIDATES if l in block_bindings or l in census_bindings)}"
        f"/{len(CANDIDATES)} candidates"
    )

    we = load_artifact("emb_WE_bf16.pt")
    assert we["base_id"] == sb["base_id"] and we["revision"] == sb["revision"], (
        "W_E cache and structural-block artifact are from different captures"
    )
    WE = we["W"].float()
    block_dims = torch.as_tensor(sb["block_dims"], dtype=torch.long)
    V = torch.nn.functional.normalize(WE, dim=1)

    de_id = CANDIDATES["的"]
    sims = V @ V[de_id]
    top = torch.topk(sims, TOPK + 1)  # +1: self at rank 0

    mask = torch.ones(WE.shape[1], dtype=torch.bool)
    mask[block_dims] = False

    full_cos: dict[str, float] = {}
    ablated_cos: dict[str, float] = {}
    for label, tid in CANDIDATES.items():
        full_cos[label] = float(
            torch.nn.functional.cosine_similarity(WE[tid], WE[de_id], dim=0)
        )
        ablated_cos[label] = float(
            torch.nn.functional.cosine_similarity(WE[tid][mask], WE[de_id][mask], dim=0)
        )

    out = {
        "candidates": CANDIDATES,
        "rows_f16": WE[list(CANDIDATES.values())].to(torch.float16),
        "block_dims": block_dims,
        "full_cos_to_de": full_cos,
        "ablated_cos_to_de": ablated_cos,
        "topk_ids": top.indices[1:].tolist(),
        "topk_cos": top.values[1:].tolist(),
        "n_vocab_rows": int(WE.shape[0]),
        "base_id": sb["base_id"],
        "revision": sb["revision"],
    }

    print(f"cos to '的' (full | block-ablated), of {out['n_vocab_rows']} rows:")
    for label in CANDIDATES:
        print(f"  {label!r:8s} {full_cos[label]:+.4f} | {ablated_cos[label]:+.4f}")
    print(f"top-{TOPK} neighbor ids of '的':", out["topk_ids"])

    path = write_artifact("emb_de_cosine_check.pt")
    torch.save(out, path)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
