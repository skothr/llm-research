"""S4 branch: characterize the entangled 21-dim block found by the sweep.

The full-vocabulary sweep (emb_fullvocab_analyze.py) found exactly one
connected component of dimensions correlated at |r| > 0.3 — 21 dims whose
extreme tokens are high-frequency structural tokens across scripts (',' and
'，', ' the' and '的'). Hypotheses to test against the full population:

  H-A (frequency): block energy tracks token frequency. Proxy: BPE token id
      is merge order — lower id ~ more frequent. Test: Spearman correlation
      between block-energy fraction and token id, vs a seeded random
      21-dim control set.
  H-B (language-independence): the block carries the same kind of signal for
      ASCII, CJK, and Cyrillic tokens — compare block-energy distributions
      per script class, again vs control dims.

Inputs : emb_fullvocab_analysis.pt (block dims), cached W_E, tokenizer.
Output : emb_structural_block.pt (class: derived) + console report.
"""

from __future__ import annotations

from typing import Any

import torch

from _emb_artifacts import load_artifact, read_artifact, write_artifact
from emb_capture import BASE_ID, REVISION
from emb_fullvocab_stats import WE_CACHE

SEED = 20260610
N_EXTREME = 30


def spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    ra = a.argsort().argsort().float()
    rb = b.argsort().argsort().float()
    ra = (ra - ra.mean()) / ra.std()
    rb = (rb - rb.mean()) / rb.std()
    return float((ra * rb).mean())


def script_class(s: str) -> str:
    if any("一" <= ch <= "鿿" for ch in s):
        return "cjk"
    if any("Ѐ" <= ch <= "ӿ" for ch in s):
        return "cyrillic"
    if s.strip() and all(ord(ch) < 128 for ch in s):
        return "ascii"
    return "other"


def main() -> None:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(BASE_ID, revision=REVISION)
    an = load_artifact("emb_fullvocab_analysis.pt")
    fv = load_artifact("emb_fullvocab_stats.pt")
    blocks = sorted(an["corr_blocks"], key=lambda b: -b["size"])
    block_dims = torch.tensor(blocks[0]["dims"], dtype=torch.long)
    nb = int(block_dims.numel())
    print(f"block dims ({nb}): {block_dims.tolist()}")

    row_ids: torch.Tensor = fv["row_ids"].long()
    we = torch.load(read_artifact(WE_CACHE), weights_only=False)
    X = we["W"][:151_665].to(torch.float32)[row_ids]
    n, d = X.shape

    g = torch.Generator().manual_seed(SEED)
    pool = torch.tensor(
        [i for i in range(d) if i not in set(block_dims.tolist())], dtype=torch.long
    )
    control_dims = pool[torch.randperm(pool.numel(), generator=g)[:nb]]

    norms = X.norm(dim=1).clamp_min(1e-12)
    out: dict[str, Any] = {
        "block_dims": block_dims,
        "control_dims": control_dims,
        "inputs": ["emb_fullvocab_analysis.pt", "emb_fullvocab_stats.pt"],
        "seed": SEED,
        "base_id": we["base_id"],
        "revision": we["revision"],
    }

    # H-A: energy fraction vs token id (frequency proxy)
    for name, dims in (("block", block_dims), ("control", control_dims)):
        frac = X[:, dims].norm(dim=1) / norms
        rho = spearman(frac, row_ids.float())
        out[f"{name}_energy_frac"] = frac.to(torch.float16)
        out[f"{name}_spearman_vs_id"] = rho
        # decile curve: mean energy fraction per token-id decile
        dec_edges = torch.quantile(row_ids.float(), torch.linspace(0, 1, 11))
        dec_means = []
        for i in range(10):
            m = (row_ids.float() >= dec_edges[i]) & (
                row_ids.float() <= dec_edges[i + 1]
            )
            dec_means.append(float(frac[m].mean()))
        out[f"{name}_decile_means"] = dec_means
        print(
            f"{name:8s} spearman(energy, token_id) = {rho:+.4f}; "
            f"decile means {['%.4f' % v for v in dec_means]}"
        )

    # H-B: per-script distributions
    strings = [tok.decode([int(i)]) for i in row_ids]
    classes = [script_class(s) for s in strings]
    frac_block: torch.Tensor = out["block_energy_frac"].float()
    frac_ctrl: torch.Tensor = out["control_energy_frac"].float()
    per_script: dict[str, dict[str, float]] = {}
    for sc in ("ascii", "cjk", "cyrillic", "other"):
        m = torch.tensor([c == sc for c in classes])
        if int(m.sum()) < 50:
            continue
        per_script[sc] = {
            "n": int(m.sum()),
            "block_mean": float(frac_block[m].mean()),
            "block_q90": float(frac_block[m].quantile(0.9)),
            "control_mean": float(frac_ctrl[m].mean()),
        }
        print(
            f"script {sc:9s} n={per_script[sc]['n']:6d} "
            f"block mean {per_script[sc]['block_mean']:.4f} "
            f"q90 {per_script[sc]['block_q90']:.4f} "
            f"control mean {per_script[sc]['control_mean']:.4f}"
        )
    out["per_script"] = per_script

    # decoded extremes
    top = torch.topk(frac_block, N_EXTREME)
    bot = torch.topk(-frac_block, N_EXTREME)
    out["top_tokens"] = [strings[int(i)] for i in top.indices]
    out["top_token_ids"] = row_ids[top.indices].tolist()
    out["top_fracs"] = top.values.tolist()
    out["bottom_tokens"] = [strings[int(i)] for i in bot.indices]
    print(
        "top block-energy tokens:", ", ".join(repr(s) for s in out["top_tokens"][:15])
    )
    print(
        "bottom block-energy tokens:",
        ", ".join(repr(s) for s in out["bottom_tokens"][:15]),
    )

    path = write_artifact("emb_structural_block.pt")
    torch.save(out, path)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
