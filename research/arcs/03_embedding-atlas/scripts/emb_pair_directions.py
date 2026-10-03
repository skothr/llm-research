"""Derive pair-difference direction consistency from the battery rows.

Model-free derivation. For each aligned pair kind (plural, case, valence,
register, xlat, gender, antonym, past, capital_of, lang_of) plus the
mechanically generated leading-space/bare twins ("space_twin"):

    delta_i = E[b_i] - E[a_i]                  (difference vector per pair)
    d_bar   = normalize(mean_i delta_i)        (the kind's mean direction)
    consistency = mean_i cos(delta_i, d_bar)

High consistency means the relation is encoded as a (near-)shared linear
direction in the embedding table — a candidate "handle". Global mean-centering
cancels in differences ((b-mu)-(a-mu) = b-a), so there is only one space here.

A permutation baseline re-pairs the b-side randomly within the kind
(N_SHUFFLE times, fixed seed) — the consistency a kind would show by chance
given the same word population.

Inputs : emb_battery_vectors.pt
Output : emb_pair_directions.pt (class: derived)
"""

from __future__ import annotations

from typing import Any

import torch

from _emb_artifacts import load_artifact, warn_if_mixed_sources, write_artifact

N_SHUFFLE = 200
SEED = 20260610


def kind_stats(deltas: torch.Tensor, g: torch.Generator) -> dict[str, Any]:
    """Consistency + pairwise-cosine stats for one kind's (n, d) deltas."""
    d_bar = deltas.mean(dim=0)
    d_bar = d_bar / d_bar.norm().clamp_min(1e-12)
    dn = deltas / deltas.norm(dim=1, keepdim=True).clamp_min(1e-12)
    cos_to_bar = dn @ d_bar
    pair_cos = dn @ dn.T
    n = deltas.shape[0]
    iu = torch.triu_indices(n, n, offset=1)
    return {
        "n": n,
        "d_bar": d_bar,
        "consistency": float(cos_to_bar.mean()),
        "cos_to_bar": cos_to_bar,
        "pairwise_cos_mean": float(pair_cos[iu[0], iu[1]].mean()),
        "delta_norm_mean": float(deltas.norm(dim=1).mean()),
    }


def main() -> None:
    warn_if_mixed_sources(["emb_battery_vectors.pt"])
    battery = load_artifact("emb_battery_vectors.pt")
    rows: list[dict[str, Any]] = battery["rows"]
    E = battery["E"].to(torch.float32)

    # primary row index per word (one per surviving word)
    primary_of: dict[str, int] = {
        r["word"]: i for i, r in enumerate(rows) if r["is_primary"]
    }
    # mechanical space/bare twins: words with both variants present
    by_word_variant: dict[tuple[str, str], int] = {
        (r["word"], r["variant"]): i for i, r in enumerate(rows)
    }
    twin_deltas: list[torch.Tensor] = []
    twin_words: list[str] = []
    for (word, variant), i in by_word_variant.items():
        if variant != "space":
            continue
        j = by_word_variant.get((word, "bare"))
        if j is not None:
            twin_deltas.append(E[i] - E[j])  # space-form minus bare-form
            twin_words.append(word)

    g = torch.Generator().manual_seed(SEED)
    out: dict[str, Any] = {
        "kinds": {},
        "inputs": ["emb_battery_vectors.pt"],
        "seed": SEED,
        "n_shuffle": N_SHUFFLE,
        "base_id": battery["base_id"],
        "revision": battery["revision"],
    }

    # group the curated PAIRS by kind, keeping only intact pairs
    by_kind: dict[str, list[tuple[str, str]]] = {}
    skipped: dict[str, list[tuple[str, str]]] = {}
    for kind, a, b in battery["pairs"]:
        if a in primary_of and b in primary_of:
            by_kind.setdefault(kind, []).append((a, b))
        else:
            skipped.setdefault(kind, []).append((a, b))

    all_kinds: list[tuple[str, torch.Tensor, list[str]]] = [
        ("space_twin", torch.stack(twin_deltas), twin_words)
    ]
    for kind, pairs in sorted(by_kind.items()):
        a_idx = torch.tensor([primary_of[a] for a, _ in pairs], dtype=torch.long)
        b_idx = torch.tensor([primary_of[b] for _, b in pairs], dtype=torch.long)
        deltas = E[b_idx] - E[a_idx]
        all_kinds.append((kind, deltas, [f"{a}->{b}" for a, b in pairs]))

    for kind, deltas, labels in all_kinds:
        stats = kind_stats(deltas, g)
        # permutation baseline: shuffle the b-side pairing
        n = deltas.shape[0]
        if kind == "space_twin":
            # rebuild from row pairs is overkill; permute deltas' b-side via
            # re-pairing delta endpoints is not well-defined here — shuffle by
            # pairing each space-form with a random *other* word's bare form
            a_rows = torch.stack([E[by_word_variant[(w, "space")]] for w in labels])
            b_rows = torch.stack([E[by_word_variant[(w, "bare")]] for w in labels])
        else:
            pairs = by_kind[kind]
            a_rows = E[torch.tensor([primary_of[a] for a, _ in pairs])]
            b_rows = E[torch.tensor([primary_of[b] for _, b in pairs])]
        shuf_cons = torch.empty(N_SHUFFLE)
        for s in range(N_SHUFFLE):
            perm = torch.randperm(n, generator=g)
            sd = b_rows[perm] - a_rows
            db = sd.mean(dim=0)
            db = db / db.norm().clamp_min(1e-12)
            sdn = sd / sd.norm(dim=1, keepdim=True).clamp_min(1e-12)
            shuf_cons[s] = (sdn @ db).mean()
        stats["labels"] = labels
        stats["shuffle_consistency_mean"] = float(shuf_cons.mean())
        stats["shuffle_consistency_std"] = float(shuf_cons.std())
        stats["skipped_pairs"] = skipped.get(kind, [])
        out["kinds"][kind] = stats
        print(
            f"{kind:12s} n={stats['n']:3d} consistency={stats['consistency']:+.4f} "
            f"shuffle={stats['shuffle_consistency_mean']:+.4f}"
            f"±{stats['shuffle_consistency_std']:.4f} "
            f"|delta|={stats['delta_norm_mean']:.3f}"
        )

    path = write_artifact("emb_pair_directions.pt")
    torch.save(out, path)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
