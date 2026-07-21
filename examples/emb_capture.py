"""Capture script for the embedding-atlas arc — the ONE model-loading step.

Slices the battery rows out of Qwen2.5-7B-Instruct's input-embedding table
(W_E) and unembedding matrix (W_U = lm_head; untied in this model), computes
full-vocabulary global statistics, a random-row baseline, and nearest-neighbor
probes, then writes four artifacts to the working cache:

    emb_battery_vectors.pt   battery anchor-variant rows (E + U, bf16) + metadata
    emb_global_stats.pt      norms / mean / anisotropy / PCA spectrum / E-U cosine
    emb_random_baseline.pt   1000 random real-token rows (E + U) for baselines
    emb_neighbor_probes.pt   top-K cosine neighbors for ~24 probe tokens

The full W_E/W_U matrices are NOT persisted (1.09 GB each): the capture-root
of this arc is the published model weight matrix itself, pinned to REVISION —
see research/arcs/03_embedding-atlas/README.md for the stated deviation from
ARC_PROCESS § "Raw data is a deliverable".

Modes:
    python examples/emb_capture.py --tokenize-only   # tokenizer pre-flight:
        battery coverage report (single-token survivors / drops), no model load
    python examples/emb_capture.py                   # full capture

All reductions are computed in float32 after casting from bf16. Padded rows
(ids >= len(tokenizer)) are excluded from every global statistic.
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Any

from emb_token_battery import BATTERY, PAIRS

BASE_ID = "Qwen/Qwen2.5-7B-Instruct"
# Pinned snapshot (refs/main of the locally cached copy). The manifest records
# this; re-capture must use the same revision for bit-identical artifacts.
REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
SEED = 20260610
N_BASELINE = 1000
TOP_PCS = 50
NEIGHBOR_K = 15
PCA_CHUNK = 8192

# Probe tokens for nearest-neighbor inspection: (label, text) — text is
# tokenized with add_special_tokens=False and must be a single token.
NEIGHBOR_PROBES: tuple[tuple[str, str], ...] = (
    ("country_space", " France"),
    ("country_bare", "France"),
    ("capital_space", " Paris"),
    ("animal_space", " dog"),
    ("animal_plural", " dogs"),
    ("emotion_space", " happy"),
    ("emotion_neg", " fear"),
    ("color_space", " red"),
    ("number_word", " seven"),
    ("digit", "7"),
    ("code_kw", "def"),
    ("function_word", " the"),
    ("preposition", " of"),
    ("pronoun", " she"),
    ("formal", " purchase"),
    ("informal", " buy"),
    ("month", " May"),
    ("name", " Bill"),
    ("punct", "."),
    ("cjk_zh", "水"),
    ("cjk_country", "中国"),
    ("cyrillic", "вода"),
    ("profession", " doctor"),
    ("material", " gold"),
)


def resolve_battery(tok: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Resolve every battery word to its single-token variants.

    Returns (rows, drops). Each row: {class, supergroup, word, variant
    ("space"|"bare"), token_id, is_primary}. Primary = the leading-space form
    where it survives (the in-running-text BPE form for English), else bare.
    A word with no single-token variant is dropped (reported, never silent).
    """
    rows: list[dict[str, Any]] = []
    drops: list[dict[str, Any]] = []
    for cls_name, cls in BATTERY.items():
        for word in cls.words:
            candidates: list[tuple[str, str]] = []
            if not cls.bare_only:
                candidates.append(("space", " " + word))
            candidates.append(("bare", word))
            found: dict[str, int] = {}
            for variant, text in candidates:
                ids = tok.encode(text, add_special_tokens=False)
                if len(ids) == 1:
                    found[variant] = ids[0]
            if not found:
                drops.append({"class": cls_name, "word": word})
                continue
            primary_variant = "space" if "space" in found else "bare"
            for variant, token_id in found.items():
                rows.append(
                    {
                        "class": cls_name,
                        "supergroup": cls.supergroup,
                        "word": word,
                        "variant": variant,
                        "token_id": token_id,
                        "is_primary": variant == primary_variant,
                    }
                )
    return rows, drops


def coverage_report(rows: list[dict[str, Any]], drops: list[dict[str, Any]]) -> None:
    by_class_rows: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_class_rows.setdefault(r["class"], []).append(r)
    by_class_drops: dict[str, list[str]] = {}
    for d in drops:
        by_class_drops.setdefault(d["class"], []).append(d["word"])

    print(
        f"{'class':15s} {'words':>5s} {'kept':>4s} {'drop':>4s} {'rows':>4s}  dropped words"
    )
    thin: list[str] = []
    for cls_name, cls in BATTERY.items():
        kept_words = {r["word"] for r in by_class_rows.get(cls_name, [])}
        dropped = by_class_drops.get(cls_name, [])
        n_rows = len(by_class_rows.get(cls_name, []))
        flag = " <5!" if len(kept_words) < 5 else ""
        print(
            f"{cls_name:15s} {len(cls.words):5d} {len(kept_words):4d} "
            f"{len(dropped):4d} {n_rows:4d}  {', '.join(dropped)}{flag}"
        )
        if len(kept_words) < 5:
            thin.append(cls_name)
    n_words = len({(r["class"], r["word"]) for r in rows})
    n_primary = sum(1 for r in rows if r["is_primary"])
    print(
        f"\nTOTAL: {n_words} words kept ({len(drops)} dropped), "
        f"{len(rows)} anchor-variant rows ({n_primary} primary)"
    )
    # PAIRS coverage: both members need a surviving primary row
    primary_words = {r["word"] for r in rows if r["is_primary"]}
    broken = [
        (k, a, b)
        for k, a, b in PAIRS
        if a not in primary_words or b not in primary_words
    ]
    print(f"PAIRS: {len(PAIRS) - len(broken)}/{len(PAIRS)} pairs intact")
    for k, a, b in broken:
        missing = [w for w in (a, b) if w not in primary_words]
        print(f"  broken {k}: ({a!r}, {b!r}) — missing {missing}")
    if thin:
        print(
            f"classes under 5 surviving words (excluded from per-class stats): {thin}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tokenize-only",
        action="store_true",
        help="tokenizer pre-flight: print the battery coverage report and exit",
    )
    args = parser.parse_args()

    if args.tokenize_only:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(BASE_ID, revision=REVISION)
        rows, drops = resolve_battery(tok)
        coverage_report(rows, drops)
        return

    import torch

    from _emb_artifacts import write_artifact
    from llm_surgeon import surgery

    print(f"loading {BASE_ID} @ {REVISION[:8]} (CPU bf16) ...")
    t0 = time.time()
    model, tok = surgery.load_model(
        BASE_ID, mode="bf16", device_map="cpu", revision=REVISION
    )
    print(f"  loaded in {time.time() - t0:.0f}s")

    cfg = model.config
    assert cfg.tie_word_embeddings is False, "expected untied embeddings"
    emb = model.get_input_embeddings()
    assert emb is not None
    W_E = emb.weight.detach()
    W_U = model.lm_head.weight.detach()
    assert W_E.shape == W_U.shape == (cfg.vocab_size, cfg.hidden_size), (
        W_E.shape,
        W_U.shape,
    )
    assert W_E.data_ptr() != W_U.data_ptr(), "weights unexpectedly tied"
    n_vocab, d = W_E.shape
    n_real = len(tok)
    print(
        f"  W_E/W_U: {n_vocab} x {d}; tokenizer len {n_real} -> {n_vocab - n_real} padded rows"
    )
    assert 0 < n_real <= n_vocab

    rows, drops = resolve_battery(tok)
    coverage_report(rows, drops)

    # ---- battery vectors ----------------------------------------------------
    ids = torch.tensor([r["token_id"] for r in rows], dtype=torch.long)
    battery_out = {
        "rows": rows,  # parallel to the tensors below
        "E": W_E[ids].clone(),  # (n_rows, d) bf16
        "U": W_U[ids].clone(),
        "drops": drops,
        "pairs": list(PAIRS),
        "base_id": BASE_ID,
        "revision": REVISION,
        "n_vocab": n_vocab,
        "n_real": n_real,
        "hidden": d,
    }

    # ---- global statistics (float32, real rows only) ------------------------
    print("computing global statistics ...")
    norms_E = torch.empty(n_vocab, dtype=torch.float32)
    norms_U = torch.empty(n_vocab, dtype=torch.float32)
    eu_cos = torch.empty(n_vocab, dtype=torch.float32)
    sum_x = torch.zeros(d, dtype=torch.float32)
    S = torch.zeros(d, d, dtype=torch.float32)  # uncentered second moment * N
    for start in range(0, n_vocab, PCA_CHUNK):
        chunk = W_E[start : start + PCA_CHUNK].to(torch.float32)
        chunk_u = W_U[start : start + PCA_CHUNK].to(torch.float32)
        norms_E[start : start + chunk.shape[0]] = chunk.norm(dim=1)
        norms_U[start : start + chunk.shape[0]] = chunk_u.norm(dim=1)
        eu_cos[start : start + chunk.shape[0]] = torch.nn.functional.cosine_similarity(
            chunk, chunk_u, dim=1
        )
        if start < n_real:  # only real rows feed the global moments
            real = chunk[: max(0, min(n_real - start, chunk.shape[0]))]
            sum_x += real.sum(dim=0)
            S += real.T @ real
    mu = sum_x / n_real
    cov = S / n_real - torch.outer(mu, mu)  # centered covariance
    second_moment = S / n_real  # uncentered

    print("eigendecomposition (centered covariance) ...")
    evals_c, evecs_c = torch.linalg.eigh(cov)
    evals_c = evals_c.flip(0)  # descending
    evecs_c = evecs_c.flip(1)
    print("eigendecomposition (uncentered second moment) ...")
    evals_u, evecs_u = torch.linalg.eigh(second_moment)
    evals_u = evals_u.flip(0)
    evecs_u = evecs_u.flip(1)

    # cosine to the mean vector (anisotropy view), real rows
    mu_unit = mu / mu.norm()
    cos_to_mu = torch.empty(n_real, dtype=torch.float32)
    for start in range(0, n_real, PCA_CHUNK):
        chunk = W_E[start : min(start + PCA_CHUNK, n_real)].to(torch.float32)
        cos_to_mu[start : start + chunk.shape[0]] = (chunk @ mu_unit) / chunk.norm(
            dim=1
        ).clamp_min(1e-12)

    near_zero = (norms_E[:n_real] < 1e-3).nonzero().flatten()

    # random-pair cosine baselines (raw and centered), fixed seed
    g = torch.Generator().manual_seed(SEED)
    pair_idx = torch.randint(0, n_real, (10_000, 2), generator=g)
    a = W_E[pair_idx[:, 0]].to(torch.float32)
    b = W_E[pair_idx[:, 1]].to(torch.float32)
    rand_cos_raw = torch.nn.functional.cosine_similarity(a, b, dim=1)
    rand_cos_centered = torch.nn.functional.cosine_similarity(a - mu, b - mu, dim=1)

    global_out = {
        "norms_E": norms_E,
        "norms_U": norms_U,
        "eu_cos": eu_cos,
        "mu": mu,
        "cos_to_mu": cos_to_mu,
        "evals_centered": evals_c,
        "pcs_centered": evecs_c[:, :TOP_PCS].clone(),  # (d, TOP_PCS)
        "evals_uncentered": evals_u,
        "pcs_uncentered": evecs_u[:, :8].clone(),
        "rand_cos_raw": rand_cos_raw,
        "rand_cos_centered": rand_cos_centered,
        "near_zero_real_ids": near_zero,
        "base_id": BASE_ID,
        "revision": REVISION,
        "n_vocab": n_vocab,
        "n_real": n_real,
        "hidden": d,
        "seed": SEED,
        "pair_sample": 10_000,
    }

    # ---- random baseline rows ------------------------------------------------
    base_ids = torch.randperm(n_real, generator=g)[:N_BASELINE]
    baseline_out = {
        "token_ids": base_ids,
        "strings": [tok.decode([int(i)]) for i in base_ids],
        "E": W_E[base_ids].clone(),
        "U": W_U[base_ids].clone(),
        "seed": SEED,
        "base_id": BASE_ID,
        "revision": REVISION,
    }

    # ---- nearest-neighbor probes ----------------------------------------------
    print("nearest-neighbor probes ...")
    probes_out: dict[str, Any] = {
        "k": NEIGHBOR_K,
        "probes": [],
        "base_id": BASE_ID,
        "revision": REVISION,
        "note": "neighbors over real rows only; self excluded",
    }
    inv_norms = 1.0 / norms_E[:n_real].clamp_min(1e-12)
    for label, text in NEIGHBOR_PROBES:
        pids = tok.encode(text, add_special_tokens=False)
        if len(pids) != 1:
            probes_out["probes"].append(
                {"label": label, "text": text, "error": f"not single-token: {pids}"}
            )
            continue
        pid = pids[0]
        entry: dict[str, Any] = {"label": label, "text": text, "token_id": pid}
        for space, shift in (("raw", None), ("centered", mu)):
            v = W_E[pid].to(torch.float32)
            if shift is not None:
                v = v - shift
            v = v / v.norm().clamp_min(1e-12)
            scores = torch.empty(n_real, dtype=torch.float32)
            for start in range(0, n_real, PCA_CHUNK):
                chunk = W_E[start : min(start + PCA_CHUNK, n_real)].to(torch.float32)
                if shift is not None:
                    chunk = chunk - shift
                    scores[start : start + chunk.shape[0]] = (chunk @ v) / chunk.norm(
                        dim=1
                    ).clamp_min(1e-12)
                else:
                    scores[start : start + chunk.shape[0]] = (chunk @ v) * inv_norms[
                        start : start + chunk.shape[0]
                    ]
            scores[pid] = -2.0  # exclude self
            top = torch.topk(scores, NEIGHBOR_K)
            entry[space] = {
                "ids": top.indices.tolist(),
                "cos": top.values.tolist(),
                "strings": [tok.decode([int(i)]) for i in top.indices],
            }
        probes_out["probes"].append(entry)

    # ---- write + immediate validation -----------------------------------------
    for name, obj in (
        ("emb_battery_vectors.pt", battery_out),
        ("emb_global_stats.pt", global_out),
        ("emb_random_baseline.pt", baseline_out),
        ("emb_neighbor_probes.pt", probes_out),
    ):
        path = write_artifact(name)
        torch.save(obj, path)
        print(f"wrote {path}")

    # protocol sanity (ARC_PROCESS step 1): fail loudly here, not downstream
    assert not torch.isnan(norms_E).any() and not torch.isnan(eu_cos).any()
    assert battery_out["E"].shape[0] == len(rows)
    assert int(evals_c.numel()) == d
    frac1 = float(evals_c[0] / evals_c.sum())
    print(
        f"validation: n_rows={len(rows)} drops={len(drops)} "
        f"padded={n_vocab - n_real} near_zero_real={near_zero.numel()} "
        f"mean|cos_to_mu|={cos_to_mu.abs().mean():.4f} "
        f"rand_cos_raw_mean={rand_cos_raw.mean():.4f} "
        f"PC1_var_frac={frac1:.4f}"
    )


if __name__ == "__main__":
    sys.exit(main())
