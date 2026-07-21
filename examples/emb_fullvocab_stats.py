"""Full-vocabulary sweep for the embedding-atlas arc — staged collection.

Reverses the battery approach: instead of curated groups probed for shared
features, sweep ALL real token embeddings for similarity and dimension-level
structure. See research/arcs/embedding-atlas/plans/2026-06-10-fullvocab-sweep.md.

Stages (scaling-step discipline — validate the protocol before the full run):

    python examples/emb_fullvocab_stats.py --dump           # S0: one model load,
        cache W_E/W_U bf16 to .cache/emb_artifacts/ (gitignored; ~2.2 GB)
    python examples/emb_fullvocab_stats.py --stage sample   # S1: 10k random
        alive rows; writes emb_sample_* artifacts (cache-only, never committed)
    python examples/emb_fullvocab_stats.py --stage full     # S2: all alive rows;
        writes the committable emb_fullvocab_* artifacts

Measurements per stage:
  * per-dimension moments (mean/std/skew/kurtosis, fp64 accumulation)
  * dimension-dimension correlation summary (histogram of off-diagonal r,
    top-1000 |r| pairs, eigen-spectrum; full matrix left in cache)
  * k-NN graph (k=32, cosine, centered, dead rows excluded; int32 ids +
    fp16 cosines)
  * handle scores: every row projected onto the battery contrast directions
    (centered space) from emb_category_stats.pt

"Alive" rows = real-vocab rows (id < tokenizer length) with norm >= 1e-3 —
the 1,959 dead rows and 399 padded rows are excluded from every statistic.

Device: tries CUDA for the k-NN matmuls, falls back to CPU. All reductions
fp32 (moments fp64); artifacts store fp16/fp32 as noted.
"""

from __future__ import annotations

import argparse
import time
from typing import Any

import torch

from _emb_artifacts import load_artifact, read_artifact, write_artifact

SEED = 20260610
KNN_K = 32
SAMPLE_N = 10_000
TILE = 2048
TOP_CORR_PAIRS = 1000
DEAD_NORM_THRESHOLD = 1e-3

WE_CACHE = "emb_WE_bf16.pt"  # S0 dump targets (gitignored cache only)
WU_CACHE = "emb_WU_bf16.pt"


def dump_matrices() -> None:
    """S0: one model load; persist W_E / W_U (bf16) to the working cache."""
    from llm_surgeon import surgery

    from emb_capture import BASE_ID, REVISION

    print(f"loading {BASE_ID} @ {REVISION[:8]} (CPU bf16) ...")
    t0 = time.time()
    model, tok = surgery.load_model(
        BASE_ID, mode="bf16", device_map="cpu", revision=REVISION
    )
    print(f"  loaded in {time.time() - t0:.0f}s")
    emb = model.get_input_embeddings()
    assert emb is not None
    meta = {"base_id": BASE_ID, "revision": REVISION, "n_real": len(tok)}
    torch.save({"W": emb.weight.detach().clone(), **meta}, write_artifact(WE_CACHE))
    torch.save(
        {"W": model.lm_head.weight.detach().clone(), **meta},
        write_artifact(WU_CACHE),
    )
    print(f"dumped W_E/W_U to cache ({WE_CACHE}, {WU_CACHE})")


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        try:
            probe = torch.zeros(8, device="cuda") + 1
            assert float(probe.sum()) == 8.0
            return torch.device("cuda")
        except Exception as e:  # noqa: BLE001 — any CUDA failure means fallback
            print(f"CUDA unusable ({e}); falling back to CPU")
    return torch.device("cpu")


def dim_moments(X: torch.Tensor) -> dict[str, torch.Tensor]:
    """Per-dimension mean/std/skew/excess-kurtosis, fp64 accumulation."""
    n = X.shape[0]
    s1 = torch.zeros(X.shape[1], dtype=torch.float64)
    s2 = torch.zeros_like(s1)
    s3 = torch.zeros_like(s1)
    s4 = torch.zeros_like(s1)
    for start in range(0, n, 8192):
        c = X[start : start + 8192].to(torch.float64)
        s1 += c.sum(0)
        s2 += (c**2).sum(0)
        s3 += (c**3).sum(0)
        s4 += (c**4).sum(0)
    mean = s1 / n
    var = s2 / n - mean**2
    std = var.clamp_min(1e-30).sqrt()
    m3 = s3 / n - 3 * mean * s2 / n + 2 * mean**3
    m4 = s4 / n - 4 * mean * s3 / n + 6 * mean**2 * s2 / n - 3 * mean**4
    skew = m3 / std**3
    kurt = m4 / var.clamp_min(1e-30) ** 2 - 3.0  # excess kurtosis
    return {
        "mean": mean.to(torch.float32),
        "std": std.to(torch.float32),
        "skew": skew.to(torch.float32),
        "kurtosis": kurt.to(torch.float32),
    }


def dim_correlation_summary(
    X: torch.Tensor, std: torch.Tensor, mean: torch.Tensor
) -> dict[str, Any]:
    """Correlation-matrix summary; full matrix returned for cache persist."""
    n, d = X.shape
    S = torch.zeros(d, d, dtype=torch.float64)
    for start in range(0, n, 8192):
        c = X[start : start + 8192].to(torch.float64) - mean
        S += c.T @ c
    cov = S / n
    denom = torch.outer(std.double(), std.double()).clamp_min(1e-30)
    corr = (cov / denom).to(torch.float32)
    iu = torch.triu_indices(d, d, offset=1)
    off = corr[iu[0], iu[1]]
    edges = torch.linspace(-1, 1, 201)
    hist = torch.histogram(off, bins=edges).hist
    top = torch.topk(off.abs(), TOP_CORR_PAIRS)
    evals = torch.linalg.eigvalsh(corr.double()).flip(0).to(torch.float32)
    return {
        "corr": corr,  # cache-only; stripped before committing summary
        "offdiag_hist": hist,
        "offdiag_hist_edges": edges,
        "offdiag_abs_mean": float(off.abs().mean()),
        "offdiag_max": float(off.abs().max()),
        "top_pairs_i": iu[0][top.indices].to(torch.int32),
        "top_pairs_j": iu[1][top.indices].to(torch.int32),
        "top_pairs_r": off[top.indices],
        "corr_evals": evals,
    }


def knn_graph(
    Xc: torch.Tensor, device: torch.device, k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """k-NN by cosine over centered rows. Returns (ids int32, cos fp16)."""
    n = Xc.shape[0]
    Xn = Xc / Xc.norm(dim=1, keepdim=True).clamp_min(1e-12)
    Xn_dev = Xn.to(device)
    ids = torch.empty((n, k), dtype=torch.int32)
    cos = torch.empty((n, k), dtype=torch.float16)
    t0 = time.time()
    for start in range(0, n, TILE):
        tile = Xn_dev[start : start + TILE]
        scores = tile @ Xn_dev.T  # (tile, n)
        for r in range(tile.shape[0]):
            scores[r, start + r] = -2.0  # exclude self
        top = torch.topk(scores, k, dim=1)
        ids[start : start + TILE] = top.indices.to(torch.int32).cpu()
        cos[start : start + TILE] = top.values.to(torch.float16).cpu()
        if start % (TILE * 16) == 0 and start:
            rate = start / (time.time() - t0)
            print(f"  knn {start}/{n} rows ({rate:.0f} rows/s)")
    return ids, cos


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump", action="store_true", help="S0: cache W_E/W_U")
    parser.add_argument("--stage", choices=["sample", "full"])
    args = parser.parse_args()

    if args.dump:
        dump_matrices()
        return
    if args.stage is None:
        parser.error("need --dump or --stage")

    we = torch.load(read_artifact(WE_CACHE), weights_only=False)
    W: torch.Tensor = we["W"]
    n_real: int = we["n_real"]
    real = W[:n_real].to(torch.float32)
    norms = real.norm(dim=1)
    alive_ids = (norms >= DEAD_NORM_THRESHOLD).nonzero().flatten()
    print(
        f"real rows {n_real}, alive {alive_ids.numel()} "
        f"(dead {n_real - alive_ids.numel()})"
    )

    g = torch.Generator().manual_seed(SEED)
    if args.stage == "sample":
        perm = torch.randperm(alive_ids.numel(), generator=g)[:SAMPLE_N]
        row_ids = alive_ids[perm].sort().values
        prefix = "emb_sample"
    else:
        row_ids = alive_ids
        prefix = "emb_fullvocab"
    X = real[row_ids]
    n = X.shape[0]
    print(f"stage={args.stage}: {n} rows")

    device = pick_device()
    print(f"device: {device}")

    # 1. per-dim moments
    t0 = time.time()
    moments = dim_moments(X)
    print(
        f"moments in {time.time() - t0:.0f}s; "
        f"kurtosis max {float(moments['kurtosis'].max()):.1f} "
        f"median {float(moments['kurtosis'].median()):.2f}"
    )
    assert not torch.isnan(moments["kurtosis"]).any()

    # 2. dim-correlation summary
    t0 = time.time()
    corr_sum = dim_correlation_summary(X, moments["std"], moments["mean"])
    print(
        f"dim-corr in {time.time() - t0:.0f}s; "
        f"|r| mean {corr_sum['offdiag_abs_mean']:.4f} "
        f"max {corr_sum['offdiag_max']:.3f}"
    )
    full_corr = corr_sum.pop("corr")
    torch.save(full_corr, write_artifact(f"{prefix}_dim_corr_matrix.pt"))

    # 3. kNN graph (centered space — consistent with battery analyses)
    mu = X.mean(dim=0)
    t0 = time.time()
    ids, cos = knn_graph(X - mu, device, KNN_K)
    print(
        f"knn in {time.time() - t0:.0f}s; "
        f"top1 cos mean {float(cos[:, 0].float().mean()):+.3f} "
        f"k32 cos mean {float(cos[:, -1].float().mean()):+.3f}"
    )

    # 4. handle scores against the battery contrast directions
    cs = load_artifact("emb_category_stats.pt")
    D: torch.Tensor = cs["spaces"]["centered"]["contrast_dirs"]  # (C, d)
    classes: list[str] = cs["classes"]
    Xc_unit = (X - mu) / (X - mu).norm(dim=1, keepdim=True).clamp_min(1e-12)
    handle = (Xc_unit @ D.T).to(torch.float16)  # (n, C)
    print(f"handle scores: {handle.shape}, max {float(handle.float().max()):+.3f}")

    out: dict[str, Any] = {
        "stage": args.stage,
        "row_ids": row_ids.to(torch.int32),
        "n_rows": n,
        "seed": SEED,
        "knn_k": KNN_K,
        "dead_norm_threshold": DEAD_NORM_THRESHOLD,
        "dim_stats": moments,
        "dim_corr_summary": corr_sum,
        "knn_ids": ids,  # indices INTO row_ids, not token ids
        "knn_cos": cos,
        "handle_scores": handle,
        "handle_classes": classes,
        "base_id": we["base_id"],
        "revision": we["revision"],
    }
    path = write_artifact(f"{prefix}_stats.pt")
    torch.save(out, path)
    print(f"wrote {path}")

    # validation gates (fail loudly, ARC_PROCESS step 1)
    assert not torch.isnan(cos.float()).any()
    assert float(cos[:, 0].float().min()) > -1.01
    assert handle.shape == (n, len(classes))
    print("validation gates passed")


if __name__ == "__main__":
    main()
