"""S3 analysis over the full-vocabulary sweep — decode structure into tokens.

Consumes emb_fullvocab_stats.pt (+ the cached W_E and dim-corr matrix) and
produces the first interpretation layer:

  1. feature-dim candidates — for the top-kurtosis dimensions, decode the
     tokens with extreme values on each (is the dimension interpretable?)
  2. entangled-dimension blocks — connected components of the |r| > R_EDGE
     dimension-correlation graph; per-block extreme tokens decoded
  3. handle-score census — per battery class, how many uncurated tokens
     score above the class's own 10th-percentile in-battery score; top
     decoded out-of-battery hits
  4. kNN-graph communities — torch label propagation over the k=32 graph;
     community size distribution; decoded samples of the largest communities
  5. branch-trigger evaluation — prints a BRANCH-TRIGGER line per fired
     trigger (bimodal edge-cosine, extreme kurtosis count, large blocks,
     large out-of-battery handle membership) for the session to act on

Tokenizer-only (no model weights); reads cache-first via _emb_artifacts.
Output: emb_fullvocab_analysis.pt + console report (quoted into the
observation write-up).
"""

from __future__ import annotations

import time
from typing import Any

import torch

from _emb_artifacts import load_artifact, read_artifact, write_artifact
from emb_capture import BASE_ID, REVISION
from emb_fullvocab_stats import WE_CACHE

TOP_KURT_DIMS = 20
EXTREME_TOKENS_PER_DIM = 12
R_EDGE = 0.3
LP_ITERS = 15
HANDLE_TOP = 30
SEED = 20260610


def decode(tok: Any, ids: list[int]) -> list[str]:
    return [tok.decode([int(i)]) for i in ids]


def main() -> None:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(BASE_ID, revision=REVISION)
    fv = load_artifact("emb_fullvocab_stats.pt")
    assert fv["stage"] == "full", "S3 runs on the full-stage artifact only"
    row_ids: torch.Tensor = fv["row_ids"].long()  # alive token ids
    n = int(fv["n_rows"])
    we = torch.load(read_artifact(WE_CACHE), weights_only=False)
    X = we["W"][:151_665].to(torch.float32)[row_ids]  # alive rows, fp32

    out: dict[str, Any] = {"inputs": ["emb_fullvocab_stats.pt"], "n_rows": n}
    triggers: list[str] = []

    # ---- 1. feature-dim candidates -----------------------------------------
    kurt: torch.Tensor = fv["dim_stats"]["kurtosis"]
    top_dims = torch.topk(kurt, TOP_KURT_DIMS)
    print(f"== top-{TOP_KURT_DIMS} kurtosis dims (median {float(kurt.median()):.2f})")
    feature_dims = []
    for rank in range(TOP_KURT_DIMS):
        d = int(top_dims.indices[rank])
        vals = X[:, d]
        ext = torch.topk(vals.abs(), EXTREME_TOKENS_PER_DIM)
        tok_ids = row_ids[ext.indices].tolist()
        entry = {
            "dim": d,
            "kurtosis": float(top_dims.values[rank]),
            "extreme_token_ids": tok_ids,
            "extreme_values": vals[ext.indices].tolist(),
            "extreme_tokens": decode(tok, tok_ids),
        }
        feature_dims.append(entry)
        toks = ", ".join(repr(s) for s in entry["extreme_tokens"][:8])
        print(f"  dim {d:4d} kurt {entry['kurtosis']:7.1f}: {toks}")
    out["feature_dims"] = feature_dims
    n_heavy = int((kurt > 20).sum())
    out["n_dims_kurt_gt20"] = n_heavy
    if n_heavy > 10:
        triggers.append(
            f"{n_heavy} dims with excess kurtosis > 20 — candidate dedicated-feature dims"
        )

    # ---- 2. entangled-dimension blocks --------------------------------------
    corr: torch.Tensor = torch.load(
        read_artifact("emb_fullvocab_dim_corr_matrix.pt"), weights_only=False
    )
    d_total = corr.shape[0]
    adj = corr.abs() > R_EDGE
    adj.fill_diagonal_(False)
    # connected components via iterative label min-propagation
    labels = torch.arange(d_total)
    for _ in range(50):
        neigh_min = labels.clone()
        rows, cols = adj.nonzero(as_tuple=True)
        neigh_min.scatter_reduce_(0, rows, labels[cols], reduce="amin")
        new = torch.minimum(labels, neigh_min)
        if torch.equal(new, labels):
            break
        labels = new
    uniq, counts = labels.unique(return_counts=True)
    block_sizes = counts[counts > 1].sort(descending=True).values
    print(
        f"== dim-corr blocks at |r|>{R_EDGE}: {len(block_sizes)} blocks of size >1; "
        f"sizes {block_sizes[:10].tolist()}"
    )
    blocks = []
    for lbl in uniq[counts > 2][:10]:
        dims = (labels == lbl).nonzero().flatten()
        # tokens with the largest energy inside the block's dims
        energy = X[:, dims].norm(dim=1)
        ext = torch.topk(energy, EXTREME_TOKENS_PER_DIM)
        tok_ids = row_ids[ext.indices].tolist()
        blocks.append(
            {
                "dims": dims.tolist(),
                "size": int(dims.numel()),
                "extreme_token_ids": tok_ids,
                "extreme_tokens": decode(tok, tok_ids),
            }
        )
        toks = ", ".join(repr(s) for s in blocks[-1]["extreme_tokens"][:8])
        print(f"  block size {dims.numel():3d} dims {dims[:6].tolist()}...: {toks}")
    out["corr_blocks"] = blocks
    out["block_sizes"] = block_sizes.tolist()
    if len(block_sizes) and int(block_sizes[0]) >= 5:
        triggers.append(
            f"largest entangled-dim block has {int(block_sizes[0])} dims at |r|>{R_EDGE}"
        )

    # ---- 3. handle-score census ----------------------------------------------
    handle: torch.Tensor = fv["handle_scores"].float()  # (n, C)
    classes: list[str] = fv["handle_classes"]
    battery = load_artifact("emb_battery_vectors.pt")
    batt_ids = {r["token_id"] for r in battery["rows"] if r["is_primary"]}
    batt_mask = torch.tensor([int(i) in batt_ids for i in row_ids])
    census = []
    print("== handle census (threshold = in-battery 10th percentile per class)")
    for ci, cname in enumerate(classes):
        in_class_ids = {
            r["token_id"]
            for r in battery["rows"]
            if r["is_primary"] and r["class"] == cname
        }
        in_mask = torch.tensor([int(i) in in_class_ids for i in row_ids])
        if int(in_mask.sum()) < 3:
            continue
        thr = float(handle[in_mask, ci].quantile(0.10))
        hits = (handle[:, ci] > thr) & ~batt_mask
        k = int(hits.sum())
        top = torch.topk(handle[:, ci].masked_fill(batt_mask, -2.0), HANDLE_TOP)
        tok_ids = row_ids[top.indices].tolist()
        census.append(
            {
                "class": cname,
                "threshold": thr,
                "out_of_battery_hits": k,
                "top_token_ids": tok_ids,
                "top_scores": top.values.tolist(),
                "top_tokens": decode(tok, tok_ids),
            }
        )
    census.sort(key=lambda e: -e["out_of_battery_hits"])
    for e in census[:12]:
        toks = ", ".join(repr(s) for s in e["top_tokens"][:6])
        print(
            f"  {e['class']:14s} thr {e['threshold']:+.3f} "
            f"hits {e['out_of_battery_hits']:6d}: {toks}"
        )
    out["handle_census"] = census
    big = [e for e in census if e["out_of_battery_hits"] > 1000]
    if big:
        triggers.append(
            f"{len(big)} handles with >1000 out-of-battery members "
            f"(largest: {big[0]['class']} at {big[0]['out_of_battery_hits']})"
        )

    # ---- 4. kNN communities (label propagation) ------------------------------
    ids: torch.Tensor = fv["knn_ids"].long()  # (n, k) indices into rows
    print("== label propagation over kNN graph")
    t0 = time.time()
    lab = torch.arange(n)
    g = torch.Generator().manual_seed(SEED)
    for it in range(LP_ITERS):
        order = torch.randperm(n, generator=g)
        neigh = lab[ids]  # (n, k)
        mode = neigh.mode(dim=1).values
        lab[order] = mode[order]
    uniq_c, counts_c = lab.unique(return_counts=True)
    comm_sizes = counts_c.sort(descending=True).values
    print(
        f"  {len(uniq_c)} communities in {time.time() - t0:.0f}s; "
        f"largest {comm_sizes[:8].tolist()}"
    )
    communities = []
    for lbl in uniq_c[counts_c.argsort(descending=True)[:12]]:
        members = (lab == lbl).nonzero().flatten()
        sample = members[torch.randperm(members.numel(), generator=g)[:12]]
        tok_ids = row_ids[sample].tolist()
        communities.append(
            {
                "size": int(members.numel()),
                "sample_token_ids": tok_ids,
                "sample_tokens": decode(tok, tok_ids),
            }
        )
        toks = ", ".join(repr(s) for s in communities[-1]["sample_tokens"][:8])
        print(f"  community size {members.numel():6d}: {toks}")
    out["communities"] = communities
    out["n_communities"] = int(len(uniq_c))
    out["community_sizes_top50"] = comm_sizes[:50].tolist()

    # edge-cosine distribution (bimodality probe)
    cos: torch.Tensor = fv["knn_cos"].float()
    edges = torch.linspace(-0.2, 1.0, 121)
    hist = torch.histogram(cos[:, 0], bins=edges).hist
    out["top1_cos_hist"] = hist
    out["top1_cos_hist_edges"] = edges
    peak = int(hist.argmax())
    # crude bimodality: a second local max separated from the global peak
    local_max = (
        (hist[1:-1] > hist[:-2]) & (hist[1:-1] > hist[2:])
    ).nonzero().flatten() + 1
    far_peaks = [
        int(i)
        for i in local_max
        if abs(int(i) - peak) > 15 and float(hist[i]) > 0.2 * float(hist[peak])
    ]
    if far_peaks:
        triggers.append(
            f"top-1 neighbor cosine distribution has secondary peak(s) at bins {far_peaks}"
        )

    out["branch_triggers"] = triggers
    print("== branch triggers")
    for t in triggers or ["(none fired)"]:
        print(f"  BRANCH-TRIGGER: {t}")

    path = write_artifact("emb_fullvocab_analysis.pt")
    torch.save(out, path)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
