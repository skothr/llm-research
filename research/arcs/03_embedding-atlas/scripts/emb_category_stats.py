"""Derive per-class category statistics from the battery embedding rows.

Model-free derivation (reads only .pt artifacts). For BOTH the raw and the
mean-centered space, computes over primary anchor rows:

  * per-class within vs between cosine distributions (classes with >=
    MIN_CLASS_N primary rows),
  * class-centroid cosine matrix,
  * contrast directions d_c = normalize(mean(A_c) - mean(battery \\ A_c)) —
    the nla-verbalizer arc's centroid-difference method (no sink removal),
    kept identical for layer-0 vs layer-20 comparability,
  * contrast-direction connectivity matrix cos(d_a, d_b).

"Centered" subtracts the full-vocabulary mean mu (from emb_global_stats.pt)
before measuring angles — the standard correction for anisotropic embedding
spaces (Mu & Viswanath 2018, arXiv:1702.01417).

Inputs : emb_battery_vectors.pt, emb_global_stats.pt
Output : emb_category_stats.pt (class: derived)
"""

from __future__ import annotations

from typing import Any

import torch

from _emb_artifacts import load_artifact, warn_if_mixed_sources, write_artifact

MIN_CLASS_N = 5
HIST_BINS = 80  # cosine histograms over [-1, 1]


def class_stats(
    X: torch.Tensor, class_of: list[str], classes: list[str]
) -> dict[str, Any]:
    """Within/between cosines + centroids + contrast directions for one space.

    X: (n, d) float32 anchor vectors (raw or centered), one row per primary
    anchor. class_of: per-row class name. classes: qualifying class names.
    """
    n, _ = X.shape
    Xn = X / X.norm(dim=1, keepdim=True).clamp_min(1e-12)
    cos = Xn @ Xn.T  # (n, n) — n is ~700, fine dense

    idx_of = {c: [i for i, ci in enumerate(class_of) if ci == c] for c in classes}
    per_class: dict[str, dict[str, Any]] = {}
    edges = torch.linspace(-1, 1, HIST_BINS + 1)
    for c in classes:
        idx = torch.tensor(idx_of[c], dtype=torch.long)
        mask = torch.zeros(n, dtype=torch.bool)
        mask[idx] = True
        sub = cos[idx][:, idx]
        iu = torch.triu_indices(len(idx), len(idx), offset=1)
        within = sub[iu[0], iu[1]]
        between = cos[idx][:, ~mask].flatten()
        per_class[c] = {
            "n": len(idx),
            "within_mean": float(within.mean()),
            "within_std": float(within.std()),
            "between_mean": float(between.mean()),
            "between_std": float(between.std()),
            "gap": float(within.mean() - between.mean()),
            "within_hist": torch.histogram(within, bins=edges).hist,
            "between_hist": torch.histogram(between, bins=edges).hist,
        }

    centroids = torch.stack([X[idx_of[c]].mean(dim=0) for c in classes])
    cn = centroids / centroids.norm(dim=1, keepdim=True).clamp_min(1e-12)
    centroid_cos = cn @ cn.T

    # contrast (centroid-difference) directions, arc-1 method
    total_sum = X.sum(dim=0)
    contrasts = []
    for c in classes:
        idx = idx_of[c]
        in_mean = X[idx].mean(dim=0)
        out_mean = (total_sum - X[idx].sum(dim=0)) / (n - len(idx))
        d = in_mean - out_mean
        contrasts.append(d / d.norm().clamp_min(1e-12))
    D = torch.stack(contrasts)
    connectivity = D @ D.T

    return {
        "per_class": per_class,
        "centroids": centroids,
        "centroid_cos": centroid_cos,
        "contrast_dirs": D,
        "connectivity": connectivity,
        "hist_edges": edges,
    }


def main() -> None:
    warn_if_mixed_sources(["emb_battery_vectors.pt", "emb_global_stats.pt"])
    battery = load_artifact("emb_battery_vectors.pt")
    glob = load_artifact("emb_global_stats.pt")
    mu: torch.Tensor = glob["mu"]

    rows: list[dict[str, Any]] = battery["rows"]
    primary = [i for i, r in enumerate(rows) if r["is_primary"]]
    E = battery["E"][primary].to(torch.float32)
    class_of = [rows[i]["class"] for i in primary]
    supergroup_of = {rows[i]["class"]: rows[i]["supergroup"] for i in primary}
    counts: dict[str, int] = {}
    for c in class_of:
        counts[c] = counts.get(c, 0) + 1
    classes = sorted(c for c, k in counts.items() if k >= MIN_CLASS_N)
    excluded = sorted(c for c, k in counts.items() if k < MIN_CLASS_N)
    print(
        f"{len(primary)} primary anchors; {len(classes)} classes >= {MIN_CLASS_N}; "
        f"excluded: {excluded}"
    )

    out: dict[str, Any] = {
        "classes": classes,
        "excluded_classes": excluded,
        "supergroup_of": supergroup_of,
        "min_class_n": MIN_CLASS_N,
        "primary_row_indices": primary,
        "spaces": {},
        "inputs": ["emb_battery_vectors.pt", "emb_global_stats.pt"],
        "base_id": battery["base_id"],
        "revision": battery["revision"],
    }
    for space, X in (("raw", E), ("centered", E - mu)):
        print(f"computing {space} space ...")
        out["spaces"][space] = class_stats(X, class_of, classes)
        stats = out["spaces"][space]["per_class"]
        gaps = sorted(stats.items(), key=lambda kv: -kv[1]["gap"])
        top = ", ".join(f"{c}={s['gap']:+.3f}" for c, s in gaps[:5])
        low = ", ".join(f"{c}={s['gap']:+.3f}" for c, s in gaps[-3:])
        print(f"  top gaps: {top}")
        print(f"  weakest : {low}")

    path = write_artifact("emb_category_stats.pt")
    torch.save(out, path)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
