"""Pretty-print the nearest-neighbor probe artifact (text report, no figure).

Model-free. Decodes emb_neighbor_probes.pt into a side-by-side raw vs
mean-centered neighbor table per probe; the output is quoted into the
observation write-up.
"""

from __future__ import annotations

from typing import Any

from _emb_artifacts import load_artifact, warn_if_mixed_sources


def main() -> None:
    warn_if_mixed_sources(["emb_neighbor_probes.pt"])
    probes = load_artifact("emb_neighbor_probes.pt")
    k: int = probes["k"]
    for p in probes["probes"]:
        if "error" in p:
            print(f"== {p['label']}  {p['text']!r}: SKIPPED ({p['error']})")
            continue
        print(f"== {p['label']}  probe={p['text']!r} (id {p['token_id']})")
        print(f"   {'raw':38s} | centered")
        for i in range(k):
            raw_s: str = p["raw"]["strings"][i]
            raw_c: float = p["raw"]["cos"][i]
            cen_s: str = p["centered"]["strings"][i]
            cen_c: float = p["centered"]["cos"][i]
            print(f"   {raw_c:+.3f} {raw_s!r:30s} | {cen_c:+.3f} {cen_s!r}")
        print()


def load_probe_labels() -> list[str]:
    """Probe labels present in the artifact (used by the audit)."""
    probes: dict[str, Any] = load_artifact("emb_neighbor_probes.pt")
    return [p["label"] for p in probes["probes"]]


if __name__ == "__main__":
    main()
