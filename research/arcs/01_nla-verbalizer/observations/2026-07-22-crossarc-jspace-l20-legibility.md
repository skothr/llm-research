# Cross-arc note: the L20 capture layer sits below the 7B J-lens legibility onset (from the jspace arc)

**Date/context:** 2026-07-22, filed at jspace arc close per its stage-6
cross-tie (`research/arcs/jspace/observations/2026-07-21-nla-crosstie-stage6.md`).

**Finding relevant to this arc:** the jspace arc's depth mapping of
Qwen2.5-7B (stage 3/4) shows J-lens readouts become contentful only
around **L22**; this arc's capture layer (`hidden_states[20]` = jlens
source_layer 19) sits *below* that onset — J-lens top-10 there is only
intermittently legible. Two implications for NLA work:

1. The AV's demonstrated ability to verbalize L20 activations is *not*
   redundant with linear token-indexed readouts at that depth — the AV
   reads content a J-lens cannot cleanly surface there. That strengthens
   the case that the AV decodes non-token-aligned structure.
2. The jspace decomposition experiment found the AV's content sits in
   the **non-J-space residual** of h_20 (component removal damages
   verbalization no more than random removal) — if a future AV is
   trained at L22–24 (above the onset), the same experiment would test
   whether that changes with depth.

Also inherited caveat, still open on this arc's side: the AV format-bias
audit (this arc's L2/D3 item) remains unrun; the jspace cross-tie
mitigated it with content-word filters and mismatched-pairing nulls but
flagged it as the main inflation risk on overlap metrics.

**References:** jspace arc README synthesis §Novel contributions;
`[gurnee2026-workspace §2.3]`.
