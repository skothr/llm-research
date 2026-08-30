# Private-tracker ID map (MAIN-N)


This arc was worked against a private issue tracker that has since been
retired. Its `MAIN-N` ticket IDs are used as shorthand labels throughout
the arc README, the observation files, and
`observations/figures/INVENTORY.md` (the `examples/nla_*.py` docstrings
that carried them were rewritten to cite observation files instead).
**A reader cannot look any of them up** — the tracker is gone and was
never public.

Every `MAIN-N` that appears anywhere in this arc is listed below with the
in-repo artifact or migrated GitHub issue it resolves to. Two were never
migrated and have no successor; they are kept as bare IDs rather than
deleted, because removing them would silently erase the provenance of the
findings they are attached to.

Paths are relative to this directory unless noted.

| ID | Resolves to | How the mapping was established |
|---|---|---|
| MAIN-24 | [`observations/2026-05-13-nla-vocab-atlas-grid.md`](observations/2026-05-13-nla-vocab-atlas-grid.md) | F3's cited result (PC1 = 33.5% of the sink-removed vocab atlas, content-vs-function) is that file's Finding 2; INVENTORY's "category-attractor subspace separate from the sink subspace" open question is that file's H11, nearly verbatim (H11's heading says "category subspace") |
| MAIN-25 | [`observations/2026-05-13-nla-interpolation-flipbook.md`](observations/2026-05-13-nla-interpolation-flipbook.md) | the dense-interp observation describes MAIN-25 as the 20-step grid that flagged the flip between step 8 and step 9, which is this file's fig17/fig18 run; the mid-seq-native observation cites it for "the strong t=0.421 transition", with the ID's inline link pointing at this file |
| MAIN-26 | [`observations/2026-05-13-nla-discriminant-validation.md`](observations/2026-05-13-nla-discriminant-validation.md) | the mid-seq null-result observation links it directly — its prose reads `MAIN-26` followed by an inline markdown link to this file; F2's cited 56-79% top-5 hit rates are that file's Finding 2, and "MAIN-26 / fig29" matches its fig29 |
| MAIN-30 | GitHub [#13](https://github.com/skothr/llm-research/issues/13) | issue body: "Migrated from Linear MAIN-30" |
| MAIN-34 | [`observations/2026-05-15-nla-dense-interp-near-pivot.md`](observations/2026-05-15-nla-dense-interp-near-pivot.md) | that file's own `Private-tracker ID` header |
| MAIN-38 | GitHub [#12](https://github.com/skothr/llm-research/issues/12) | issue body: "Migrated from Linear MAIN-38" (no reference to it survives in repo prose) |
| MAIN-41 | GitHub [#11](https://github.com/skothr/llm-research/issues/11) | issue body: "Migrated from Linear MAIN-41" |
| MAIN-44 | [`observations/2026-05-14-nla-mid-seq-vocab-atlas-null-result.md`](observations/2026-05-14-nla-mid-seq-vocab-atlas-null-result.md) | that file's own `Private-tracker ID` header |
| MAIN-47 | [`observations/2026-05-14-nla-hierarchical-classifier-null-result.md`](observations/2026-05-14-nla-hierarchical-classifier-null-result.md) | that file's own `Private-tracker ID` header |
| MAIN-48 | [`observations/2026-05-14-nla-concept-arithmetic-atlas.md`](observations/2026-05-14-nla-concept-arithmetic-atlas.md) | that file's own `Private-tracker ID` header |
| MAIN-68 | GitHub [#10](https://github.com/skothr/llm-research/issues/10) | issue body: "Migrated from Linear MAIN-68" |
| MAIN-70 | [`observations/2026-05-14-nla-mid-seq-native-discriminants.md`](observations/2026-05-14-nla-mid-seq-native-discriminants.md) | that file's own `Private-tracker ID` header |
| MAIN-71 | [`observations/2026-05-15-nla-plateau-attractor-strength.md`](observations/2026-05-15-nla-plateau-attractor-strength.md) | that file's own `Private-tracker ID` header (part 2 of 2) |
| MAIN-265 | **not migrated** | D1 (discovery-viz frontend). No GitHub issue was migrated from it (issue #30's prose quotes the MAIN-265..272 range, nothing more); [D1](README.md#d1-discovery-viz-frontend) is the only surviving description |
| MAIN-266 | GitHub [#9](https://github.com/skothr/llm-research/issues/9) | issue body: "Migrated from Linear MAIN-266" |
| MAIN-267 | GitHub [#8](https://github.com/skothr/llm-research/issues/8) | issue body: "Migrated from Linear MAIN-267 (folds MAIN-347; …)" |
| MAIN-268 | GitHub [#7](https://github.com/skothr/llm-research/issues/7) | issue body: "Migrated from Linear MAIN-268" |
| MAIN-269 | GitHub [#6](https://github.com/skothr/llm-research/issues/6) | issue body: "Migrated from Linear MAIN-269" |
| MAIN-270 | GitHub [#5](https://github.com/skothr/llm-research/issues/5) | issue body: "Migrated from Linear MAIN-270" |
| MAIN-271 | **not migrated** | D7 (per-token live-trajectory viz). No GitHub issue was migrated from it; [D7](README.md#d7-per-token-live-trajectory-viz) is the only surviving description |
| MAIN-272 | GitHub [#4](https://github.com/skothr/llm-research/issues/4) | issue body: "Migrated from Linear MAIN-272" |
| MAIN-347 | GitHub [#8](https://github.com/skothr/llm-research/issues/8), folded into MAIN-267 | the only surviving mention is #8's own migration note, quoted in the MAIN-267 row above |

Migration context: `docs/planning/2026-07-25-backlog-groom.md` (repo root).

`CC-MAIN-2024-10` under `theory/` is unrelated — it is a Common Crawl
snapshot name, not a tracker ID.
