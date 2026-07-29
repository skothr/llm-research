# Backlog groom — 2026-07-25

Full-backlog verification of the 17 open GitHub issues (#1–#17) against the live
repo, per the grooming discipline (verify every body's premises, verdict with
evidence, persist before applying). All 17 were migrated from Linear on
2026-07-15 and none had been triaged since. Verification: four parallel
read-only passes over the repo (git log, committed artifacts, MANIFESTs,
observations) on 2026-07-25; repo state = `main` @ origin (0 ahead / 0 behind).

Verdict vocabulary: `close-resolved` / `close-invalid` / `refine` / `keep` /
`merge` / `needs-user` (see the github-issues skill, grooming.md).

## Verdict table

| # | Kind | Verdict | Evidence anchor |
|---|---|---|---|
| 1 | tech-debt | **close-resolved** | `git rev-list --left-right --count origin/main...main` → `0 0`; the "11 commits ahead, unpushed" premise is gone. Second sub-item (pip_freeze keep-vs-scrub) is moot: `research/arcs/02_subliminal/data/step0-owl-neutral-decode/pip_freeze.txt:45` records the editable install as a path-free comment; no absolute paths in the file. |
| 2 | idea | **keep + needs-user (routing)** | Premise verified: `theory/kb/index/` has `papers.json`/`topics.md`/`timeline.md` (+ `contradictions.md`, `_phase2-additions/` since filing); no theory-KB skill exists in any skills dir; no counterpart issue in claude-config. Deliverable would land in claude-config — user decides where the tracker lives. |
| 3 | tech-debt | **refine** | Half-landed: `.gitattributes` gained `theory/sources/papers/*.pdf` LFS rule (2026-07-18) but the 6 other tracked PDFs (5× `theory/series/paper-*/main.pdf` + `theory/archive/2026-05-03-pre-expansion/llm-core-architecture.pdf`, ~6.3 MB total) are raw packfile blobs (verified `%PDF` heads, `git ls-tree -l`). Re-scan also found two large text blobs: `research/arcs/04_jspace/data/fitting_prompts_c4en_n1000.json` (3.5 MB) + `fitting_prompts_wikitext103_n1000.json` (1.0 MB). |
| 4 | research | **refine** | Rabbit-haiku part done (`arcs/01.../observations/2026-05-12-nla-trajectory-rabbit-haiku.md`, `...-prompt-battery-rabbit-eval.md`); two existing negative eval-awareness results uncited by the body; the Anthropic announcement is now in the KB as `anthropic2026-nla` (`theory/kb/index/papers.json`, excerpts `#sec-auditing`). Planned-rhyme + ethical-discussion probes still unrun. |
| 5 | research | **keep** | Single anchor-pair basin data only (`dense_interp_near_pivot.pt`, `plateau_attractor_test.pt`); no multi-pair sweep. Dependency on #8 is real — make it an explicit `Blocked by #8`. Also `examples/nla_plateau_attractor_test.py:126-135` self-documents ad-hoc thresholds → second reason a calibrated baseline is required. |
| 6 | research | **keep + parent of #11** | No cross-model NLA work anywhere; arc 03 lists an overlapping unrun TinyLlama item (`arcs/03_embedding-atlas/README.md:176`), so one TinyLlama run can serve both arcs. #11 is the concrete first step → wire as native sub-issue. |
| 7 | research | **refine** | Body's "the dual question was not posed" is now false: `arcs/01.../observations/2026-05-14-nla-mid-seq-native-discriminants.md` computed the 23×23 cross-protocol cosine matrix (mean same-category diagonal +0.0784). The rank-K SVD subspace question remains genuinely open; protocol 3 (post-assistant-turn-start) needs a new capture; protocols 1–2 have committed `.pt` data. |
| 8 | research | **keep + priority → high** | Confirmed unrun: zero `randn` in `examples/nla_*.py`; arc README Theme 9 / L2 "never investigated"; independently reconfirmed unrun by `observations/2026-07-22-crossarc-jspace-l20-legibility.md`. Blast radius grew: arc 04 inherits the caveat as a named limitation (`arcs/04_jspace/README.md`, **Limitations**: "stage-6 inherits the NLA arc's unaudited AV format bias (mitigated by nulls, not eliminated)"), and `examples/jspace_nla_crosstie.py:83-98` (`AV_TEMPLATE_STOP`) provides a ready phrase list for the probe. Gates #5 and colors two arcs → high. |
| 9 | research | **refine** | Both sub-items undone: `examples/nla_forced_continuation.py` has exactly 4 pairs, no self-output pair; matched-pair eval probe unrun (its H2 design already specified in the 2026-05-12 prompt-battery observation). Script's own 2026-05-29 `PROTOCOL CAVEAT` (BPE seam at prompt/completion concat) is load-bearing for the transcript extension. |
| 10 | follow-up | **refine** | Unrun; 34% figure verified (`observations/2026-05-13-nla-discriminant-validation.md:51`, n=29 expected-country rows — not the 13-prompt pool). Paths stale (monorepo `testing/` prefix). Scope correction: re-capture must ADD a strict pool/artifact (LFS capture-root `country_concept_vector.pt` has consumers fig13, `pairwise_and_hotdims.pt`, AUDIT 8) rather than overwrite, and update MANIFEST + audit + INVENTORY. GPU re-capture, not cheap CPU. |
| 11 | idea | **refine + child of #6** | Unrun; PC numbers verified exact (`...vocab-atlas-grid.md:59-62`, INVENTORY:127). Method flaw: `cos(PC1_qwen, PC1_tinyllama)` undefined across d_model 3584 vs 2048 — replace with anchor-score correlation over the 128 shared anchors (or Procrustes/CCA). Sink-dim set is Qwen-L20-specific → TinyLlama needs its own sink identification. Prefer `--base-id`/`--layer` CLI args + separate artifact over in-place edits. |
| 12 | idea | **refine** | Unrun; the 7 sink dims verified (machine-asserted in `examples/nla_audit_findings.py:342`). Wrong hook host: `llm_surgeon.surgery` exposes structural/weight ops only, no h-dim masking API. Correct mechanism: new dim-mask hook in `examples/_layer_hooks.py` (existing `LayerOutputReplaceHook` fires prefill-only, single position — insufficient as-is). Add a third arm: the 8 feature dims, per `geometric-deep-dive.md:104` (sink-vs-feature is the sharper test). Artifact path: `arcs/01.../data/aggregate_faithfulness.pt`. |
| 13 | research → follow-up | **refine + relabel** | Finding already published (arc README F4, fig15/fig16 committed, AUDIT 10); all four Δh numbers verified exact. Only the 20+20 battery remains → `follow-up`. Preconditions from F4: mid-seq tokenize-then-locate capture (BPE seam) + position-matched Δpos=0. Cited observation path + artifact path both stale. |
| 14 | research | **keep** (precision fixes) | Numbers verified exact (`arcs/03.../observations/2026-06-10-emb-category-structure.md` F-C1); `vocab_atlas.pt` committed (LFS, MANIFEST-pinned) so the prerequisite is satisfied. Fixes: "23 contrast directions" must read "arc-1's 23 L20 discriminants recomputed at L0" (arc 3's own battery has 54); `emb_layer20_bridge.py` is named in H2 + README deferred list, not the plans file. |
| 15 | research | **keep** (one wording fix) | Numbers verified exact (capital_of shuffle +0.3881 ±0.0022, margins +0.02–0.05); `emb_pair_directions.pt` committed. Fix: "model-free" → "requires one small W_E row capture (pinned model); analysis thereafter model-free". |
| 16 | tech-debt | **close-resolved** | Fixed in `d984aa44` ("matplotlib stub drift — import Rectangle/Circle from matplotlib.patches"): zero `plt.Rectangle`/`plt.Circle` remain; all 7 scripts import from `matplotlib.patches`; no suppressions added. Root `pyrightconfig.json` landed on main (`fb401bc4`). Residual: the final `pyright examples/` clean-run can't execute until a venv exists — folded into #17 as a verification step. |
| 17 | tech-debt | **refine + priority → high** | State moved past the body: the `.venv` symlink is now gone entirely (no venv at all; `.venv` gitignored). The committed root `pyrightconfig.json` still carries machine-absolute `extends`/`venvPath`/`extraPaths` entries (lines 11–24), and `theory/series/pyrightconfig.json` a machine-absolute `extends` — a portability and information-hygiene problem in a public repo, and it now also blocks #16's final verification. Fix: create the documented venv, then make both configs machine-portable (relative paths / local-override layer). |

## Cross-cutting fixes (applied with the body edits)

1. **Dead archive footer** — every migrated body cites `references/linear-archive/issues.jsonl`, a path that does not exist in this repo. Reword to "private tracker archive (not in this repo)".
2. **Dead private-tracker links** — `linear.app/...` MAIN-N hyperlinks in #4–#10 bodies are unreachable for anyone reading this public repo. Replace with plain `MAIN-N` text, the migrated GitHub issue number where one exists (MAIN-41 → #11), or the in-repo observation file the MAIN ticket resolved to (map recorded in the verification transcripts).
3. **Repo files carry the same two problems** — `research/arcs/01_nla-verbalizer/README.md` (D-series section) has eight live `linear.app` links, and several tracked files embed machine-absolute paths (root + series pyrightconfig, `examples/jspace_lens_eval.py:51` default, `examples/jspace_data_manifest.py:771`, `arcs/04_jspace/data/MANIFEST.json:5`). Tracked as new issues (below), not fixed in this pass.

## New issues proposed

- **[tech-debt] Make committed configs and script defaults machine-portable** — root + `theory/series` pyrightconfig, `jspace_lens_eval.py` default eval dir, `jspace_data_manifest.py` / arc-4 MANIFEST provenance strings. Related to #17. (Privacy-audit detail routes to the private config repo per routing policy.)
- **[docs] Replace private-tracker links in arc-01 README** — eight `linear.app` MAIN-26x links in the D-series section → plain MAIN-N text + in-repo observation paths / GH issue numbers.

## Ordered path forward (post-apply)

1. **#17** (venv + portable configs) — unblocks #16 verification, kills the config-portability debt. Then the new portability issue rides the same change.
2. **#8** (AV format-bias audit, high) — gates #5 and de-risks re-readings in two arcs; `AV_TEMPLATE_STOP` gives it a head start.
3. **#3** (finish LFS migration for remaining PDFs) — mechanical, isolated.
4. Cheap-first research ladder: **#11** (CPU-only, serves #6 + arc-03's deferred item) → **#13** battery → **#15** / **#14** (model-light atlas follow-ups) → **#9** → **#12** (new hook + GPU) → **#10** (GPU re-capture + artifact churn) → **#7** (needs new capture) → **#5** (after #8) → **#4** / **#6** umbrellas.
5. **#2** — after the user's routing call.

## Needs-user list

1. Sign-off on applying all verdicts above (two closes, ten body REPLACEs, two priority raises, one relabel, #6→#11 parenting, two new issues).
2. #2 routing: keep the tracker in llm-research or move it to the private config repo (deliverable lands there).
3. #16 close despite the pyright clean-run being unverifiable until #17 lands — acceptable, or keep open pending venv?
4. History note: the machine-absolute paths above are also in past commits; removing them from HEAD does not remove them from history. A history rewrite is almost certainly not worth it (no credentials involved) — confirm the leave-history-alone default.
5. One further account-side privacy item was reported privately in-session (not repeated here).

## Decisions (user sign-off, 2026-07-25)

1. **Apply everything** — all verdicts, both new issues, groom-doc PR.
2. **#2 stays in llm-research**, scope expanded: the skill must carry robust
   instructions for *exhaustive* paper search/filter (systematic index sweep,
   not first-hit lookup). Recorded alternative: a dedicated `llm-theory` repo
   housing KB + skill, wired/symlinked so llm-research and other
   technical/scientific repos resolve the same KB reference — open design
   decision, captured in #2's body.
3. **#16 is NOT closed on code inspection alone** — user directed unblocking
   the verification instead. The documented venv build was started during the
   groom (advancing #17); #16 closes after a clean `pyright examples/` run.
4. **History left alone** — no credentials involved; rewrite cost outweighs
   the benefit.
