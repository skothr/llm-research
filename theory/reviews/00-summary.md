# Review-wave consolidated findings (2026-05-06)

Six independent fresh-context opus reviewers, each with a different lens, audited
the just-completed five-paper LaTeX series at `theory/series/` (70 sections,
~394pp, 5 PDFs). Each reviewer was read-only on the source and wrote one
markdown report; this document triages their combined findings.

| # | Reviewer | Lens | Report | Findings |
|---|----------|------|--------|----------|
| 1 | Adversarial-content | Hostile / skeptical: unsupported claims, citation/scope mismatch | `01-adversarial-content.md` | 0 BLOCKER, 4 MAJOR, 6 MINOR, 3 NIT |
| 2 | Cross-paper coherence | Notation drift, contradictions between papers, thread duplication | `02-cross-paper-coherence.md` | 0 BLOCKER, 2 MAJOR, 5 MINOR |
| 3 | Math correctness | Equation-vs-source verbatim check, dimensional consistency | `03-math-correctness.md` | 0 BLOCKER, 0 MAJOR, 4 MINOR |
| 4 | Citation-tier discipline | Tier-A backing, deepencite hiding load-bearing claims | `04-citation-tier.md` | 0 BLOCKER, 0 MAJOR, 2 MINOR, 3 NIT |
| 5 | Frontier currency (web-enabled) | 2025-26 numerical claims still hold; recent releases | `05-frontier-currency.md` | **1 BLOCKER**, 4 MAJOR |
| 6 | Feynman-bar pedagogy | Analogy-returns-to-math, intuitions grounded, callout discipline | `06-pedagogy.md` | 0 BLOCKER, 1 MAJOR, 2 MINOR |

**Aggregate: 1 BLOCKER, 11 MAJOR, ~19 MINOR, 6 NIT.**

The single BLOCKER is from the web-enabled currency reviewer: a contradiction
that Paper 3 §14 frames as "open" has been partially closed by a Nature paper
that landed after the KB build cutoff. The 11 MAJORs are concentrated in two
clusters — frontier-currency drift and citation-source mismatch — and both
clusters are short-form mechanical fixes rather than re-writes.

## BLOCKER (must fix before publication)

### B1 — AIME contamination contradiction is now partially closed

- **Source:** Reviewer #5 (frontier-currency).
- **Where:** Paper 3 §14 (`14-contradictions.tex`), contradiction C5 ("AIME contamination").
- **What's stale:** the section frames the AIME contamination contradiction as open. DeepSeek-R1 was peer-reviewed in *Nature* (Sept 2025) and the published version includes a decontamination audit plus a Qwen2-7B control experiment that addresses the contamination concern.
- **Fix:** add a one-paragraph update to Paper 3 §14 C5: "The Sept 2025 *Nature* publication of DeepSeek-R1 includes a decontamination audit \citep{...} and a Qwen2-7B control that reduces but does not fully close this contradiction." Add the Nature publication as a new key in `papers.json`.
- **Estimated cost:** 15 minutes including bib-key add and rebuild.

## MAJOR (concentrated mechanical fixes)

### Citation / source mismatch cluster (5 issues)

| ID | Reviewer | Where | Issue | Fix |
|---|---|---|---|---|
| M1 | R1 | Paper 3 §11 | AIME number error: "15.6%→71%" should be "15.6%→77.9%" (canonical R1 number) | sed-replace one number |
| M2 | R1 | Paper 2 §14 + Paper 3 §14 | `rlvr-limits-2025` cited bare as load-bearing counter-claim despite no KB excerpt | Add `\deepencite{rlvr-limits-2025 §X, citation-pending}` flag, or add the KB excerpt |
| M3 | R1 | Paper 5 §14 | `sadasivan2023-detection` cited without `\deepencite` marker that §12 uses | Add the marker for consistency |
| M4 | R2 | Paper 3 (3 sites) | Hoffmann Chinchilla parametric loss cited as "Eq. 10" but it's "Eq. 2" in source | sed-replace 3 occurrences |
| M5 | R3 | Paper 3 §7 | GRPO Eq. cites `deepseek-r1` for the token-level form that's actually from `shao2024` | swap cite-key |

### Frontier-currency drift cluster (4 issues)

| ID | Reviewer | Where | Issue | Fix |
|---|---|---|---|---|
| M6 | R5 | Paper 1 §13 + §14 | Closed-row labels stale 6-12 months: GPT-5/5.4, Claude 4-family, Gemini 2.5 Deep Think | Update the snapshot table closed-vendor rows + add 1-paragraph caveat |
| M7 | R5 | Paper 1 §13 | Mistral Large 3 (Dec 2025) is MoE — flips Paper 1's dense-flagship Mistral row | Update one snapshot cell + add brief note |
| M8 | R5 | Paper 5 §10 + §14 | OpenAI/Apollo deliberative-alignment paper (Sept 2025) sharpens scheming contradiction | Add 1-paragraph update + new bib key |
| M9 | R5 | Paper 3 §11 | "2026-05 consensus leans Yue with caveats" framing is one-sided per recent counter-evidence | Soften the consensus claim + cite the counter-paper |

### Single-issue MAJORs (2)

| ID | Reviewer | Where | Issue | Fix |
|---|---|---|---|---|
| M10 | R1 | Paper 1 §13 + Paper 3 §13 | Universal-quantifier convergence claims built on closed-vendor system cards (tier B by project rules) | Add explicit tier-B hedge to the snapshot intro paragraph |
| M11 | R2 | Paper 3 §7 | Intuition box overstates contested Yue RLVR-limit hypothesis as fact (next paragraph walks it back, but the intuition reads ungated) | Soften the intuition box copy |

## MINOR (cleanup, batchable)

19 minor findings spread across the six reports. All are mechanical and can be
batched into a single sweep:

- **Notation drift in Paper 4 from canonical shape macros** (R2): Paper 4 sections use $h_\ell$ and $H$ inconsistently with Paper 1's $\BSH$/$\BHSdh$ macros. Define explicit aliases or normalize on the canonical macros.
- **Paper 2 one-off `C_{\text{infer}}` slip** (R2): inconsistent with the rest of the series's $C_{\text{test}}$.
- **Paper 5 `\deepencite` misuse for known cross-paper section refs** (R2): a few `\deepencite{}` markers are pointing at content that's actually in Paper 4 — should be inline-text Paper-N §X mentions instead.
- **Snell `θ*` subscript drift** (R3): minor notation inconsistency between §4 introduction and equation.
- **PUCT formula labeled "UCT"** (R3): the rStar-Math formula uses PUCT (with prior $\pi_\theta$) but Paper 3 §6 calls it UCT.
- **Watermark spike-entropy form** (R3): properly `\deepencite`-flagged but the section claims more about it than the abstract supports.
- **`\begin{speculation}` env defined but never used** (R6): ~12 [SPECULATION] markers in KB notes never landed in the LaTeX as `\begin{speculation}` callouts. Tag-coverage gap.
- **Paper 4 callout density lowest of all five papers** (R6): consider adding 2-3 more intuition/contradiction callouts.
- **Forumsignal undercount** (R6): only 9 across the series; nostalgebraist + Nanda + a few should be tagged in body sections, not just snapshot/14-contradictions.
- **Tier-4 `lindsey2025-circuit-tracing` is single-source methodology** (R4): hedge with a margin note or a forumsignal tag.
- **Paper 3 §13 o-series RLVR cell** (R4): soften from a positive assertion to "claimed in the system card; not externally reproduced."
- (~7 other minors per individual reports)

## NIT (cosmetic, optional)

6 nits across the reports — mostly capitalization, hyphenation, spacing in the
contradiction tables. Not worth a sweep on their own.

## Recommended fix order

1. **B1 alone** — single-section update; immediately publishable.
2. **M1, M3, M4, M5** — sed-class fixes; ~10 minutes total.
3. **M2** — either add the KB excerpt for `rlvr-limits-2025` (opens Phase 2.5 territory) or add the deepencite flag (1 minute). Recommend the deepencite flag in this pass; KB excerpt during the optional Phase 2.5.
4. **M6, M7, M8, M9** — frontier-currency drift, requires WebSearch + bib additions. Single subagent or a focused 30-minute orchestrator session.
5. **M10, M11** — copy-edit pass on the two relevant sections; ~15 minutes.
6. **MINORs** — single batch sweep, ~30 minutes.
7. **NITs** — defer indefinitely or include with the next deepening pass.

## Bottom line

The series is publication-quality at thesis level. The reviewers found
**zero structural issues** and **one factual issue closed by a post-cutoff
publication**. The remaining MAJORs are concentrated, mechanical, and
collectively closeable in under 2 hours of orchestrator time. The MINORs and
NITs are accumulating tax that's worth a single batched sweep but not blocking.

The structural theses of all five papers — five-axis convergence (P1),
six-stage pipeline (P2), reasoning triangle (P3), method-pluralism (P4),
layered defense (P5) — survive intact across all six review lenses.

## Resolution (verified 2026-08-17)

Status of each BLOCKER / MAJOR against the current `theory/series/` sources.
Line numbers are the evidence found on re-verification, not the original fix
commits. "Not re-verified" means the finding was not checked in this pass — it
is not a claim that the fix is missing.

| ID | Status | Evidence |
|---|---|---|
| B1 | **APPLIED** | Decontamination / Qwen2-7B-control update paragraph at `paper-3/sections/14-contradictions.tex:176-192`, citing the new key `deepseek-r1-nature-2025` (`kb/index/papers.json:3872`, `series/references.bib:408`). |
| M1 | **APPLIED** | The AIME figure reads $15.6\%\!\to\!77.9\%$ at `paper-3/sections/14-contradictions.tex:87` and `:152`; the corrected pair also appears at `03-self-consistency.tex:173`, `05-process-reward-models.tex:304`, `10-search-vs-rl.tex:128,157,162`. No `71\%` variant remains. |
| M2 | **APPLIED** (flag option) | `\deepencite{rlvr-limits-2025 §3-§5, KB-excerpt-pending}` at `paper-2/sections/14-contradictions.tex:278` and `paper-3/sections/14-contradictions.tex:89`. The KB excerpt itself is still pending, which is what the finding's recommended option allowed. |
| M3 | **APPLIED** | `\deepencite{sadasivan2023-detection §main-result, citation-pending}` at `paper-5/sections/14-contradictions.tex:145`, matching the §12 usage at `12-watermarking.tex:285`. |
| M4 | **APPLIED** | All three sites cite `\citet[\S 3, Eq.~2]{hoffmann2022-chinchilla}`: `paper-3/sections/04-inference-compute-scaling.tex:328` and `:398`, `paper-3/sections/10-search-vs-rl.tex:205`. No `Eq.~10` occurrence remains in Paper 3. |
| M5 | **APPLIED** | The token-level GRPO objective is cited to `\citep[\S 4.1, Eq.\ 3]{shao2024}` at `paper-3/sections/07-rlvr-grpo-dapo.tex:102`, with `deepseek-r1` demoted to the sequence-level restatement at `:103-106`. |
| M6 | **APPLIED** | "Caveat: closed-vendor row labels age faster than the architectural picture" paragraph at `paper-1/sections/13-frontier-snapshot.tex:225-250`, naming the Claude 4-family, GPT-5/5.x, and Gemini 2.5 Deep Think releases and date-stamping the retained 2025-Q1 labels. |
| M7 | **APPLIED** | Mistral Large 3 MoE note (675B total / 41B active) at `paper-1/sections/13-frontier-snapshot.tex:140-143`, citing `mistral-large3-2025` and stating why the Mistral Large 2 row is retained. |
| M8 | **APPLIED** | "Update (post-2025-09): deliberative alignment as a partial sharpener" at `paper-5/sections/10-scheming.tex:262-270`, plus the cite at `paper-5/sections/14-contradictions.tex:109`; new bib key `openai2025-deliberative-alignment` at `series/references.bib:1289`. |
| M9 | **APPLIED in substance; location differs from the report** | The one-sided framing is gone from the contradiction's home section: `paper-3/sections/14-contradictions.tex:97-105` reads "The 2026 picture is contested rather than settled … one camp's first-order [reading]" and cites 2025-Q3+ counter-evidence (`wen2025-cot-passk`). The report located this at Paper 3 §11, which is `11-cot-faithfulness.tex` and carries no Yue framing at all. The "consensus leans toward Yue with caveats" sentence still stands at `paper-2/sections/11-rlvr-and-grpo.tex:409` and `paper-2/sections/14-contradictions.tex:286` — both caveated and both forward-pointing to Paper 3, but treat them as the open remainder of M9. |
| M10 | **APPLIED** | Explicit tier-B hedge in the `\deepencite{}` at `paper-1/sections/13-frontier-snapshot.tex:242-245`: closed-vendor system cards "are tier-B per `theory/sources/README.md` and back the version-lineage / API-surface claims, not the architectural cells; those cells remain 'ND.'" |
| M11 | **APPLIED** | The intuition box at `paper-3/sections/07-rlvr-grpo-dapo.tex:168-191` gates the claim: "Under the \citet{rlvr-limits-2025} reading discussed in \cref{sec:contradictions} … a claim contested by the R1 narrative … and not yet decisively settled." |

**MINORs and NITs: not re-verified in this pass.** The 19 MINOR and 6 NIT
findings were not individually checked against the current sources; their
status is unknown — neither confirmed open nor confirmed applied.
