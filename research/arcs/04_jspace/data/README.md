# jspace arc — data

> ## ⚠ The C4-en corpora are REDACTED (2026-07-29)
>
> `fitting_prompts_c4en_n1000.json` and `heldout_prompts_c4en_n30.json` are
> **not** byte-identical to the text the committed artifacts were computed on.
> **120 pieces of third-party personal data were removed** from them:
>
> | Class | fitting (n=1000) | held-out (n=30) |
> |---|---|---|
> | Email addresses | 27 | 0 |
> | Phone numbers | 42 | 2 |
> | Street addresses | 38 | 1 |
> | Postal codes (state+ZIP) | 9 | 1 |
> | **Documents modified** | **59 of 1000** | **3 of 30** |
>
> **Why.** C4 is Common Crawl — the open web. Its cleaning pipeline filters for
> quality (boilerplate, length, offensive content); it does **not** filter
> personal data `[raffel2020-t5 §2.2]`. A uniform 1000-document sample
> therefore carries real people's contact details at the corpus base rate
> (~20 emails / ~50 phones per 1000 docs — Elazar et al. 2024, *What's In My
> Big Data?*, ICLR, Table 5); the counts above are consistent with that rate,
> so this is the expected draw, not an unlucky one. Several documents carried a
> named individual together with a direct email and phone. Republishing that is
> not something the upstream ODC-BY licence can authorise: data-subject rights
> attach to the person, not the licensor. Removed on scientific-integrity and
> ethical grounds, independent of any liability question.
>
> **Consequence for the data.** Redaction replaces each match with a bracketed
> sentinel (`[EMAIL]`, `[PHONE]`, `[STREET-ADDRESS]`, `[POSTAL-CODE]`). It is
> **not length-preserving**, so a redacted document tokenises differently from
> its original. Any lens re-fit on this corpus will differ slightly from the
> committed C4-derived artifacts, which were fit **before** redaction. The
> affected results are listed in the warning at the top of
> [`../README.md`](../README.md) and are scheduled for re-run.
>
> **Scope.** Only the two C4-en files are redacted. The **wikitext-103 corpora
> — the arc's primary fitting corpora — are unmodified**, and every result on
> the primary path is therefore unaffected. wikitext was scanned; its only two
> matches are the same encyclopedic biography of Mortimer Wheeler (archaeologist,
> 1890–1976) giving his 1908 and 1950s London residences — historical fact about
> a long-deceased public figure, no living person and no contact channel, so not
> personal data. That judgment is recorded explicitly as `REVIEWED_NOT_PII` in
> `examples/jspace_redact_corpus.py` rather than hidden by loosening a regex.
>
> **Known limit.** The redactor removes contact *channels*, not personal
> *names*. General name detection over web text has a false-positive rate that
> would gut the corpus ("Washington", "Brooks", "Reed" are places, surnames and
> common nouns alike), and a bare name in web prose is not a contact channel.
> Removing the channel breaks the linkage that makes the combination
> identifying. Stated so readers can judge the residual rather than assume
> completeness.
>
> ### Reproducing the redacted corpora
>
> Deterministic — no randomness, no model. The raw slice is regenerable from
> the public source, and the redaction is pure text substitution:
>
> ```bash
> # 1. regenerate the raw seeded slice from HF allenai/c4 (config `en`, split
> #    `train`, streaming shuffle seed=42 buffer=10000, keep len>=600, first N).
> #    Recipe also pinned in each corpus JSON's own source/selection fields.
> python examples/jspace_freeze_c4_corpus.py                 # n=1000 fitting
> python examples/jspace_freeze_c4_corpus.py --offset 1000   # n=30 held-out
>
> # 2. apply the identical redaction pass
> python examples/jspace_redact_corpus.py --apply <file.json>
>
> # 3. confirm you reproduced the committed bytes
> python examples/jspace_data_manifest.py --check
> ```
>
> Each redacted file also carries an in-file `redaction` block recording the
> script version, per-class counts, and the **pre-redaction sha256**, so the
> transformation is auditable from the artifact alone. Verify no PII remains
> with `python examples/jspace_redact_corpus.py --check <file.json>`.
>
> Data licensing and attribution for these corpora: [`LICENSE-DATA.md`](LICENSE-DATA.md).

Raw and derived `.pt`/JSON artifacts for the J-lens/J-space replication on
Qwen2.5-7B-Instruct (and the Qwen2.5-1.5B-Instruct bf16 primary/control),
tracked via Git LFS with per-file sha256 + provenance in `MANIFEST.json`
(written/verified by `examples/jspace_data_manifest.py`; `--check` is the
drift detector).

Artifact classes (see the MANIFEST for the per-file registry):

- `raw`: frozen fitting/held-out corpora (wikitext-103 + seeded C4-en),
  fitted-lens **layer subsets** (`jlens_*_layer-subset.pt` + `.config.json`
  sidecars; design Decision 4 — the full 27-layer lenses stay in the
  gitignored `cache/`, regenerable via `examples/jspace_fit_lens.py`), and
  the hand-written paper-verbatim item bank.
- `derived`: the promoted metric/scan/swap products the audit
  (`examples/jspace_audit_findings.py`) re-derives from — lens_eval,
  readout_scan, structure_scan, verbal_report, entailed_swap (+ paper-verbatim
  probes), nla_crosstie, and the issue-#26 metric-correction set
  (`paper_metric_varfrac_*` ×8 incl. the four robustness axes + 7B
  held-out, `atom_norm_bias_*` ×2) — so checks B–M and every committed
  figure reproduce from a clean clone after `git lfs pull`.

`cache/` is a byte-identical, gitignored working mirror (plus the full
lenses); render scripts and the audit resolve `data/`-first,
`cache/`-fallback via `examples/_jspace_paths.resolve`.
