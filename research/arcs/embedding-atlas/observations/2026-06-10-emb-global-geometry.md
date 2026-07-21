# Global geometry of Qwen2.5-7B's input-embedding table: near-isotropic, flat-spectrum, input/output-decoupled

**Date / context:** 2026-06-10. Model: Qwen/Qwen2.5-7B-Instruct, HF revision
`a09a3545…bc28`, bf16 weights, CPU; all reductions in float32. Matrices:
`W_E = model.get_input_embeddings().weight` and `W_U = model.lm_head.weight`,
both (152064, 3584); `tie_word_embeddings: false` (verified at load). Real
vocabulary rows: 151,665 (tokenizer length); statistics exclude the 399
padded rows. Capture: `examples/emb_capture.py`; numbers locked by
`examples/emb_audit_findings.py` (AUDIT 1-4).

## Findings

**F-G1 (null with teeth). The table is nearly isotropic — the classic
"anisotropic narrow cone" expectation fails here.** Random-pair cosine over
10k sampled real-token pairs: **+0.0097 raw** (+0.0007 after mean-centering).
Mean cosine to the mean vector mu: **+0.0980**. Prior literature on
contextual/static embedding spaces (Mu & Viswanath 2018, arXiv:1702.01417;
Ethayarajh 2019, arXiv:1909.00512) describes random-pair cosines of +0.3-0.9
in GPT-2-era models; this table shows two random tokens essentially
orthogonal. Practical consequence for the whole arc: raw-space and
mean-centered analyses give near-identical results (fig5's two panels), so
anisotropy correction is NOT load-bearing for this model's W_E.

**F-G2. The variance spectrum is remarkably flat.** PC1 explains **1.21%**
of variance; top-10 explain 4.68%; top-50 explain 12.13%. The participation
ratio (effective dimensionality, (sum lambda)^2 / sum lambda^2) is **~1003 of
3584**. No dominant "common direction" exists to remove — consistent with
F-G1. The spectrum (fig3, left) shows a knee near rank ~600 and a sharp drop
near rank ~1700 whose cause is uncharacterized (bf16 quantization floor is a
candidate — [SPECULATION], follow-up below).

**F-G3. ~2k vocabulary rows are dead (exactly/near zero).** 1,959 real-vocab
rows have norm < 1e-3 (visible as the spike at 0 in fig1/fig2-left); the 399
config-padding rows beyond the tokenizer length are exactly zero (norm
median 0.0000). Dead rows are presumably never-trained reserved/unused
tokens. Real-row norm median: 0.859.

**F-G4. Input and output embeddings of the same token are mutually
orthogonal.** Per-token cos(E_i, U_i): mean **+0.0017**, median +0.0014, max
< 0.1, zero tokens above 0.5 (fig4). This is NOT a degenerate-matrix
artifact: U norms are healthy (median 0.697, no near-zero rows), and U-space
carries its own category structure (month-class within-cosine **+0.439** in
U-space, AUDIT 4). Interpretation: with untied matrices the model is free to
choose unrelated coordinates for "reading" vs "writing" a token, and it
fully uses that freedom — the residual stream must rotate between the two
bases across the 28 layers. [INTUITION: E and U are two different
dictionaries for the same words; the layers translate. Canonically: there is
no constraint tying row i of W_E to row i of W_U once tying is disabled, and
empirically cos(E_i, U_i) ~ 0 at chance level.]

## Evidence

```
validation: n_rows=1062 drops=25 padded=399 near_zero_real=1959
            mean|cos_to_mu|=0.1006 rand_cos_raw_mean=0.0097 PC1_var_frac=0.0121
E/U cos: mean +0.0017 median +0.0014 frac>0.5: 0.000
rand raw +0.0097 centered +0.0007
PC1 0.0121 top10 0.0468 top50 0.1213 PR 1003
U norms: median 0.697; U-space month within-cos +0.439
```

Figures: fig1 (norms), fig2 (anisotropy), fig3 (PCA spectrum), fig4 (E vs U);
provenance in [`figures/INVENTORY.md`](figures/INVENTORY.md).

## Reproducibility

```bash
python examples/emb_capture.py          # capture (one model load, ~10 min CPU)
python examples/emb_global_render.py    # fig1-fig4 (model-free)
python examples/emb_audit_findings.py   # locks every number above (AUDIT 1-4)
```

## Hypotheses / open questions

- H1: near-isotropy is a property of modern large-vocab tokenizers + training
  recipes (weight decay on embeddings?) rather than Qwen-specific. To test:
  same capture on TinyLlama-1.1B (cached locally) — cheap.
- H2: the rank-~1700 spectrum cliff reflects bf16's coarse mantissa rather
  than trained structure. To test: re-run the covariance accumulation after
  adding tiny float32 noise, or compare against an fp32-cast-with-dither.
- H3 (E/U): layer-by-layer, the residual stream's overlap with U_i for the
  current token should grow toward late layers (logit-lens-style). Direct
  bridge experiment to the planned rope-vis / trace-back work.

## Caveats

- Random-pair sampling includes the 1,959 dead rows (1.3% of vocab); their
  cosines are meaningless but at this rate shift the mean by < 0.002.
  Follow-up: exclude dead rows from mu/covariance/sampling and re-lock.
- bf16 -> float32 casting precision bounds every cosine at ~1e-3 accuracy.

## References

- Mu & Viswanath 2018, "All-but-the-Top" (arXiv:1702.01417)
- Ethayarajh 2019, "How Contextual are Contextualized Word Representations?"
  (arXiv:1909.00512)
