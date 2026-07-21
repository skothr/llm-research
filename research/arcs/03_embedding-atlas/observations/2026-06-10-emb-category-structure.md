# Category structure in W_E: paradigm classes cluster hardest; topics are graded; cross-script neighbors

**Date / context:** 2026-06-10. Model/protocol as in
[2026-06-10-emb-global-geometry.md](2026-06-10-emb-global-geometry.md).
Battery: 690 curated words across 70 classes / 10 supergroups
(`examples/emb_token_battery.py`), resolved to 665 surviving single-token
words = 1,062 anchor-variant rows (leading-space + bare variants); per-class
statistics use the 665 primary rows, classes with >= 5 survivors (54 of 70).
Derivation: `examples/emb_category_stats.py`; locked by AUDIT 5-6.

## Findings

**F-C1. Every qualifying class is internally coherent, but with a strong
hierarchy: closed paradigm sets >> function-word classes > topics >
connotation classes.** Within-class minus between-class mean cosine
(centered space; between baseline ~+0.03):

| tier | classes (gap) |
|---|---|
| paradigm sets | number +0.428, month +0.416, day_of_week +0.400 |
| function words | auxiliary +0.286, pronoun +0.266, preposition +0.252, wh_word +0.238 |
| strong topics | country +0.179, color +0.176, us_city +0.176, family +0.167 |
| typical topics | body_part +0.104, food +0.090, profession +0.080, animal +0.078 |
| weakest | formal +0.052, positive +0.045, case_lower +0.023 |

(All values from `emb_category_stats.pt` (centered space); the first- and
last-tier values are additionally re-derived from raw vectors by AUDIT 5.)

Reading: categories whose members are *mutually substitutable in text*
(digits, month names, weekdays, auxiliaries) share the most direction;
loose semantic fields share less; abstract connotation groupings (positive
valence, formal register) barely cohere as *clusters* — though their
pair-contrasts still carry signal (see the pair-directions observation).
Raw vs centered panels are near-identical (fig5) per the isotropy null.

**F-C2. Contrast-direction connectivity shows a function-word block.** The
layer-0 mirror of the nla-verbalizer arc's discriminant-connectivity figure
(fig7, same centroid-difference formula, no sink removal): most contrast
directions are near-orthogonal (matrix mostly pale), with a clear correlated
block among function-word classes (auxiliary/conjunction/preposition/
pronoun/quantifier/wh_word) and a number/number_word association. Topic
contrast directions are largely mutually orthogonal — at layer 0 each topic
occupies its own direction rather than a shared "topicness" axis.

**F-C3. Nearest neighbors are sharply semantic and cross-script.** Top
cosine neighbors over the full real vocabulary (raw space, self excluded):

```
' France' -> 'France' +0.462, ' france' +0.299, ' Russia' +0.283,
             ' Germany' +0.275, ' Italy' +0.274, ... '法国' +0.267 ...
' Paris'  -> 'Paris' +0.508, ' paris' +0.287, ' Berlin' +0.260,
             '巴黎' +0.258, ' London' +0.254, ' France' +0.248 ...
' dog'    -> ' Dog' +0.406, ' dogs' +0.374, 'Dog' +0.323, 'dog' +0.296 ...
```

The Chinese exonyms 法国 ("France") and 巴黎 ("Paris") rank inside the top
English neighbors: cross-lingual, cross-script alignment exists already in
the static lookup table, before any attention layer runs. Note the absolute
cosines are modest (+0.25-0.5) — "nearest" in this near-isotropic space is
nothing like "identical".

**F-C4. Case/space variants are a token's closest family.** A word's own
orthographic variants (' France'/'France', ' dog'/' Dog'/' dogs') dominate
its neighborhood, ahead of semantic siblings — orthography binds tighter
than meaning at layer 0.

## Evidence

Audit-locked: AUDIT 5 re-derives the gaps in F-C1's table from
`emb_battery_vectors.pt` from first principles (not via the derived
artifact); AUDIT 6 asserts the neighbor identities quoted above (decoded
strings, not just non-emptiness). Full neighbor tables:
`python examples/emb_neighbors_report.py`.

## Reproducibility

```bash
python examples/emb_category_stats.py     # derive (model-free)
python examples/emb_category_render.py    # fig5-fig8
python examples/emb_pca_map_render.py     # fig9
python examples/emb_neighbors_report.py   # neighbor tables
python examples/emb_audit_findings.py     # AUDIT 5-6
```

## Hypotheses / follow-ups

- H1: the paradigm > topic hierarchy mirrors substitutability statistics
  (words interchangeable in context get near-identical gradient updates).
  To test: correlate gap with a corpus substitutability measure.
- H2: the layer-20 bridge — do the SAME classes that are weak here (valence,
  register) become strong at layer 20, where context has been integrated?
  That asymmetry, if present, would locate where "connotation" becomes
  linearly readable. (Planned: `emb_layer20_bridge.py` against arc 1's
  committed `vocab_atlas.pt`.)
- H3: cross-script neighbors (法国/巴黎) suggest a multilingual alignment
  subspace; a dedicated battery of exonym pairs could extract its direction.

## Caveats

- Class sizes 5-25 words; centroid estimates are noisy for small classes.
- Battery words are hand-curated (selection bias toward common, prototypical
  members); gaps are statements about these anchors, not the whole vocab.
- 16 classes (punctuation, math_op, number_multi, 4 xlat concepts) fell
  below the 5-survivor threshold and are excluded from per-class stats.
