---
url: https://www.lesswrong.com/posts/zFJ3ZdQwrTWE9jT5S/a-review-of-anthropic-s-global-workspace-paper
captured: 2026-07-18
informed-notes: [interpretability/j-space]
tier: B
---

# A Review of Anthropic's Global Workspace Paper

[Neel Nanda, 2026-07-06]

Critical passages that shaped the "Frontier and open questions" section of the
J-space note. Quotes fetched and captured verbatim 2026-07-18.

On J-lens as a lossy approximation:

> J-Lens is not going to give the true representation of this cognitive
> space... We should expect noise and error when applying J-Lens.

On an alternative explanation for the two-hop reasoning results ("Assessment
of Evidence"):

> It is plausible... that there is a general 'Frenchness' direction, which
> combines with the 'is capital city' direction to give Paris... From this
> perspective, the model isn't really doing multi-hop factual recall.

On steering confounds ("Further Musings"):

> The boring hypothesis is that you're just steering the model to say / not
> say a given token, and when doing sampling, whether or not the model says
> e.g. eval, will significantly affect how likely it is to eval game.

On multilingual confounds ("Assessment"):

> Plausibly the English token unembeddings are just a bit higher norm... which
> essentially makes them higher variance logits, and as we're taking a Top K
> over the logits this biases towards high variance categories.

**Why captured:** The most substantive early critical review of
[gurnee2026-workspace]; its lossy-approximation, feature-composition, and
token-steering counter-hypotheses are carried into the note's open-questions
section and directly shape the replication design in `research/arcs/04_jspace/`.
The review is *not* the citation for any hard claim; the paper is. This is
here for provenance.
