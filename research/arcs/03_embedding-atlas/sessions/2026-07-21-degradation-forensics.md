# Session note — model-degradation exposure forensics (2026-07-21)

**Context.** The arc's kickoff session (2026-06-10 10:40 → 06-11 12:32 UTC;
battery probes, full-vocab sweep, F-V1 block discovery, start of tracing)
ran entirely on `claude-fable-5`, within days of that model's release.
Question examined, at the user's direction: if the vendor did silently
degrade output quality on frontier-AI-development prompts during the
model's initial-release window, did that leave artifacts of degraded
judgment in this arc's findings?

**Provenance of the exposure hypothesis — read this before citing it.**
The hypothesis entered this repo through the *user's recollection*, given
in-session on 2026-07-21 while correcting this session's account of the
vendor timeline. It is **not** independently sourced here to tier A, and the
sourcing splits three ways:

- **Tier A, confirmed.** Claude Fable 5 and Mythos 5 were released
  Tuesday 2026-06-09; the released safeguard routes flagged requests
  (cybersecurity, biology/chemistry, distillation) to Claude Opus 4.8 and
  states "users will be informed whenever this occurs"
  ([Anthropic, *Claude Fable 5 and Claude Mythos 5*](https://www.anthropic.com/news/claude-fable-5-mythos-5)).
  A US government export-control directive was received 2026-06-12 and
  both models were pulled
  ([Anthropic, *Statement on the US government directive…*](https://www.anthropic.com/news/fable-mythos-access));
  controls were lifted 2026-06-30 and access was restored 2026-07-01
  ([Anthropic, *Redeploying Claude Fable 5*](https://www.anthropic.com/news/redeploying-fable-5)).
  This corrects an earlier version of this note, which gave the suspension
  as "June 12-30".
- **Not tier-A-backed.** The claim that the *initial* release silently
  degraded quality on frontier-AI-development prompts, and that this was
  reversed to the visible fallback after ~48 h of public criticism, is
  **not** documented on Anthropic's release page as it currently stands —
  that page describes only the visible-fallback policy and carries no
  changelog of an earlier clause. Under this repo's source-tier rule a
  tier-B/C source cannot solely back a hard claim, so this one is recorded
  as unconfirmed, not asserted.
- **Tier B/C corroboration only.** Contemporaneous commentary
  ([Clawd.rip incident log](https://clawd.rip/events/fable-5-research-sabotage/),
  [Developers Digest](https://www.developersdigest.tech/blog/fable-5-silent-guardrails-trust-problem))
  describes a silent-degradation clause in the originally-published system
  card covering frontier LLM pretraining, distributed training
  infrastructure, and ML accelerator design at ~0.03% of traffic, reversed
  within about two days. Discovery signal only.

**This does not weaken the audit below.** The exposure hypothesis was the
audit's *motivation*, never a premise of its method: the audit compares the
kickoff session against a same-arc baseline session blind, and re-derives
the numeric layer from committed artifacts. A negative result stands whether
or not the vendor window was real, and the same audit would have detected
degradation from any cause.

**Method (2026-07-21).** Run as an empirical test at the user's demand,
after a reassuring argument that the findings were probably fine was
refused; the separation of the deterministic numeric layer
(artifact-re-derived) from the interpretive-prose layer is the
human-directed research-integrity call this audit rests on.
19-agent blinded audit: per-chunk judgment-quality
review of the window session AND a post-fix baseline session (2026-07-15
arc-wrap, same arc/work type, also Fable) with auditors blind to which was
which; a targeted detector for reluctance/shallowness on frontier-ML subject
matter; every window finding adversarially verified by an
alternative-explanation skeptic plus a ground-truth check against the repo's
artifacts and audit.

**Result: no credible degradation artifact.**

- Blinded scores (window vs baseline): coherence 5.0/5.0, correction-
  responsiveness 5.0/4.67, technical care 4.5/4.33, directness 5.0/5.0,
  thoroughness 4.83/4.67 — the window session scored equal or higher.
- 6 raw window flags → 3 genuine after verification, all minor, all in one
  conversational summary (over-sharpened phrasing: "decays monotonically",
  "~2x the control floor", "FFN ~1.00 at every layer"). None reached
  committed docs — the locked observations state the correct nuanced
  versions, one via the session's own audit catching the slip. The baseline
  session showed 2 comparable minor flags: indistinguishable defect rates.
- 2 flags were the auditors' own errors, refuted by ground-truth
  re-derivation (a √-vs-square baseline confusion — which surfaced the
  norm-fraction terminology fix — and a false 3.28-conflation claim: both
  3.28 values recompute independently from `emb_trace_components.pt`).
- ML-reluctance detector: zero signals.

**Standing defenses that bounded the exposure:** every load-bearing number
is re-derived from committed artifacts by `emb_audit_findings.py`
(94 PASS / 0 FAIL as this note was written; 99 PASS / 0 FAIL after the
2026-08-17 coverage extension), so silent model-output degradation had no
path into the
numeric layer; only conversational prose was exposed, and its slips were
caught by the audit-before-lock discipline. This is the generalizable part
of the result and it does not depend on the vendor-timeline question:
artifact-re-derived numbers are inert to *any* degradation of the agent's
prose, from any cause.
