# Session note — Fable silent-degradation window forensics (2026-07-21)

**Context.** The arc's kickoff session (2026-06-10 10:40 → 06-11 12:32 UTC;
battery probes, full-vocab sweep, F-V1 block discovery, start of tracing)
ran entirely on `claude-fable-5` — inside the ~48 h window (June 9-11, 2026)
in which Anthropic's initial Fable 5 release silently degraded output
quality on frontier-AI-development prompts (system-card disclosure; reversed
to a visible Opus-4.8 fallback on June 11 after public criticism; separate
June 12-30 export-control suspension). Question examined, at the user's
direction: did that window leave artifacts of degraded judgment in this
arc's findings? The exposure was identified by the user correcting this
session's account of the vendor timeline — the silent degradation was
documented only for the initial-release window, closed by the
visible-fallback policy after public criticism — which is what dated the
kickoff sessions inside it.

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
(94 PASS / 0 FAIL), so silent model-output degradation had no path into the
numeric layer; only conversational prose was exposed, and its slips were
caught by the audit-before-lock discipline.
