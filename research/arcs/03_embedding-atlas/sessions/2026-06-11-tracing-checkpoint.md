# Session checkpoint — 2026-06-11, tracing phase mid-flight

Operational snapshot (stale-fast; never load-bearing — see ARC_PROCESS
"Sessions are not findings"). Worktree
`.claude/worktrees/embedding-atlas`, branch `worktree-embedding-atlas`,
pushed through commit 8a2186e7. Audit 61 PASS / 0 FAIL; manifest 13 files.

**State:** T0/T1 (emb_trace_capture.py + emb_trace_analyze.py) and T1.5
(emb_trace_corpus.py + emb_trace_components.py) captured, analyzed,
committed. Headline results in the arc-state memory and the commit
messages of 82c11a11 / 8a2186e7. The four-regime carrier story (L0
interface → L1-2 handoff → stable mid carriers → L27-28 output re-encode)
is NOT yet an observation file — no figures, no audit section 9 yet.

**Next actions (order):**
1. fig16-18 (carrier SV profile vs control; component delta profiles;
   static norm/write maps) + observation
   `2026-06-11-emb-trace-block-through-layers.md` + AUDIT 9 + INVENTORY
   rows + manifest is already current.
2. T2: eager-attention capture (`attn_implementation="eager"`) on list
   probes; P1a-P1e against `plans/2026-06-11-predictions.md`; target heads
   L0H15/L0H10/L0H2/L0H20/L0H21/L18H7 + delimiter-position attention
   census across all heads.
3. T4: runtime ablation of the 21 W_E block dims (zero + mean variants) →
   sink formation (dims 458/2570/1427) + attention-pattern degradation.
   Causal arbiter for H-SINK (currently trending null).

**Gotchas for a fresh session:** run everything with the env vars in the
arc-state memory; pyright CLI via the venv symlink is authoritative; the
W_E/W_U S0 dumps in `.cache/emb_artifacts/` skip the model load for
table-only work; probe corpus lives in emb_trace_capture.PROBES +
emb_trace_corpus.PROBES_EXT (ids must stay unique).
