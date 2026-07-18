# Arc: jspace — replicating the J-lens / J-space on Qwen2.5-7B-Instruct

**Research question:** Does the J-space phenomenon reported for Claude-family
models `[gurnee2026-workspace]` — a sparse, low-variance, causally privileged
band of verbalizable representations — replicate on `Qwen/Qwen2.5-7B-Instruct`,
and how do J-lens readouts at layer 20 relate to the NLA verbalizer readouts
studied in `research/arcs/nla-verbalizer/`?

**Status:** design stage. The informed design/test plan is at
`plans/2026-07-18-jspace-design.md`, awaiting human review before capture
begins (ARC_PROCESS step 0 → 1 gate).

**Theory grounding:** `theory/kb/notes/interpretability/j-space.md`,
excerpts in `theory/kb/excerpts/gurnee2026-workspace.md`, archived paper PDF
in `theory/sources/papers/gurnee2026-workspace_verbalizable-global-workspace.pdf`.

**Attribution:** research direction (target model, replication goal) — human;
paper digestion, KB grounding, experiment design — Claude (Fable 5 session
2026-07-18, with opus/sonnet subagents).

Findings, limitations, and next paths will be synthesized here at arc close
per `research/ARC_PROCESS.md` step 6.
