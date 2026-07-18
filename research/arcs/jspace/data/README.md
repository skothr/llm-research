# jspace arc — data

Raw and derived `.pt` artifacts for the J-lens/J-space replication on
Qwen2.5-7B-Instruct (and the Qwen2.5-1.5B-Instruct bf16 control), tracked via
Git LFS with checksums in `MANIFEST.json` (created with the first capture —
none exist yet; capture is gated on design-plan sign-off, see
`../plans/2026-07-18-jspace-design.md`).

Planned artifact classes (per that plan §2):

- `raw`: fitted lens tensors `jlens_{model}_{L}.pt`, fitting config, corpus
  manifest (seed + sample indices).
- `derived`: split-half stability metrics, per-layer variance/sparsity
  profiles, swap-experiment trial tables.

Which layers' lens tensors are committed vs cache-only is decision point 4 of
the design plan.
