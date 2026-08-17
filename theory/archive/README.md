# Theory archive

Retired `theory/` material, kept for archaeology. Nothing here is load-bearing
for a current claim; files are preserved because they record what the workspace
was before the 2026-05-03 expansion, not because they are maintained.

## Contents

### `2026-05-03-pre-expansion/` — the v1 single-document deliverable

The whole of `theory/` as it stood on 2026-05-03: one LaTeX/PDF reference,
*LLM Core Architecture*, covering the Transformer (2017) through LLaMA (2023)
at matrix-dimension depth.

| Path | What it is |
|---|---|
| `llm-core-architecture.pdf` | the built v1 document |
| `build/llm-core-architecture/` | its LaTeX source tree |
| `GLOSSARY.md` | the v1 glossary, superseded by `theory/kb/glossary.md` |
| `visuals/llm-architecture-diagram.html` | standalone architecture diagram |
| `Makefile` | the v1 build — **not** the current build path, see below |

**Why it was archived.** The design spec
`theory/docs/design/2026-05-03-theory-expansion-design.md` replaced the
single-document shape with the two-layer structure the workspace uses now: a
citation-grounded knowledge base at `theory/kb/` as the substrate, and a
five-paper LaTeX series at `theory/series/` as the deliverable drawn from it.
The v1 document's scope is a strict subset of Paper 1 (architecture) plus parts
of Paper 2 (training); its prose was partially reused, its structure was not.
The original design spec for v1,
`theory/docs/design/2026-04-06-llm-core-architecture-design.md`, is marked
superseded.

## The `Makefile` here is not the current build path

`2026-05-03-pre-expansion/Makefile` is the only Makefile in this repository,
which makes it a trap: `make` at the repo root or in `theory/` does nothing,
and running it here rebuilds the **archived 2023-era document**, not anything
current.

The current build is a shell script, run from the repo root:

```bash
bash theory/series/build.sh            # clean + build all 5 papers + collect dist/
bash theory/series/build.sh collect    # re-collect dist/ symlinks only
```

See `theory/series/README.md` for what that script does and
`theory/README.md` for how the two layers fit together.
