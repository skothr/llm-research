# theory/sources/

Primary sources for the KB.

## Layout

- `papers/` — PDF copies of canonical papers, named `{paper-key}_{slug}.pdf`.
  Registered in `kb/index/papers.json`. Source of truth.
- `forums/` — selectively archived blog/forum/thread snapshots. Used as
  discovery-path provenance, **never** as the sole backing for a hard claim.

## Source tiers

| Tier | Sources | Can solely back hard claims? | Storage |
|------|---------|----------------------------|---------|
| **A — Canonical** | arxiv preprints, peer-reviewed venues, official tech reports / model cards (Llama, Qwen, DeepSeek, GPT-4 system card, Claude/Gemini model cards), reference github repos | **Yes** | `papers/<key>_<slug>.pdf` + `kb/excerpts/<key>.md` |
| **B — High-signal commentary** | Vendor research blogs (Anthropic / OpenAI / DeepMind / HF), respected lab blogs (EleutherAI / Databricks-MosaicML / Together / Anyscale), individual respected researchers (Dettmers, Rush, Alammar, Lilian Weng) | **Only if** the underlying tier-A source is also cited | Optional snapshot to `forums/` if uniquely informative |
| **C — Community signal** | r/LocalLLaMA, r/MachineLearning, HuggingFace community, X/Twitter (named researchers preferred), HN comment threads | **No** — discovery only | Selective: only when a thread materially shaped a synthesis. Tagged `[FORUM-SIGNAL]` for context, not authority |

## Forum/blog archival format

When a tier-B or tier-C source materially shapes a KB note, snapshot it
under `forums/`:

```markdown
---
url: https://reddit.com/r/LocalLLaMA/comments/...
captured: YYYY-MM-DD
informed-notes: [<area>/<topic>, ...]
tier: B | C
---

# [Thread/post title]

[author, date]

> [verbatim, blockquoted excerpt]

**Why captured:** Surfaced X claim that I then verified against [paper-key].
The thread is *not* the citation; the paper is. This is here for provenance
of the discovery path.
```

## Adding a new paper

1. **Check redistribution rights before committing the PDF.** Committing a PDF
   here republishes it from this public repo under this repo's name, so the
   licence decision happens *before* `git add`, not at review time.
   - Find the licence on the source page: arXiv shows it per-version on the
     abstract page under "Licence". The arXiv *default* (`arXiv.org perpetual,
     non-exclusive license`) grants arXiv distribution rights only — it does
     **not** grant you redistribution rights. `CC BY 4.0`, `CC BY-SA 4.0`, and
     `CC0` do; `CC BY-NC-*` and `CC BY-ND` add restrictions (no commercial use,
     no derivatives) that make a committed copy risky.
   - Conference/journal PDFs follow the venue: ACL Anthology and JMLR are
     CC BY; NeurIPS/ICML proceedings and publisher-hosted PDFs frequently are
     not. Vendor tech reports and model cards are usually all-rights-reserved
     regardless of being free to download.
   - **If the licence does not permit redistribution, do not commit the PDF.**
     Register the paper in `papers.json` with `local_file: null` and its URL,
     and write the excerpts file from the source instead — short quoted
     passages with attribution are what the excerpts tier is for.
   - Record the decision in the `papers.json` entry: the licence name, whether
     redistribution is permitted, and any obligation it carries (a required
     attribution line, a `ShareAlike` term, a "no derivatives" limit). The
     schema has no dedicated field for this — append a "Licence:" sentence to
     the entry's `summary` (the twelve keys in `papers.json` are fixed: `key`,
     `title`, `authors`, `year`, `venue`, `url`, `local_file`, `excerpts_file`,
     `notes_referenced_by`, `topics`, `category`, `summary`; do not invent
     new ones).
   - This is the `theory/`-side instance of the repo-wide third-party-data
     gate in the root `CLAUDE.md` § "Third-party data — vet BEFORE first use".
2. Download the PDF to `papers/<paper-key>_<slug>.pdf` (slug is a
   filesystem-safe short title). **These are Git LFS objects** —
   `theory/sources/papers/*.pdf` is an LFS rule in `.gitattributes`, so
   `git lfs install` must have been run in the clone or the file commits as a
   pointer stub.
3. Add an entry to `kb/index/papers.json` with all schema fields. `topics`
   should align with leaves in `kb/index/topics.md`.
4. If the paper is going to be cited from a note, write
   `kb/excerpts/<paper-key>.md` with the relevant verbatim passages and
   set `excerpts_file` in `papers.json`.
5. Run `python3 kb/lint.py` from `theory/` — it checks that every
   `excerpts_file` resolves and that citations against the new key resolve to
   real anchors.
