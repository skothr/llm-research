# Data licensing and attribution — arc 04 corpora

**The repository's `GPL-3.0-only` licence covers source code and original
prose only. It does not apply to the third-party datasets in this directory**,
which remain under their upstream terms as recorded below. The maintainer is
not the rights holder for this data and cannot relicense it.

## C4 (`fitting_prompts_c4en_n1000.json`, `heldout_prompts_c4en_n30.json`)

Derived from **AllenAI C4**, config `en`, split `train` — a cleaned extract of
Common Crawl.

- **Licence:** Open Data Commons Attribution License (ODC-BY) v1.0 —
  <https://opendatacommons.org/licenses/by/1-0/>
- Subject additionally to the **Common Crawl Terms of Use** —
  <https://commoncrawl.org/terms-of-use/>
- **Source dataset:** <https://huggingface.co/datasets/allenai/c4>
- **Attribution** (ODC-BY §4.3 — content sourced from the C4 database):
  Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Matena, M., Zhou,
  Y., Li, W., & Liu, P. J. (2020). *Exploring the Limits of Transfer Learning
  with a Unified Text-to-Text Transformer.* JMLR 21(140).
  <https://arxiv.org/abs/1910.10683>
- **Modifications:** a seeded 1000-document (and 30-document held-out) slice,
  then **PII redaction** — see the disclosure in [`README.md`](README.md).

## WikiText-103 (`fitting_prompts_wikitext103_n1000.json`, `heldout_prompts_wikitext103_n30.json`)

Derived from **WikiText-103** (`wikitext-103-raw-v1`), extracted from verified
Good and Featured articles on Wikipedia.

- **Licence:** Creative Commons Attribution-ShareAlike —
  <https://creativecommons.org/licenses/by-sa/3.0/>
- **Version note, stated honestly:** the upstream dataset card is
  self-inconsistent, tagging `cc-by-sa-3.0` + `gfdl` in its metadata while its
  prose cites CC BY-SA 4.0. The corpus derives from a 2016 Wikipedia
  extraction, when Wikipedia text was CC BY-SA 3.0, so **3.0 is assumed here**
  as the conservative reading. This matters: CC BY-SA **3.0 was never declared
  one-way compatible with GPLv3** (only 4.0 was, 2015-10-08), which is why the
  repository licence is explicitly scoped away from this directory.
- **Source dataset:** <https://huggingface.co/datasets/Salesforce/wikitext>
- **Attribution:** Merity, S., Xiong, C., Bradbury, J., & Socher, R. (2016).
  *Pointer Sentinel Mixture Models.* <https://arxiv.org/abs/1609.07843>
- **Modifications:** a 1000-document (and 30-document held-out) slice by the
  companion repo's selection rule. **Not redacted** — scanned and found to
  contain no personal data (see [`README.md`](README.md)).

## Model-derived artifacts

The `.pt` tensors in this directory are activations, fitted lenses, and metric
products computed from **Qwen2.5-1.5B-Instruct** and **Qwen2.5-7B-Instruct**
(Apache-2.0, <https://huggingface.co/Qwen/Qwen2.5-7B-Instruct>) over the
corpora above. They contain no verbatim corpus text.

## Reporting content

If you are named in, or otherwise identified by, any text in this directory
and want it removed, open an issue at
<https://github.com/skothr/llm-research/issues> or contact the maintainer
via the address on the repository owner's GitHub profile. Removal requests
will be honoured — these are small files and no research purpose requires any
particular document, so nothing here depends on retaining an individual's
data. This follows the recommendation of Dodge et al. (2021), *Documenting
Large Webtext Corpora* (<https://arxiv.org/abs/2104.08758>), that
redistributors of web-scraped corpora provide a contact path for content
reports.
