# jspace figures — data provenance

Per-figure record of the exact artifacts, prompt/item sets, and raw-row dump
recipes behind each committed figure. Companion to `INVENTORY.md` (which maps
each figure to its render script); every `INVENTORY.md` row links here.

**Artifact locations — `data/` vs `data/cache/`.** Every `.pt` below exists in
two places under `research/arcs/jspace/data/`:

- `data/*.pt` — the **committed deliverable** (Git-LFS tracked, sha256-registered
  in `data/MANIFEST.json`). This is the clean-clone source of truth.
- `data/cache/*.pt` — a **byte-identical, gitignored working mirror**. The render
  scripts (`examples/jspace_render_*.py`) set `CACHE = data/cache` and read from
  here. `data/` and `data/cache/` copies are identical (verified `diff`), so a
  clean clone re-renders every figure after `git lfs pull` + copying/symlinking
  `data/*.pt` into `data/cache/` (or pointing a script at `data/`).

All `n=100` in filenames is the fitted **JacobianLens** rank/config label
(`jlens_qwen2.5-{1.5b_bf16,7b_nf4}_n100...pt`), not a prompt count. Lens fitting
corpus: `data/fitting_prompts_wikitext103_n1000.json` (default) /
`data/fitting_prompts_c4en_n1000.json` (C4 variant).

Prompt-set files referenced below (all under `research/arcs/jspace/data/`):

- `heldout_prompts_wikitext103_n30.json` — 30 held-out wikitext-103 records
  (records 1001-1030), keys `{"source","selection","n","fitting_set","prompts"}`.
- `heldout_prompts_c4en_n30.json` — 30 held-out C4-en passages, same schema.

Item banks live in the capture scripts (not JSON):

- `examples/jspace_verbal_report.py` — `CATEGORIES` dict (26 categories).
- `examples/jspace_entailed_swap.py` — `ITEMS` list (33 entailed-property items).
- `examples/jspace_nla_crosstie.py` — `CONCEPT_PROMPTS` list (concept-loaded prompts).

---

## structure-depth-map
`2026-07-20-jspace-structure-depth-map.png` — render: `examples/jspace_render_structure_figures.py`

**Data artifacts** (`data/` committed + `data/cache/` mirror):
- `structure_scan_qwen2.5-1.5b-instruct_jlens_qwen2.5-1.5b_bf16_n100.pt`
- `structure_scan_qwen2.5-7b-instruct_jlens_qwen2.5-7b_nf4_n100.pt`

**Prompt set:** `data/heldout_prompts_wikitext103_n30.json` — all **30** held-out
wikitext prompts (`summary.n_prompts == 30`). Per-prompt rich token data
(decoded top J-space atoms) is inside each artifact under
`per_prompt[i]["top_atom_strs"]`.

Values plotted: `summary.mean_varfrac[25]` (variance fraction at k=25, panel a)
and `summary.mean_readout_kurtosis` (panel b), vs `summary.layers`.

**Dump the raw rows:**
```python
import torch
d = torch.load("research/arcs/jspace/data/structure_scan_qwen2.5-1.5b-instruct_jlens_qwen2.5-1.5b_bf16_n100.pt", weights_only=False)
s = d["summary"]; print(s["layers"], s["mean_varfrac"][25], s["mean_readout_kurtosis"])
```

---

## swap-causality
`2026-07-21-jspace-swap-causality.png` — render: `examples/jspace_render_swap_causality.py`

**Data artifacts** (`data/` committed + `data/cache/` mirror):
- `verbal_report_chat_6c_qwen2.5-1.5b-instruct_jlens_qwen2.5-1.5b_bf16_n100.pt` (layer 21)
- `verbal_report_chat_6c_qwen2.5-7b-instruct_jlens_qwen2.5-7b_nf4_n100.pt` (layer 22)

**Item set:** the `CATEGORIES` bank in `examples/jspace_verbal_report.py`
(26 categories, single-token instances validated against the Qwen2.5 tokenizer;
`summary.n_items == 78` swap items). Task: prompt the instruct model to "think of
a {category}" and report it in one word, then inject the swap direction.
Verbatim category instances (from `CATEGORIES`):
- `sport`: Soccer, Rugby, Tennis, Golf, Boxing, Cricket, …
- `fruit`: Apple, Banana, Orange, Mango, Grape, Cherry, …
- `animal`: Dog, Cat, Horse, Cow, Sheep, Goat, Pig, Lion, …

Values plotted: `summary.metrics["{cond}@{s}"]["target_top5_rate_all"]` per
condition/strength; per-item rows in `per_item` carry
`category / source_word / target_word / conditions`.

**Dump the raw rows:**
```python
import torch
d = torch.load("research/arcs/jspace/data/verbal_report_chat_6c_qwen2.5-1.5b-instruct_jlens_qwen2.5-1.5b_bf16_n100.pt", weights_only=False)
print({k: v["target_top5_rate_all"] for k, v in d["summary"]["metrics"].items()})
print(d["per_item"][0]["category"], d["per_item"][0]["source_word"], d["per_item"][0]["target_word"])
```

---

## corpus-invariance
`2026-07-21-jspace-corpus-invariance.png` — render: `examples/jspace_render_corpus_invariance.py`

**Data artifacts** (`data/` committed + `data/cache/` mirror):
- `structure_scan_qwen2.5-1.5b-instruct_jlens_qwen2.5-1.5b_bf16_n100.pt` (wikitext lens)
- `structure_scan_qwen2.5-1.5b-instruct_jlens_qwen2.5-1.5b_bf16_n100_c4en.pt` (C4 lens)
- `lens_eval_qwen2.5-1.5b_bf16_n100.pt` (wikitext lens)
- `lens_eval_qwen2.5-1.5b_bf16_n100_c4en.pt` (C4 lens)

**Prompt sets:** structure scans run on `data/heldout_prompts_wikitext103_n30.json`
and `data/heldout_prompts_c4en_n30.json` (30 prompts each). The lens itself is fit
on wikitext-103 vs C4-en fitting corpora (the `{,_c4en}` filename suffix). Panel
(b) reads the `lens_eval` multihop eval item set (internal to
`examples/jspace_lens_eval.py`).

Values plotted: panel (a) `summary.mean_varfrac[25]` vs layer, both corpora;
panel (b) `per_eval["multihop"]["rates_j"|"rates_l"][band][10]` per depth band.

**Dump the raw rows:**
```python
import torch
sv = torch.load("research/arcs/jspace/data/structure_scan_qwen2.5-1.5b-instruct_jlens_qwen2.5-1.5b_bf16_n100_c4en.pt", weights_only=False)["summary"]
ev = torch.load("research/arcs/jspace/data/lens_eval_qwen2.5-1.5b_bf16_n100.pt", weights_only=False)["summary"]
print(sv["mean_varfrac"][25]); print(ev["per_eval"]["multihop"]["rates_j"])
```

---

## nla-crosstie
`2026-07-21-jspace-nla-crosstie.png` — render: `examples/jspace_render_nla_crosstie.py`

**Data artifact** (`data/` committed + `data/cache/` mirror):
- `nla_crosstie_qwen2.5-7b-instruct_jlens_qwen2.5-7b_nf4_n100.pt`
  (J-lens source-layer 19 = NLA `hidden_states[20]`).

**Prompt sets** (`summary.n_neutral / n_concept / n_decomp` == 12 / 12 / 12):
- **neutral prose (n=12):** first 12 of `data/heldout_prompts_wikitext103_n30.json`.
- **concept-loaded (n=12):** `CONCEPT_PROMPTS` in `examples/jspace_nla_crosstie.py`.
  Verbatim examples: `"The capital of France is"`, `"The chemical symbol for gold is"`,
  `"Water is made of hydrogen and"`, `"The tallest animal in the world is the"`.
- **decomposition (n=12):** 6 neutral + 6 concept subset of the above.

Per-prompt rows carry the raw text and metrics: `tag`, `prompt`, `nla_rank_median`,
`carrier_component/residual/resctrl`, plus decoded `h_text/c_text/res_text`.

**Dump the raw rows:**
```python
import torch
d = torch.load("research/arcs/jspace/data/nla_crosstie_qwen2.5-7b-instruct_jlens_qwen2.5-7b_nf4_n100.pt", weights_only=False)
print(d["summary"]["metric1_rank_median_concept"], d["summary"]["expB_carrier_delta_mean"])
r = d["per_prompt"][0]; print(r["tag"], repr(r["prompt"]), r.get("nla_rank_median"))
```

---

## emergence
`2026-07-21-jspace-emergence.png` — render: `examples/jspace_render_emergence.py`

**Data artifacts** (`data/` committed + `data/cache/` mirror):
- `readout_scan_qwen2.5-1.5b-instruct_jlens_qwen2.5-1.5b_bf16_n100.pt`
- `readout_scan_qwen2.5-7b-instruct_jlens_qwen2.5-7b_nf4_n100.pt`

**Prompt set:** `data/heldout_prompts_wikitext103_n30.json`, first **12** prompts
× 9 scored positions (`summary.n_prompts == 12`). Per-prompt rich token data
(`topk_ids/probs/strs_{j,l,model}`) is inside each artifact.

Values plotted: panel (a) `summary.layer_mean_spearman_{j,l}_last` vs layer;
panel (b) `summary.depth_of_emergence_top10` (median emergence layer + never-emerged counts).

**Dump the raw rows:**
```python
import torch
s = torch.load("research/arcs/jspace/data/readout_scan_qwen2.5-7b-instruct_jlens_qwen2.5-7b_nf4_n100.pt", weights_only=False)["summary"]
print(s["layer_mean_spearman_j_last"]); print(s["depth_of_emergence_top10"])
```

---

## unspoken-words (trajectory)
`2026-07-21-jspace-unspoken-words.png` — render: `examples/jspace_render_trajectory.py`

**Data artifacts** (`data/` committed + `data/cache/` mirror):
- `readout_scan_qwen2.5-1.5b-instruct_jlens_qwen2.5-1.5b_bf16_n100.pt`
- `readout_scan_qwen2.5-7b-instruct_jlens_qwen2.5-7b_nf4_n100.pt`
  (requires the **rich** `per_prompt[i]["topk_strs_{j,l,model}"]` keys.)

**Prompt set:** `data/heldout_prompts_wikitext103_n30.json`. The figure renders a
single position: **prompt index 0, position index 4 = token 127 of 224**
(mid-context; held concept = critical book reviews). Full text of prompt 0:

> ` " By the end of 1984 , From Time Immemorial had ... received some two hundred [ favorable ] notices ... in the United States . The only ' false ' notes in this crescendoing chorus of praise were the Journal of Palestine Studies , which ran a highly critical review by Bill Farrell ; the small Chicago @-@ based newsweekly In These Times , which published a condensed version of this writer 's findings ; and Alexander Cockburn , who devoted a series of columns in The Nation exposing the hoax . ... The periodicals in which From Time Immemorial had already been favorably reviewed refused to run any critical correspondence ( e.g. The New Republic , The Atlantic Monthly , Commentary ) . Periodicals that had yet to review the book rejected a manuscript on the subject as of little or no consequence ( e.g. The Village Voice , Dissent , The New York Review of Books ) . Not a single national newspaper or columnist contacted found newsworthy that a best @-@ selling , effusively praised ' study ' of the Middle East conflict was a threadbare hoax . " `

Local context at the scored position (the figure's caption snippet):
*"...ran a highly critical review by Bill Farrell ... favorably reviewed ... The New York Review of Books..."* — the held concept the J-lens surfaces mid-depth
(criticism → critiques → allegations → commentary → reviews).

**Dump the raw rows:**
```python
import torch, json
d = torch.load("research/arcs/jspace/data/readout_scan_qwen2.5-1.5b-instruct_jlens_qwen2.5-1.5b_bf16_n100.pt", weights_only=False)
pp = d["per_prompt"][0]; print(pp["positions"][4], pp["seq_len"])
print([row[4] for row in pp["topk_strs_j"]])  # per-layer top-5 J-lens tokens at pos idx 4
```

---

## entailed-property
`2026-07-22-jspace-entailed-property.png` — render: `examples/jspace_render_entailed.py`

**Data artifacts** (`data/` committed + `data/cache/` mirror), 6 files:
- 1.5B: `entailed_swap_chat_L{18,21,24}_qwen2.5-1.5b-instruct_jlens_qwen2.5-1.5b_bf16_n100.pt`
- 7B:   `entailed_swap_chat_L{18,19,22}_qwen2.5-7b-instruct_jlens_qwen2.5-7b_nf4_n100.pt`

**Item set:** the `ITEMS` bank in `examples/jspace_entailed_swap.py`
(`summary.n_items == 33` entailed-property items). Each item swaps an unspoken
concept along the J-lens vector and measures Δlog-p of the entailed property.
Verbatim example items (`ITEMS`):
- spider → ant: *"The number of legs on the creature that spins silky webs to trap flies is"* (8 → 6)
- ant → spider: *"The number of legs on the tiny insect that lives in colonies and marches in lines is"* (6 → 8)
- bee → spider: *"The number of legs on the busy insect that makes honey inside a hive is"* (6 → 8)

Values plotted: mean Δlog-p from `per_item[i]["conditions"]["{cond}@2.0"]["dlogp_swap_answer"]`
over the `baseline_correct` subset (the render script re-derives these and asserts
equality with `summary.metrics` at load — a mismatch aborts).

**Dump the raw rows:**
```python
import torch
d = torch.load("research/arcs/jspace/data/entailed_swap_chat_L18_qwen2.5-1.5b-instruct_jlens_qwen2.5-1.5b_bf16_n100.pt", weights_only=False)
correct = [it for it in d["per_item"] if it["baseline_correct"]]
print([it["conditions"]["jlens@2.0"]["dlogp_swap_answer"] for it in correct][:5])
print(d["summary"]["metrics"]["jlens@2.0"]["mean_dlogp_swap_answer"])
```
