# jspace arc — figure inventory

Bijection between committed figures and their render scripts
(ARC_PROCESS step 3). Rows are added here in the same commit as each
figure + its render script. The **Provenance** column links each figure to its
`DATA_PROVENANCE.md` section (exact artifacts, prompt/item sets + text, and a
raw-row dump recipe); each figure also carries an in-image footer naming its
source artifacts.

| Figure | Render script | Data inputs | Provenance | Status |
|--------|--------------|-------------|------------|--------|
| `2026-07-20-jspace-structure-depth-map.png` | `examples/jspace_render_structure_figures.py` | `data/structure_scan_qwen2.5-{1.5b,7b}-instruct_jlens_*_n100.pt` | [§structure-depth-map](DATA_PROVENANCE.md#structure-depth-map) | committed (stage 4; inventory row added late, 2026-07-21) |
| `2026-07-21-jspace-swap-causality.png` | `examples/jspace_render_swap_causality.py` | `data/verbal_report_chat_6c_qwen2.5-{1.5b,7b}-instruct_jlens_*_n100.pt` | [§swap-causality](DATA_PROVENANCE.md#swap-causality) | committed (stage 5) |
| `2026-07-21-jspace-corpus-invariance.png` | `examples/jspace_render_corpus_invariance.py` | `data/structure_scan_qwen2.5-1.5b-instruct_jlens_*_n100{,_c4en}.pt`, `data/lens_eval_qwen2.5-1.5b_bf16_n100{,_c4en}.pt` | [§corpus-invariance](DATA_PROVENANCE.md#corpus-invariance) | committed (corpus check) |
| `2026-07-21-jspace-nla-crosstie.png` | `examples/jspace_render_nla_crosstie.py` | `data/nla_crosstie_qwen2.5-7b-instruct_jlens_qwen2.5-7b_nf4_n100.pt` | [§nla-crosstie](DATA_PROVENANCE.md#nla-crosstie) | committed (stage 6) |
| `2026-07-21-jspace-emergence.png` | `examples/jspace_render_emergence.py` | `data/readout_scan_qwen2.5-{1.5b,7b}-instruct_jlens_*_n100.pt` | [§emergence](DATA_PROVENANCE.md#emergence) | committed (stage 3) |
| `2026-07-21-jspace-unspoken-words.png` | `examples/jspace_render_trajectory.py` | `data/readout_scan_qwen2.5-{1.5b,7b}-instruct_jlens_*_n100.pt` (rich `topk_strs_{j,l,model}`) | [§unspoken-words](DATA_PROVENANCE.md#unspoken-words-trajectory) | committed (rich capture) |
| `2026-07-22-jspace-entailed-property.png` | `examples/jspace_render_entailed.py` | `data/entailed_swap_chat_L{18,21,24}_qwen2.5-1.5b-instruct_jlens_*_n100.pt`, `data/entailed_swap_chat_L{18,19,22}_qwen2.5-7b-instruct_jlens_*_n100.pt` | [§entailed-property](DATA_PROVENANCE.md#entailed-property) | committed (stage 5.2) |
| `2026-07-24-jspace-paper-metric-excess.png` | `examples/jspace_render_paper_metric_figure.py` | `data/paper_metric_varfrac_qwen2.5-{1.5b,7b}-instruct_jlens_*_n100.pt`, `data/paper_metric_varfrac_...bf16_n100_allpos.pt` | [§paper-metric-excess](DATA_PROVENANCE.md#paper-metric-excess) | committed (issue #26 metric correction) |
