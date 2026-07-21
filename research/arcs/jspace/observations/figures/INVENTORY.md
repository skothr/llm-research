# jspace arc — figure inventory

Bijection between committed figures and their render scripts
(ARC_PROCESS step 3). Rows are added here in the same commit as each
figure + its render script.

| Figure | Render script | Data inputs | Status |
|--------|--------------|-------------|--------|
| `2026-07-20-jspace-structure-depth-map.png` | `examples/jspace_render_structure_figures.py` | `data/cache/structure_scan_qwen2.5-{1.5b,7b}-instruct_jlens_*_n100.pt` | committed (stage 4; inventory row added late, 2026-07-21) |
| `2026-07-21-jspace-swap-causality.png` | `examples/jspace_render_swap_causality.py` | `data/cache/verbal_report_chat_6c_qwen2.5-{1.5b,7b}-instruct_jlens_*_n100.pt` | committed (stage 5) |
| `2026-07-21-jspace-corpus-invariance.png` | `examples/jspace_render_corpus_invariance.py` | `data/cache/structure_scan_qwen2.5-1.5b-instruct_jlens_*_n100{,_c4en}.pt`, `data/cache/lens_eval_qwen2.5-1.5b_bf16_n100{,_c4en}.pt` | committed (corpus check) |
| `2026-07-21-jspace-nla-crosstie.png` | `examples/jspace_render_nla_crosstie.py` | `data/cache/nla_crosstie_qwen2.5-7b-instruct_jlens_qwen2.5-7b_nf4_n100.pt` | committed (stage 6) |
| `2026-07-21-jspace-emergence.png` | `examples/jspace_render_emergence.py` | `data/cache/readout_scan_qwen2.5-{1.5b,7b}-instruct_jlens_*_n100.pt` | committed (stage 3) |
