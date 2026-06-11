# Figure inventory — embedding-atlas arc

Common assumptions (all figures): model Qwen/Qwen2.5-7B-Instruct, HF revision
`a09a35458c702b33eeacc393d103063234e8bc28`, bf16 weights cast to float32
before any reduction; "real rows" = token ids < 151,665 (tokenizer length),
excluding the 399 zero padded rows; "centered" = global mean mu (over real
rows) subtracted; battery = `examples/emb_token_battery.py` resolved to 1,062
anchor-variant rows / 665 primary (see capture coverage report); fixed seed
20260610 for all sampling. Every figure renders model-free from the committed
`data/*.pt` (cache-first, committed fallback via `_emb_artifacts`).

| fig | what it shows | source script | source data |
|---|---|---|---|
| fig1_norms.png | Row-norm histograms, real vs padded rows; near-zero (dead) row count annotated | emb_global_render.py | emb_global_stats.pt |
| fig2_anisotropy.png | cos(E_i, mu) distribution; 10k random-pair cosine raw vs centered (the isotropy null) | emb_global_render.py | emb_global_stats.pt |
| fig3_pca_spectrum.png | Centered covariance eigenspectrum (loglog) + cumulative variance; participation ratio 1003 | emb_global_render.py | emb_global_stats.pt |
| fig4_e_vs_u.png | Per-token cos(E_i, U_i) over real rows (input/output embedding orthogonality) | emb_global_render.py | emb_global_stats.pt |
| fig5_within_between.png | Per-class within vs between mean cosine, raw + centered panels, supergroup-colored | emb_category_render.py | emb_category_stats.pt |
| fig6_centroid_matrix.png | Class-centroid cosine heatmap (centered), supergroup-ordered | emb_category_render.py | emb_category_stats.pt |
| fig7_contrast_connectivity.png | Contrast-direction connectivity cos(d_a, d_b) (centered) — layer-0 mirror of the nla-verbalizer discriminant-connectivity figure | emb_category_render.py | emb_category_stats.pt |
| fig8_e_vs_u_by_class.png | Per-class boxplots of cos(E_i, U_i) over primary anchors | emb_category_render.py | emb_category_stats.pt + emb_battery_vectors.pt |
| fig9_pca_map.png | Battery anchors on battery-only PCA axes (PC1-2, PC2-3), supergroup-colored, exemplars labeled; CJK labels omitted (matplotlib default font lacks the glyphs) | emb_pca_map_render.py | emb_battery_vectors.pt |
| fig10_pair_directions.png | Per-kind difference-direction consistency vs 200-permutation baseline + per-pair strip plot (worst pair labeled) | emb_pairs_render.py | emb_pair_directions.pt |
| fig11_dim_kurtosis.png | Per-dimension excess-kurtosis spectrum + histogram over 149,706 alive rows (feature-dim candidates) | emb_fullvocab_render.py | emb_fullvocab_stats.pt |
| fig12_dim_corr.png | Dimension-correlation off-diagonal histogram + correlation eigen-spectrum (full population) | emb_fullvocab_render.py | emb_fullvocab_stats.pt |
| fig13_knn_cosine.png | Top-1 and 32nd-neighbor cosine distributions over the full k-NN graph (unimodal; bimodality trigger did not fire) | emb_fullvocab_render.py | emb_fullvocab_stats.pt |
| fig14_handle_census.png | Out-of-battery membership per battery handle at in-class 10th-percentile thresholds (log scale) | emb_fullvocab_render.py | emb_fullvocab_analysis.pt |
| fig15_structural_block.png | The 21-dim entangled block: energy-vs-token-id deciles vs random 21-dim control + per-script means (head-loaded, script-independent) | emb_structural_block_render.py | emb_structural_block.pt |
| fig16_carrier_sv.png | Carrier analysis: per-layer top singular value of the (21 x 3584) layer-0-block-coords vs layer-L-residual correlation, block vs random-21 control + in-block correlation mass (moves-not-dissolves; 1,130 corpus tokens, positions > 0) | emb_trace_render.py | emb_trace_components.pt |
| fig17_component_deltas.png | Attention vs FFN residual-delta block-energy fraction by layer + static block/control weight ratios (RMSNorm gain, o_proj/down_proj writes, gate/up reads) | emb_trace_render.py | emb_trace_components.pt |
| fig18_sink_census.png | Massive-activation dims (peak vs first layer; arc-1 sink dims marked) + P2 block-energy persistence profile, delimiter vs control vs first position | emb_trace_render.py | emb_trace_components.pt + emb_trace_analysis.pt |
