"""Tracing-phase capture (T0+T1+T3 data) — one model load, two artifacts.

T0 — massive-activation census on Qwen2.5-7B (protocol of Sun et al. 2024,
arXiv:2402.17762): per layer, the top-|value| activations with their
(dimension, position, token), the per-layer median |activation|, and counts
above the |a| > 100 and a > 1000x-median criteria. Anchors the downstream
end of the planned W_E-block trace, including the nla-verbalizer arc's
hand-identified layer-20 sink dims.

T1 — "who reads the block": per-layer, per-head column-norm map of the
q/k/v projection weights restricted to the 21 W_E block dims vs a seeded
random 21-dim control, broken out by RoPE frequency band for q/k (HF
rotate_half pairing: band b = head rows (b, b+64) for b in 0..63; v has no
rotary). Identifies candidate structural-reader heads BEFORE any attention
capture.

T3 data — block persistence (prediction P2): per layer, per tracked
position, the residual-stream norm fraction (unsquared, ||h_S||/||h||) inside the 21-dim block
subspace, for delimiter vs control positions; full hidden vectors at
tracked positions for a subset of layers (committed) for later analysis.

Probe corpus: committed inline (PROBES) — comma lists, periods/prose, code,
CJK, newline-structured text. Raw text, NO chat template (mechanism study;
matches Sun et al.'s raw-sequence protocol).

Outputs (cache; promote via emb_data_manifest):
  emb_trace_weightmap.pt   T1 maps
  emb_trace_layers.pt      T0 census + T3 per-layer stats + tracked vectors
"""

from __future__ import annotations

import time
from typing import Any

import torch

from _emb_artifacts import load_artifact, write_artifact
from emb_capture import BASE_ID, REVISION

SEED = 20260611
VEC_LAYERS = (0, 1, 2, 3, 7, 14, 20, 27, 28)  # hidden_states indices (0 = embeddings)
ARC1_SINK_DIMS = (277, 458, 1427, 1627, 2107, 2570, 3110)  # nla arc, layer 20
TOPK_PER_LAYER = 8

# (id, category, text) — delimiter-rich and control texts. Committed data.
PROBES: tuple[tuple[str, str, str], ...] = (
    (
        "list_shopping",
        "comma_list",
        "For the trip we packed bread, cheese, apples, rice, coffee, sugar, salt, and a small tent.",
    ),
    (
        "list_countries",
        "comma_list",
        "The treaty was signed by France, Germany, Japan, Brazil, Canada, Spain, Norway, and Greece.",
    ),
    (
        "list_mixed",
        "comma_list",
        "Her toolkit held a hammer, two screwdrivers, a roll of tape, copper wire, pliers, and gloves.",
    ),
    (
        "list_long",
        "comma_list",
        "The menu listed soup, salad, pasta, pizza, grilled fish, roast chicken, rice bowls, dumplings, pancakes, and pie.",
    ),
    (
        "prose_two_sent",
        "prose",
        "The river ran low after the long summer. Farmers watched the sky and waited for rain.",
    ),
    (
        "prose_clauses",
        "prose",
        "Although the library was crowded, she found a quiet corner, opened her notebook, and began to write.",
    ),
    (
        "prose_plain",
        "prose",
        "The engineer checked the bridge supports carefully before the morning traffic arrived.",
    ),
    (
        "code_python",
        "code",
        "def total(xs):\n    s = 0\n    for x in xs:\n        s += x\n    return s\n",
    ),
    (
        "code_call",
        "code",
        'result = process(data, mode="fast", retries=3, verbose=True)\n',
    ),
    ("cjk_list", "cjk", "我们买了苹果、面包、奶酪、咖啡和茶。今天天气很好。"),
    ("cjk_prose", "cjk", "河水在长夏之后变浅了。农民们望着天空，等待下雨。"),
    (
        "newline_items",
        "newline",
        "Tasks:\n- water the plants\n- fix the gate\n- call the dentist\n- buy stamps\n",
    ),
    (
        "numbers_inline",
        "numbers",
        "The samples weighed 3, 7, 12, 28, and 45 grams respectively.",
    ),
    (
        "qa_short",
        "prose",
        "What is the capital of France? The capital of France is Paris.",
    ),
)

DELIM_STRINGS = {",", ".", ";", ":", "、", "，", "。", "\n", " ,", " ."}


def rope_band_rows(head_dim: int) -> list[tuple[int, int]]:
    """HF rotate_half pairing: band b pairs rows (b, b + head_dim//2)."""
    half = head_dim // 2
    return [(b, b + half) for b in range(half)]


def main() -> None:
    from llm_surgeon import surgery

    an = load_artifact("emb_fullvocab_analysis.pt")
    block_dims = torch.tensor(
        sorted(sorted(an["corr_blocks"], key=lambda b: -b["size"])[0]["dims"]),
        dtype=torch.long,
    )
    nb = int(block_dims.numel())
    g = torch.Generator().manual_seed(SEED)
    pool = torch.tensor(
        [i for i in range(3584) if i not in set(block_dims.tolist())],
        dtype=torch.long,
    )
    control_dims = pool[torch.randperm(pool.numel(), generator=g)[:nb]]

    print(f"loading {BASE_ID} @ {REVISION[:8]} (CPU bf16) ...")
    t0 = time.time()
    model, tok = surgery.load_model(
        BASE_ID, mode="bf16", device_map="cpu", revision=REVISION
    )
    print(f"  loaded in {time.time() - t0:.0f}s")
    cfg = model.config
    n_layers: int = cfg.num_hidden_layers
    n_q_heads: int = cfg.num_attention_heads
    n_kv_heads: int = cfg.num_key_value_heads
    head_dim: int = cfg.hidden_size // n_q_heads
    bands = rope_band_rows(head_dim)
    n_bands = len(bands)

    # ---- T1: weight map -------------------------------------------------------
    print("T1: q/k/v block-column read map ...")
    # q/k: (layers, heads, bands) norm over block cols, control cols, all cols
    maps: dict[str, torch.Tensor] = {}
    for proj, n_heads in (("q", n_q_heads), ("k", n_kv_heads), ("v", n_kv_heads)):
        maps[f"{proj}_block"] = torch.zeros(n_layers, n_heads, n_bands)
        maps[f"{proj}_control"] = torch.zeros(n_layers, n_heads, n_bands)
        maps[f"{proj}_all"] = torch.zeros(n_layers, n_heads, n_bands)
    for li, layer in enumerate(model.model.layers):
        for proj, n_heads in (("q", n_q_heads), ("k", n_kv_heads), ("v", n_kv_heads)):
            W = (
                getattr(layer.self_attn, f"{proj}_proj")
                .weight.detach()
                .to(torch.float32)
            )  # (n_heads*head_dim, 3584)
            for h in range(n_heads):
                Wh = W[h * head_dim : (h + 1) * head_dim]  # (head_dim, 3584)
                for bi, (r1, r2) in enumerate(bands):
                    rows = Wh[[r1, r2]]  # (2, 3584) — band b's two rows (v: pairing
                    #                       meaningless but kept for uniform shape)
                    maps[f"{proj}_block"][li, h, bi] = rows[:, block_dims].norm()
                    maps[f"{proj}_control"][li, h, bi] = rows[:, control_dims].norm()
                    maps[f"{proj}_all"][li, h, bi] = rows.norm()
    weightmap: dict[str, Any] = {
        **maps,
        "block_dims": block_dims,
        "control_dims": control_dims,
        "n_layers": n_layers,
        "n_q_heads": n_q_heads,
        "n_kv_heads": n_kv_heads,
        "head_dim": head_dim,
        "rope_pairing": "rotate_half: band b = rows (b, b+64)",
        "base_id": BASE_ID,
        "revision": REVISION,
        "seed": SEED,
    }
    torch.save(weightmap, write_artifact("emb_trace_weightmap.pt"))
    ratio = (maps["q_block"].sum() / maps["q_control"].sum()).item()
    print(f"  q block/control total-norm ratio: {ratio:.3f}")

    # ---- T0 + T3: forward passes ----------------------------------------------
    print("T0/T3: probe-corpus forward passes ...")
    W_E = model.get_input_embeddings().weight.detach()
    mu = W_E[: len(tok)].to(torch.float32).mean(dim=0)  # probe-consistent centering
    block_set = set(block_dims.tolist())
    prompts_out: list[dict[str, Any]] = []
    census: list[dict[str, Any]] = []
    for pid, cat, text in PROBES:
        enc = tok(text, return_tensors="pt")
        ids = enc.input_ids[0]
        toks = [tok.decode([int(i)]) for i in ids]
        n_pos = len(toks)
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True, use_cache=False)
        hs = out.hidden_states  # tuple len n_layers+1, each (1, n_pos, 3584)
        delim_pos = [i for i, s in enumerate(toks) if s in DELIM_STRINGS]
        # controls: non-delimiter, non-first positions
        ctrl_pos = [i for i in range(1, n_pos) if i not in delim_pos]

        L = len(hs)
        block_frac = torch.zeros(L, n_pos)
        norms = torch.zeros(L, n_pos)
        sink_vals = torch.zeros(L, n_pos, len(ARC1_SINK_DIMS))
        for li in range(L):
            h = hs[li][0].to(torch.float32)  # (n_pos, 3584)
            norms[li] = h.norm(dim=1)
            block_frac[li] = h[:, block_dims].norm(dim=1) / h.norm(dim=1).clamp_min(
                1e-12
            )
            sink_vals[li] = h[:, list(ARC1_SINK_DIMS)]
            med = h.abs().median()
            top = torch.topk(h.abs().flatten(), TOPK_PER_LAYER)
            entries = []
            for v, flat in zip(top.values, top.indices):
                pos, dim = int(flat // 3584), int(flat % 3584)
                entries.append(
                    {
                        "value": float(h[pos, dim]),
                        "dim": dim,
                        "pos": pos,
                        "token": toks[pos],
                        "in_block": dim in block_set,
                    }
                )
            census.append(
                {
                    "prompt": pid,
                    "layer": li,  # 0 = embeddings, li = after layer li
                    "median_abs": float(med),
                    "n_above_100": int((h.abs() > 100).sum()),
                    "n_above_1000x_med": int((h.abs() > 1000 * med).sum()),
                    "top": entries,
                }
            )
        tracked = sorted(set(delim_pos) | set(ctrl_pos[:6]) | {0})
        vecs = torch.stack(
            [hs[li][0, tracked].to(torch.float16) for li in VEC_LAYERS]
        )  # (len(VEC_LAYERS), n_tracked, 3584)
        prompts_out.append(
            {
                "id": pid,
                "category": cat,
                "text": text,
                "tokens": toks,
                "delim_positions": delim_pos,
                "control_positions": ctrl_pos,
                "tracked_positions": tracked,
                "block_frac": block_frac,  # (L, n_pos) fp32
                "norms": norms,
                "sink_vals": sink_vals,
                "tracked_vecs": vecs,
            }
        )
        bf_d = float(block_frac[1:, delim_pos].mean()) if delim_pos else float("nan")
        bf_c = float(block_frac[1:, ctrl_pos].mean())
        print(
            f"  {pid:15s} ({n_pos:3d} tok, {len(delim_pos):2d} delim) "
            f"block-frac delim {bf_d:.4f} vs ctrl {bf_c:.4f}; "
            f"max|h| {float(norms.max()):.0f}"
        )

    layers_out: dict[str, Any] = {
        "prompts": prompts_out,
        "census": census,
        "vec_layers": list(VEC_LAYERS),
        "arc1_sink_dims": list(ARC1_SINK_DIMS),
        "block_dims": block_dims,
        "control_dims": control_dims,
        "mu": mu,
        "delim_strings": sorted(DELIM_STRINGS),
        "protocol": "raw text, no chat template, prefill only, bf16 CPU, fp32 stats",
        "base_id": BASE_ID,
        "revision": REVISION,
        "seed": SEED,
    }
    torch.save(layers_out, write_artifact("emb_trace_layers.pt"))
    print("wrote emb_trace_weightmap.pt, emb_trace_layers.pt")

    # validation gates
    assert not torch.isnan(maps["q_block"]).any()
    glob_max = max(float(p["norms"].max()) for p in prompts_out)
    n_mass = sum(c["n_above_100"] for c in census)
    print(
        f"validation: layers={n_layers} bands={n_bands} prompts={len(prompts_out)} "
        f"global max|h|-norm {glob_max:.0f}, total activations |a|>100: {n_mass}"
    )


if __name__ == "__main__":
    main()
