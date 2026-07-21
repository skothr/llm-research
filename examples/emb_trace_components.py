"""Component-level trace (T1.5): where does the W_E block's content GO?

Answers three questions the boundary-level trace (emb_trace_capture.py)
could not:

  1. PER-COMPONENT ATTRIBUTION — hooks on each layer's self_attn and mlp
     modules capture the two residual DELTAS separately, so block norm-
     fraction changes are attributed to attention vs FFN per layer per position.
  2. CARRIER ANALYSIS (basis-free) — per layer, the (21 x 3584) cross-
     correlation between each token's LAYER-0 block coordinates and its
     layer-L residual, accumulated over all corpus tokens. Singular values
     = how much of the original block content is linearly recoverable
     ANYWHERE in the state; right singular vectors = the carrier dims;
     the in-block column mass = how much stayed in the original dims.
     CONTROL: identical pipeline run from 21 seeded random h0 dims — the
     persistence floor any 21 dims enjoy via the residual stream.
  3. STATIC FFN/ATTN READ-WRITE MAPS — per layer: RMSNorm per-dim weights
     at block dims; gate/up column norms (FFN reads block); down_proj row
     norms (FFN writes block); o_proj row norms (attention writes block);
     each vs the seeded control dims.

Corpus: emb_trace_corpus.all_probes() (~51 committed texts). Raw text, no
chat template, prefill only. Output: emb_trace_components.pt (cache;
promote via manifest).
"""

from __future__ import annotations

import time
from typing import Any

import torch

from _emb_artifacts import load_artifact, write_artifact
from emb_capture import BASE_ID, REVISION
from emb_trace_corpus import all_probes

SEED = 20260611
D = 3584


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
        [i for i in range(D) if i not in set(block_dims.tolist())], dtype=torch.long
    )
    control_dims = pool[torch.randperm(pool.numel(), generator=g)[:nb]]

    print(f"loading {BASE_ID} @ {REVISION[:8]} (CPU bf16) ...")
    t0 = time.time()
    model, tok = surgery.load_model(
        BASE_ID, mode="bf16", device_map="cpu", revision=REVISION
    )
    print(f"  loaded in {time.time() - t0:.0f}s")
    n_layers: int = model.config.num_hidden_layers

    # ---- 3. static read/write maps --------------------------------------------
    print("static maps ...")
    static: dict[str, torch.Tensor] = {
        k: torch.zeros(n_layers)
        for k in (
            "norm1_block",
            "norm1_control",
            "norm2_block",
            "norm2_control",
            "ffn_read_block",
            "ffn_read_control",
            "ffn_write_block",
            "ffn_write_control",
            "attn_write_block",
            "attn_write_control",
        )
    }
    for li, layer in enumerate(model.model.layers):
        w1 = layer.input_layernorm.weight.detach().to(torch.float32)
        w2 = layer.post_attention_layernorm.weight.detach().to(torch.float32)
        static["norm1_block"][li] = w1[block_dims].abs().mean()
        static["norm1_control"][li] = w1[control_dims].abs().mean()
        static["norm2_block"][li] = w2[block_dims].abs().mean()
        static["norm2_control"][li] = w2[control_dims].abs().mean()
        gate = layer.mlp.gate_proj.weight.detach().to(torch.float32)
        up = layer.mlp.up_proj.weight.detach().to(torch.float32)
        down = layer.mlp.down_proj.weight.detach().to(torch.float32)
        o = layer.self_attn.o_proj.weight.detach().to(torch.float32)
        # reads: columns of gate/up (in-dim = residual dim)
        static["ffn_read_block"][li] = (
            gate[:, block_dims].norm() ** 2 + up[:, block_dims].norm() ** 2
        ).sqrt()
        static["ffn_read_control"][li] = (
            gate[:, control_dims].norm() ** 2 + up[:, control_dims].norm() ** 2
        ).sqrt()
        # writes: rows of down/o (out-dim = residual dim)
        static["ffn_write_block"][li] = down[block_dims].norm()
        static["ffn_write_control"][li] = down[control_dims].norm()
        static["attn_write_block"][li] = o[block_dims].norm()
        static["attn_write_control"][li] = o[control_dims].norm()

    # ---- hooks for component deltas --------------------------------------------
    deltas: dict[tuple[int, str], torch.Tensor] = {}

    def make_hook(li: int, comp: str):
        def hook(module: Any, args: Any, output: Any) -> None:
            out = output[0] if isinstance(output, tuple) else output
            deltas[(li, comp)] = out.detach()[0].to(torch.float32)  # (n_pos, D)

        return hook

    handles = []
    for li, layer in enumerate(model.model.layers):
        handles.append(layer.self_attn.register_forward_hook(make_hook(li, "attn")))
        handles.append(layer.mlp.register_forward_hook(make_hook(li, "ffn")))

    # ---- forward passes ----------------------------------------------------------
    probes = all_probes()
    print(f"forward passes over {len(probes)} probes ...")
    L_states = n_layers + 1  # hidden_states checkpoints
    # carrier accumulators, fp64: separate sets for block-source and control-source
    acc: dict[str, Any] = {}
    for src in ("block", "control"):
        acc[src] = {
            "n": 0,
            "sb": torch.zeros(nb, dtype=torch.float64),
            "sb2": torch.zeros(nb, dtype=torch.float64),
            "sh": torch.zeros(L_states, D, dtype=torch.float64),
            "sh2": torch.zeros(L_states, D, dtype=torch.float64),
            "sbh": torch.zeros(L_states, nb, D, dtype=torch.float64),
        }
    comp_stats: list[dict[str, Any]] = []
    t0 = time.time()
    for pi, (pid, cat, text) in enumerate(probes):
        enc = tok(text, return_tensors="pt")
        with torch.no_grad():
            res = model(**enc, output_hidden_states=True, use_cache=False)
        hs = res.hidden_states
        n_pos = hs[0].shape[1]
        positions = list(range(1, n_pos))  # skip pos 0 (sink position)
        # component deltas: block-frac + norm per layer (mean over positions)
        a_frac = torch.zeros(n_layers)
        f_frac = torch.zeros(n_layers)
        a_norm = torch.zeros(n_layers)
        f_norm = torch.zeros(n_layers)
        for li in range(n_layers):
            for comp, frac_t, norm_t in (
                ("attn", a_frac, a_norm),
                ("ffn", f_frac, f_norm),
            ):
                d = deltas[(li, comp)][positions]
                norms = d.norm(dim=1).clamp_min(1e-12)
                frac_t[li] = (d[:, block_dims].norm(dim=1) / norms).mean()
                norm_t[li] = norms.mean()
        comp_stats.append(
            {
                "id": pid,
                "category": cat,
                "n_pos": n_pos,
                "attn_block_frac": a_frac,
                "ffn_block_frac": f_frac,
                "attn_norm": a_norm,
                "ffn_norm": f_norm,
            }
        )
        # carrier accumulation
        h0 = hs[0][0].to(torch.float64)  # (n_pos, D)
        srcs = {
            "block": h0[positions][:, block_dims],
            "control": h0[positions][:, control_dims],
        }
        for src, b in srcs.items():
            a = acc[src]
            a["n"] += len(positions)
            a["sb"] += b.sum(0)
            a["sb2"] += (b**2).sum(0)
            for li in range(L_states):
                h = hs[li][0, positions].to(torch.float64)
                if src == "block":  # h sums identical for both sources
                    a["sh"][li] += h.sum(0)
                    a["sh2"][li] += (h**2).sum(0)
                a["sbh"][li] += b.T @ h
        if (pi + 1) % 10 == 0:
            print(
                f"  {pi + 1}/{len(probes)} ({(pi + 1) / (time.time() - t0):.2f} probes/s)"
            )
    for h in handles:
        h.remove()
    # control shares the h sums
    acc["control"]["sh"] = acc["block"]["sh"]
    acc["control"]["sh2"] = acc["block"]["sh2"]

    # ---- finalize carrier correlations -------------------------------------------
    print("finalizing carrier correlations ...")
    carrier: dict[str, Any] = {}
    for src in ("block", "control"):
        a = acc[src]
        n = a["n"]
        mb = a["sb"] / n
        vb = (a["sb2"] / n - mb**2).clamp_min(1e-12).sqrt()
        sv = torch.zeros(L_states, nb)
        in_block_mass = torch.zeros(L_states)
        carrier_dims_top: list[list[int]] = []
        for li in range(L_states):
            mh = a["sh"][li] / n
            vh = (a["sh2"][li] / n - mh**2).clamp_min(1e-12).sqrt()
            cov = a["sbh"][li] / n - torch.outer(mb, mh)
            corr = (cov / torch.outer(vb, vh)).to(torch.float32)  # (nb, D)
            U, S, Vh = torch.linalg.svd(corr, full_matrices=False)
            sv[li] = S
            mass_block = float((corr[:, block_dims] ** 2).sum())
            in_block_mass[li] = mass_block / float((corr**2).sum())
            top_carriers = (corr**2).sum(0).topk(10).indices.tolist()
            carrier_dims_top.append(top_carriers)
        carrier[src] = {
            "singular_values": sv,  # (L_states, 21)
            "in_block_mass_frac": in_block_mass,
            "top_carrier_dims": carrier_dims_top,
        }

    out: dict[str, Any] = {
        "static_maps": static,
        "component_stats": comp_stats,
        "carrier": carrier,
        "block_dims": block_dims,
        "control_dims": control_dims,
        "n_probes": len(probes),
        "n_tokens_accumulated": acc["block"]["n"],
        "protocol": "raw text, no chat template, positions>0 only, bf16 CPU/fp64 acc",
        "base_id": BASE_ID,
        "revision": REVISION,
        "seed": SEED,
    }
    torch.save(out, write_artifact("emb_trace_components.pt"))
    print("wrote emb_trace_components.pt")

    # console summary
    bsv = carrier["block"]["singular_values"]
    csv = carrier["control"]["singular_values"]
    print("carrier top-SV by layer (block vs control):")
    for li in (0, 1, 2, 4, 8, 14, 20, 28):
        print(
            f"  L{li:2d}: block sv1 {float(bsv[li, 0]):.3f} sum {float(bsv[li].sum()):.2f} "
            f"in-block-mass {float(carrier['block']['in_block_mass_frac'][li]):.3f} | "
            f"control sv1 {float(csv[li, 0]):.3f} sum {float(csv[li].sum()):.2f}"
        )
    af = torch.stack([c["attn_block_frac"] for c in comp_stats]).mean(0)
    ff = torch.stack([c["ffn_block_frac"] for c in comp_stats]).mean(0)
    print("mean delta block-frac (attn | ffn) at L0,1,2,7,14,20,27:")
    for li in (0, 1, 2, 7, 14, 20, 27):
        print(f"  L{li:2d}: attn {float(af[li]):.4f} | ffn {float(ff[li]):.4f}")
    assert not torch.isnan(bsv).any() and not torch.isnan(af).any()
    print("validation gates passed")


if __name__ == "__main__":
    main()
