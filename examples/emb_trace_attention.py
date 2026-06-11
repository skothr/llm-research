"""T2 capture — delimiter-attention census + RoPE-band QK decomposition.

Tests the pre-registered P1 predictions
(research/arcs/03_embedding-atlas/plans/2026-06-11-predictions.md):

  P1a  per-(layer, head) attention census: mean attention weight from
       delimiter QUERY positions to PRECEDING delimiter KEY positions,
       vs an exactly offset-matched control (non-delimiter query to
       non-delimiter key at the same offset). Heads where
       delim->delim >> ctrl->ctrl are delimiter-tracking candidates.
  P1c  same census restricted to '.'/'。' queries attending to preceding
       ','/'、'/'，' keys (period-collects-commas), by layer.
  P1d  RoPE-band decomposition: from hooked PRE-rotary q/k (rotary applied
       manually in fp32, HF rotate_half convention, theta=1e6), the
       per-band contribution to the q.k logit for delimiter pairs vs the
       matched control pairs, accumulated per (layer, head, band).

Also recorded: attention mass to position 0 (the sink) from delimiter vs
control queries — the sink context every other number sits inside.

Protocol: loads the model directly via AutoModelForCausalLM with
attn_implementation="eager" (output_attentions=True needs it; the sdpa
path returns no weights) — deliberate deviation from surgery.load_model,
same revision pin and bf16 CPU. Corpus: the 51 committed probes. V-side
prediction P1e is NOT covered here (needs ablation runs; T4).

Output: emb_trace_attention.pt (cache; promote via manifest).
"""

from __future__ import annotations

import time
from typing import Any

import torch

from _emb_artifacts import write_artifact
from emb_capture import BASE_ID, REVISION
from emb_trace_capture import DELIM_STRINGS
from emb_trace_corpus import all_probes

SEED = 20260611
COMMA_SET = {",", "，", "、", " ,"}
PERIOD_SET = {".", "。", " ."}


def rotary_cos_sin(
    n_pos: int, head_dim: int, theta: float
) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    )
    t = torch.arange(n_pos, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)  # (n_pos, head_dim/2)
    emb = torch.cat((freqs, freqs), dim=-1)  # (n_pos, head_dim)
    return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def main() -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from llm_surgeon.surgery import MODEL_CACHE_DIR

    print(f"loading {BASE_ID} @ {REVISION[:8]} (CPU bf16, eager attention) ...")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(
        BASE_ID, revision=REVISION, cache_dir=MODEL_CACHE_DIR
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_ID,
        revision=REVISION,
        dtype=torch.bfloat16,
        device_map="cpu",
        attn_implementation="eager",
        cache_dir=MODEL_CACHE_DIR,
    )
    model.eval()
    print(f"  loaded in {time.time() - t0:.0f}s")
    cfg = model.config
    n_layers: int = cfg.num_hidden_layers
    n_q: int = cfg.num_attention_heads
    n_kv: int = cfg.num_key_value_heads
    head_dim: int = cfg.hidden_size // n_q
    n_bands = head_dim // 2
    # transformers 5.x moved rope_theta into config.rope_parameters
    rope_params = getattr(cfg, "rope_parameters", None) or {}
    theta = float(rope_params.get("rope_theta", getattr(cfg, "rope_theta", 0.0)) or 0.0)
    assert theta == 1_000_000.0, f"unexpected rope_theta {theta} (config.json says 1e6)"
    group = n_q // n_kv  # q heads per kv head

    # q/k pre-rotary capture hooks
    qk_store: dict[tuple[int, str], torch.Tensor] = {}

    def mk_hook(li: int, kind: str):
        def hook(module: Any, args: Any, output: Any) -> None:
            qk_store[(li, kind)] = output.detach()[0].to(torch.float32)

        return hook

    handles = []
    for li, layer in enumerate(model.model.layers):
        handles.append(layer.self_attn.q_proj.register_forward_hook(mk_hook(li, "q")))
        handles.append(layer.self_attn.k_proj.register_forward_hook(mk_hook(li, "k")))

    g = torch.Generator().manual_seed(SEED)
    # accumulators
    acc = {
        "d2d_sum": torch.zeros(n_layers, n_q),
        "d2d_n": torch.zeros(n_layers, n_q),
        "c2c_sum": torch.zeros(n_layers, n_q),
        "c2c_n": torch.zeros(n_layers, n_q),
        "p2c_sum": torch.zeros(n_layers, n_q),
        "p2c_n": torch.zeros(n_layers, n_q),
        "d2sink_sum": torch.zeros(n_layers, n_q),
        "d2sink_n": torch.zeros(n_layers, n_q),
        "c2sink_sum": torch.zeros(n_layers, n_q),
        "c2sink_n": torch.zeros(n_layers, n_q),
        # band logit decomposition for delim pairs and matched controls
        "band_d_sum": torch.zeros(n_layers, n_q, n_bands),
        "band_c_sum": torch.zeros(n_layers, n_q, n_bands),
        "band_n": torch.zeros(n_layers, n_q),
    }
    offsets_seen: list[int] = []
    probes = all_probes()
    print(f"forward passes over {len(probes)} probes ...")
    t0 = time.time()
    for pi, (pid, cat, text) in enumerate(probes):
        enc = tok(text, return_tensors="pt")
        ids = enc.input_ids[0]
        toks = [tok.decode([int(i)]) for i in ids]
        n_pos = len(toks)
        delims = [i for i, s in enumerate(toks) if s in DELIM_STRINGS and i > 0]
        commas = [i for i in delims if toks[i] in COMMA_SET]
        periods = [i for i in delims if toks[i] in PERIOD_SET]
        ctrls = [i for i in range(1, n_pos) if i not in delims]
        # delim->preceding-delim pairs (same type set: any delim to any delim)
        d_pairs = [(i, j) for i in delims for j in delims if j < i]
        # offset-matched control pairs
        c_pairs = []
        for i, j in d_pairs:
            off = i - j
            cands = [ic for ic in ctrls if ic - off >= 1 and (ic - off) in set(ctrls)]
            if cands:
                ic = cands[int(torch.randint(len(cands), (1,), generator=g))]
                c_pairs.append((ic, ic - off))
        p_pairs = [(i, j) for i in periods for j in commas if j < i]
        offsets_seen.extend(i - j for i, j in d_pairs)

        with torch.no_grad():
            res = model(**enc, output_attentions=True, use_cache=False)
        attn = res.attentions  # tuple of (1, n_q, n, n)

        for li in range(n_layers):
            A = attn[li][0].to(torch.float32)  # (n_q, n, n)
            for pairs, skey, nkey in (
                (d_pairs, "d2d_sum", "d2d_n"),
                (c_pairs, "c2c_sum", "c2c_n"),
                (p_pairs, "p2c_sum", "p2c_n"),
            ):
                if pairs:
                    qi = torch.tensor([p[0] for p in pairs])
                    kj = torch.tensor([p[1] for p in pairs])
                    acc[skey][li] += A[:, qi, kj].sum(dim=1)
                    acc[nkey][li] += len(pairs)
            if delims:
                acc["d2sink_sum"][li] += A[:, delims, 0].sum(dim=1)
                acc["d2sink_n"][li] += len(delims)
            if ctrls:
                acc["c2sink_sum"][li] += A[:, ctrls, 0].sum(dim=1)
                acc["c2sink_n"][li] += len(ctrls)

        # band decomposition from hooked q/k
        if d_pairs and c_pairs:
            cos, sin = rotary_cos_sin(n_pos, head_dim, theta)
            for li in range(n_layers):
                q = qk_store[(li, "q")].view(n_pos, n_q, head_dim)
                k = qk_store[(li, "k")].view(n_pos, n_kv, head_dim)
                qr = q * cos[:, None, :] + rotate_half(q) * sin[:, None, :]
                kr = k * cos[:, None, :] + rotate_half(k) * sin[:, None, :]
                for pairs, key in ((d_pairs, "band_d_sum"), (c_pairs, "band_c_sum")):
                    qi = torch.tensor([p[0] for p in pairs])
                    kj = torch.tensor([p[1] for p in pairs])
                    qv = qr[qi]  # (P, n_q, head_dim)
                    kv = kr[kj][:, :, None, :].expand(-1, n_kv, group, head_dim)
                    kv = kv.reshape(len(pairs), n_q, head_dim)
                    prod = qv * kv  # (P, n_q, head_dim)
                    band = prod[..., :n_bands] + prod[..., n_bands:]  # (P, n_q, bands)
                    acc[key][li] += band.sum(dim=0)
                acc["band_n"][li] += len(d_pairs)
        qk_store.clear()
        if (pi + 1) % 10 == 0:
            print(
                f"  {pi + 1}/{len(probes)} ({(time.time() - t0) / (pi + 1):.1f}s/probe)"
            )

    for h in handles:
        h.remove()

    out: dict[str, Any] = {
        **{k: v for k, v in acc.items()},
        "offsets_seen": torch.tensor(offsets_seen, dtype=torch.int32),
        "n_layers": n_layers,
        "n_q_heads": n_q,
        "n_kv_heads": n_kv,
        "head_dim": head_dim,
        "rope_theta": theta,
        "delim_strings": sorted(DELIM_STRINGS),
        "protocol": "eager attention, raw text, offset-matched controls, fp32 stats",
        "base_id": BASE_ID,
        "revision": REVISION,
        "seed": SEED,
    }
    torch.save(out, write_artifact("emb_trace_attention.pt"))
    print("wrote emb_trace_attention.pt")

    d2d = acc["d2d_sum"] / acc["d2d_n"].clamp_min(1)
    c2c = acc["c2c_sum"] / acc["c2c_n"].clamp_min(1)
    diff = d2d - c2c
    top = torch.topk(diff.flatten(), 8)
    print("top delim->delim heads (excess attention over offset-matched control):")
    for v, idx in zip(top.values, top.indices):
        li, hi = int(idx // n_q), int(idx % n_q)
        print(
            f"  L{li}H{hi}: d2d {float(d2d[li, hi]):.4f} vs c2c {float(c2c[li, hi]):.4f} "
            f"(+{float(v):.4f})"
        )
    d2s = (acc["d2sink_sum"] / acc["d2sink_n"].clamp_min(1)).mean()
    c2s = (acc["c2sink_sum"] / acc["c2sink_n"].clamp_min(1)).mean()
    print(
        f"mean attention to sink pos0: delim queries {float(d2s):.3f}, ctrl {float(c2s):.3f}"
    )
    assert not torch.isnan(diff).any()
    print("validation gates passed")


if __name__ == "__main__":
    main()
