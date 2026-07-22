#!/usr/bin/env python3
"""Stage 6 of the jspace arc: the NLA cross-tie.

At the activation the NLA verbalizer was trained to read
(`kitft/nla-qwen2.5-7b-L20-av`, HF ``hidden_states[20]``), compare two
independent activation->language channels on the *same* hidden state:

* the **J-lens** readout ``W_U norm(J_l h)`` (a *derived* readout), and
* the **NLA AV** free-text verbalization (a *trained* one,
  ``llm_surgeon.probe.nla_verbalize``).

Two questions (design: ``plans/2026-07-20-stage6-design.md``):
  A. Do the channels agree on the active concepts? (Metric 1: where the AV's
     words fall in the J-lens ranking; Metric 2: J-lens top-k in the AV text;
     Metric 3: both recover a known concept — concept-loaded set only.)
  B. Is the NLA content concentrated in the activation's small J-space
     component? Decompose ``h -> c (J-space, k=25) + residual``, re-verbalize
     ``c`` and the residual separately (with a norm-matched random control for
     the residual), and score which carries the AV content.

LAYER INDEX (load-bearing, verified 2026-07-20): jlens source_layer L == block-L
output == HF ``hidden_states[L+1]``. The NLA's ``hidden_states[20]`` is therefore
jlens source_layer **19** (``--jlens-layer`` default 19), NOT 20. The full cache
lens has layer 19; the committed LFS subset does not.

Two-phase (avoids co-resident base+AV; the NLA arc's pattern): Phase 1 loads the
7B nf4 base + lens on GPU, captures every vector to verbalize + the J-lens
readouts, frees the base; Phase 2 loads the AV (CPU bf16) and verbalizes.

Usage (primary — 7B nf4 + NLA AV, needs GPU for the base):
    python examples/jspace_nla_crosstie.py \
        --model Qwen/Qwen2.5-7B-Instruct --mode nf4 \
        --lens research/arcs/04_jspace/data/cache/jlens_qwen2.5-7b_nf4_n100.pt \
        --n-neutral 12 --n-concept 12 --n-decomp 12

Plumbing smoke (no NLA; validates capture/readout/decomposition on any model):
    python examples/jspace_nla_crosstie.py --skip-nla --n-neutral 2 --n-concept 2 \
        --model Qwen/Qwen2.5-1.5B-Instruct --mode bf16 --device cpu --jlens-layer 19 \
        --lens research/arcs/04_jspace/data/cache/jlens_qwen2.5-1.5b_bf16_n100.pt
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import Tensor

# Shared conventions: model/lens loading + prompt/position helpers from the
# stage-3 scan, and the *identical* gradient-pursuit decomposition (stage 4).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _jspace_pursuit import gradient_pursuit_layer  # noqa: E402
from jspace_readout_scan import build_model, load_prompts, slug  # noqa: E402

NLA_HIDDEN_STATE = 20  # the NLA AV's training layer, as HF hidden_states index
DEFAULT_JLENS_LAYER = NLA_HIDDEN_STATE - 1  # == block-19 output == hidden_states[20]
TOPK = 50  # J-lens readout top-k for content overlap + in-topk rank threshold
DECOMP_K = 25  # headline J-space sparsity (design default)
WORD_RE = re.compile(r"[a-zA-Z]{3,}")

# Fixed English function-word stoplist (articles / preps / conjunctions /
# pronouns / auxiliaries / light verbs). Both channels are filtered to content.
STOPWORDS: frozenset[str] = frozenset(
    """the and for are but not you your yours with that this these those they them their
    from have has had was were will would could should can may might must shall about into
    over under then than there here where when what which who whom whose why how all any
    each few more most other some such only own same both again further once out off down
    upon while because during before after above below between through against without within
    among across around above being been does did done doing itself himself herself themselves
    our ours their its his her hers him she was are were also very much many one two get got
    let put use used using make made like just now new see seen say said way well still ever
    per via etc yes""".split()
)

# The AV's meta-description vocabulary (NLA arc Theme 9 / L2: AV format-bias).
# These words carry verbalizer prior, not input content, so they are filtered
# from both channels' content sets.
AV_TEMPLATE_STOP: frozenset[str] = frozenset(
    """activation activations vector vectors concept concepts semantic semantics content
    represents represent representing representation appears appear seems seem suggests
    suggest indicating indicate context contexts token tokens model models text following
    related relate associated association meaning means indicates structured structure format
    formatted explanation describe description describing describes information includes
    include including contains contain conveys convey reflects reflect focus focused theme
    themes idea ideas discussion notion notions element elements aspect aspects general
    specific particular various several primarily likely possibly overall""".split()
)

# Concept-loaded prompts (raw text; concept at the final token). Tied to the NLA
# arc's concept-direction / country themes (Theme 6). Each carries the expected
# concept token(s) for Metric 3. Named CONCEPT_PROMPTS in the design.
CONCEPT_PROMPTS: list[tuple[str, list[str]]] = [
    ("The capital of France is", ["paris", "france"]),
    ("The capital of Japan is", ["tokyo", "japan"]),
    ("The chemical symbol for gold is", ["gold", "au"]),
    ("The largest planet in the solar system is", ["jupiter", "planet"]),
    ("Water is made of hydrogen and", ["oxygen", "hydrogen"]),
    ("The opposite of hot is", ["cold", "cool"]),
    ("A doctor works in a", ["hospital", "clinic"]),
    ("The color of a clear daytime sky is", ["blue", "sky"]),
    ("Two plus two equals", ["four", "number"]),
    ("The author of Romeo and Juliet is", ["shakespeare", "author"]),
    ("A baby dog is called a", ["puppy", "dog"]),
    ("The freezing point of water in Celsius is", ["zero", "cold"]),
    ("The currency used in Japan is the", ["yen", "currency"]),
    ("The tallest animal in the world is the", ["giraffe", "animal"]),
    ("The sun rises in the", ["east", "morning"]),
    ("A group of wolves is called a", ["pack", "wolves"]),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="J-lens / NLA cross-tie (stage 6).")
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--mode", default="nf4", choices=["nf4", "int8", "bf16", "fp16"])
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument(
        "--lens",
        default="research/arcs/04_jspace/data/cache/jlens_qwen2.5-7b_nf4_n100.pt",
        help="Fitted JacobianLens .pt (must include --jlens-layer).",
    )
    p.add_argument(
        "--jlens-layer",
        type=int,
        default=DEFAULT_JLENS_LAYER,
        help=f"jlens source layer for the NLA activation (default {DEFAULT_JLENS_LAYER} "
        f"== HF hidden_states[{NLA_HIDDEN_STATE}]).",
    )
    p.add_argument(
        "--prompts",
        default="research/arcs/04_jspace/data/heldout_prompts_wikitext103_n30.json",
        help="Neutral held-out prompts JSON ({'prompts': [str, ...]}).",
    )
    p.add_argument(
        "--n-neutral", type=int, default=12, help="Neutral prompts (Exp. A)."
    )
    p.add_argument(
        "--n-concept", type=int, default=12, help="Concept prompts (Exp. A)."
    )
    p.add_argument(
        "--n-decomp",
        type=int,
        default=12,
        help="Prompts run through Experiment B (split evenly neutral/concept).",
    )
    p.add_argument("--topk", type=int, default=TOPK)
    p.add_argument("--decomp-k", type=int, default=DECOMP_K)
    p.add_argument("--max-seq-len", type=int, default=512)
    p.add_argument("--max-new-tokens", type=int, default=200, help="AV output cap.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--skip-nla",
        action="store_true",
        help="Phase 1 only: capture + readout + decomposition, no AV. Smoke aid.",
    )
    p.add_argument("--out", default=None, help="Output .pt (default: auto-named).")
    return p.parse_args()


def default_out(model: str, lens: str) -> str:
    stem = Path(lens).stem
    return f"research/arcs/04_jspace/data/cache/nla_crosstie_{slug(model)}_{stem}.pt"


# ---------------------------------------------------------------------------
# Text / channel-agreement helpers (pure Python + torch; model-free).
# ---------------------------------------------------------------------------
def content_words(text: str) -> list[str]:
    """Lowercased content words (>=3 letters) minus the two stoplists, deduped."""
    out: list[str] = []
    seen: set[str] = set()
    for w in WORD_RE.findall(text.lower()):
        if w in STOPWORDS or w in AV_TEMPLATE_STOP or w in seen:
            continue
        seen.add(w)
        out.append(w)
    return out


def jlens_content_tokens(readout: Tensor, tok: Any, k: int) -> list[str]:
    """Decoded top-k J-lens tokens, stripped/lowercased, filtered to content."""
    ids = readout.topk(k).indices.tolist()
    out: list[str] = []
    seen: set[str] = set()
    for tid in ids:
        s = tok.decode([int(tid)]).strip().lower()
        if not s or not s.isalpha() or len(s) < 3:
            continue
        if s in STOPWORDS or s in AV_TEMPLATE_STOP or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def token_rank(readout: Tensor, token_id: int) -> int:
    """0-based rank of ``token_id`` in ``readout`` (0 == top-1)."""
    return int((readout > readout[token_id]).sum().item())


def first_subtoken_id(tok: Any, word: str) -> int | None:
    """Base-vocab id of the leading-space form of ``word`` (mid-sentence form)."""
    ids = tok(" " + word, add_special_tokens=False)["input_ids"]
    return int(ids[0]) if ids else None


def nla_ranks(words: list[str], readout: Tensor, tok: Any) -> list[int]:
    """Rank of each content word in the J-lens readout distribution."""
    ranks: list[int] = []
    for w in words:
        tid = first_subtoken_id(tok, w)
        if tid is not None:
            ranks.append(token_rank(readout, tid))
    return ranks


def topk_overlap(jlens_toks: list[str], nla_words: list[str]) -> float:
    """Fraction of J-lens content tokens that appear (whole/substring) in AV text."""
    if not jlens_toks:
        return float("nan")
    ws = set(nla_words)
    hits = sum(1 for t in jlens_toks if t in ws or any(t in w or w in t for w in ws))
    return hits / len(jlens_toks)


def jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return float("nan")
    return len(sa & sb) / len(sa | sb)


def median(xs: list[float]) -> float:
    arr = np.array([x for x in xs if not (isinstance(x, float) and np.isnan(x))])
    return float(np.median(arr)) if arr.size else float("nan")


def nanmean(xs: list[float]) -> float:
    arr = np.array([x for x in xs if not (isinstance(x, float) and np.isnan(x))])
    return float(np.mean(arr)) if arr.size else float("nan")


# ---------------------------------------------------------------------------
# Phase 1: base + lens — capture, readout, decomposition. No AV.
# ---------------------------------------------------------------------------
def build_prompt_set(args: argparse.Namespace) -> list[dict[str, Any]]:
    neutral = load_prompts(args.prompts, args.n_neutral)
    items: list[dict[str, Any]] = [
        {"tag": "neutral", "prompt": p, "target": []} for p in neutral
    ]
    for prompt, target in CONCEPT_PROMPTS[: args.n_concept]:
        items.append({"tag": "concept", "prompt": prompt, "target": target})
    return items


def phase1_capture(
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[Tensor], str, int]:
    """Returns (per_prompt, readout_logits[cpu fp32 per prompt], tok_name, vocab)."""
    from jlens import JacobianLens

    from jspace_structure_scan import capture_residuals

    t_load = time.time()
    m = build_model(args)
    lens = JacobianLens.load(args.lens)
    print(f"[load] model + lens in {time.time() - t_load:.1f}s; {lens!r}")

    layer = int(args.jlens_layer)
    if layer not in lens.source_layers:
        raise SystemExit(
            f"--jlens-layer {layer} not in fitted source_layers {list(lens.source_layers)} "
            f"(the NLA activation is hidden_states[{NLA_HIDDEN_STATE}] == jlens layer "
            f"{DEFAULT_JLENS_LAYER}; use the full cache lens, not the LFS subset)."
        )

    mm = cast(Any, m)
    w_u = cast(Tensor, mm._lm_head.weight)
    if not (torch.is_tensor(w_u) and w_u.dim() == 2 and w_u.is_floating_point()):
        raise SystemExit(
            "lm_head.weight is not a plain 2-D float tensor (need raw W_U)."
        )
    device = w_u.device
    vocab = int(w_u.shape[0])
    print(f"[wu] W_U {tuple(w_u.shape)} {w_u.dtype} on {device}; jlens layer {layer}")

    jac = lens.jacobians[layer].to(device)  # fp32 d x d
    d = jac.shape[0]

    items = build_prompt_set(args)
    n_decomp = int(args.n_decomp)
    # Even split of the decomposition budget across the two set tags.
    n_dec_neutral = n_decomp // 2
    n_dec_concept = n_decomp - n_dec_neutral
    dec_flags: list[bool] = []
    seen_n = seen_c = 0
    for it in items:
        if it["tag"] == "neutral" and seen_n < n_dec_neutral:
            dec_flags.append(True)
            seen_n += 1
        elif it["tag"] == "concept" and seen_c < n_dec_concept:
            dec_flags.append(True)
            seen_c += 1
        else:
            dec_flags.append(False)

    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    per_prompt: list[dict[str, Any]] = []
    readouts: list[Tensor] = []

    for pi, it in enumerate(items):
        prompt = cast(str, it["prompt"])
        seq_len = int(m.encode(prompt, max_length=args.max_seq_len).shape[1])
        last = seq_len - 1
        resid = capture_residuals(m, prompt, [layer], [last], args.max_seq_len)
        h = resid[layer][0].float()  # [d] on device

        with torch.no_grad():
            readout = m.unembed((h @ jac.t()).unsqueeze(0)).float()[0]  # [V]
        readouts.append(readout.detach().cpu())

        rec: dict[str, Any] = {
            "tag": it["tag"],
            "prompt": prompt,
            "target": it["target"],
            "seq_len": seq_len,
            "position": last,
            "readout_topk_ids": readout.topk(args.topk).indices.tolist(),
            # Full readout persisted (fp16) so the rank-based metrics + nulls are
            # re-derivable offline by the audit (repo: raw data is a deliverable).
            "readout_fp16": readout.detach().half().cpu().clone(),
            "h_vec": h.detach().cpu().clone(),  # verbalized in Phase 2
            "decomp": dec_flags[pi],
        }

        if dec_flags[pi]:
            res = gradient_pursuit_layer(
                h.unsqueeze(0), jac, w_u, [args.decomp_k], return_components=True
            )
            comp = cast(Tensor, res["components"])[0].float()  # [d] J-space component
            residual = (h - comp).detach()
            c_norm = float(comp.norm())
            r = torch.randn(d, generator=gen).to(device)
            r = r / r.norm().clamp_min(1e-12) * c_norm
            res_ctrl = (h - r).detach()
            rec["c_vec"] = comp.detach().cpu().clone()
            rec["res_vec"] = residual.detach().cpu().clone()
            rec["resctrl_vec"] = res_ctrl.detach().cpu().clone()
            rec["varfrac"] = float(res["varfrac"][args.decomp_k][0])
            rec["c_norm"] = c_norm
            rec["h_norm"] = float(h.norm())
        per_prompt.append(rec)
        print(
            f"[cap {pi + 1}/{len(items)}] {it['tag']:>7} seq={seq_len} "
            f"decomp={dec_flags[pi]} :: {prompt[:52]!r}"
        )

    # One shared random-unit control direction (guards: does the AV emit a
    # coherent concept for ANY direction?).
    rand_unit = torch.randn(d, generator=gen)
    rand_unit = rand_unit / rand_unit.norm().clamp_min(1e-12) * float(np.sqrt(d))

    tok_name = args.model
    control = {"rand_unit_vec": rand_unit}
    per_prompt.append({"tag": "_control", "control": control})
    return per_prompt, readouts, tok_name, vocab


# ---------------------------------------------------------------------------
# Phase 2: AV verbalization of every captured vector.
# ---------------------------------------------------------------------------
def _av_jobs(per_prompt: list[dict[str, Any]]) -> list[tuple[int, str, Tensor]]:
    """(prompt_index, text_key, vector) for every AV verbalization to run."""
    jobs: list[tuple[int, str, Tensor]] = []
    for pidx, rec in enumerate(per_prompt):
        if rec["tag"] == "_control":
            jobs.append((pidx, "rand_unit_text", rec["control"]["rand_unit_vec"]))
            continue
        jobs.append((pidx, "h_text", rec["h_vec"]))
        if rec.get("decomp"):
            jobs.append((pidx, "c_text", rec["c_vec"]))
            jobs.append((pidx, "res_text", rec["res_vec"]))
            jobs.append((pidx, "resctrl_text", rec["resctrl_vec"]))
    return jobs


def _load_partial(path: str) -> dict[tuple[int, str], str]:
    """Read the AV checkpoint sidecar: {(prompt_index, key): text}."""
    done: dict[tuple[int, str], str] = {}
    if not os.path.exists(path):
        return done
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                done[(int(rec["pidx"]), str(rec["key"]))] = str(rec["text"])
            except (json.JSONDecodeError, KeyError, ValueError):
                continue  # tolerate a torn final line from a hard kill
    return done


def phase2_verbalize(
    per_prompt: list[dict[str, Any]], args: argparse.Namespace, partial_path: str
) -> None:
    """AV-verbalize every captured vector, checkpointing each result as it lands.

    Each completed verbalization is appended (flushed) to ``partial_path`` as one
    JSON line ``{pidx, key, text, elapsed}`` BEFORE moving on, and on startup any
    (prompt, key) pair already in the sidecar is skipped — so a mid-phase kill
    loses at most the single in-flight verbalization, not the whole phase. The
    sidecar is authoritative for this config (cleared whenever Phase 1 recomputes).
    """
    from llm_surgeon.probe import load_av, nla_verbalize

    jobs = _av_jobs(per_prompt)
    done = _load_partial(partial_path)
    todo = [j for j in jobs if (j[0], j[1]) not in done]
    print(
        f"[av] {len(jobs)} verbalizations; {len(done)} checkpointed, "
        f"{len(todo)} to run (~{len(todo) * 110 / 60:.0f} min est.)"
    )

    if todo:
        t0 = time.time()
        av_model, av_tok, meta = load_av()
        print(f"[av] loaded in {time.time() - t0:.1f}s; d_model={meta['d_model']}")
        with open(partial_path, "a", encoding="utf-8") as f:
            for i, (pidx, key, vec) in enumerate(todo, 1):
                t = time.time()
                text = nla_verbalize(
                    vec.float(),
                    model=av_model,
                    tok=av_tok,
                    meta=meta,
                    max_new_tokens=args.max_new_tokens,
                )
                el = time.time() - t
                f.write(
                    json.dumps({"pidx": pidx, "key": key, "text": text, "elapsed": el})
                    + "\n"
                )
                f.flush()
                os.fsync(f.fileno())
                done[(pidx, key)] = text
                print(f"[av {i}/{len(todo)}] pidx={pidx} {key:>14} {el:.0f}s")

    # Fold the checkpointed texts back onto their prompt records for scoring.
    for (pidx, key), text in done.items():
        per_prompt[pidx][key] = text


# ---------------------------------------------------------------------------
# Scoring + summary.
# ---------------------------------------------------------------------------
def score(
    per_prompt: list[dict[str, Any]],
    readouts: list[Tensor],
    tok: Any,
    vocab: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    prompts = [r for r in per_prompt if r["tag"] != "_control"]
    control = next((r for r in per_prompt if r["tag"] == "_control"), None)
    have_nla = all("h_text" in r for r in prompts)

    # J-lens-side content tokens per prompt (channel-independent).
    for i, r in enumerate(prompts):
        r["jlens_content"] = jlens_content_tokens(readouts[i], tok, args.topk)

    # ---- Experiment A ----
    for i, r in enumerate(prompts):
        target = cast(list[str], r["target"])
        jt = cast(list[str], r["jlens_content"])
        r["jlens_hit"] = bool(target) and any(
            any(t in s or s in t for s in jt) for t in target
        )
        if not have_nla:
            continue
        nw = content_words(cast(str, r["h_text"]))
        r["nla_content"] = nw
        ranks = nla_ranks(nw, readouts[i], tok)
        r["nla_rank_median"] = median([float(x) for x in ranks])
        r["nla_in_topk_frac"] = (
            float(np.mean([1.0 if x < args.topk else 0.0 for x in ranks]))
            if ranks
            else float("nan")
        )
        r["topk_overlap"] = topk_overlap(jt, nw)
        r["nla_hit"] = bool(target) and any(
            any(t in w or w in t for w in nw) for t in target
        )

    summary: dict[str, Any] = {
        "model": args.model,
        "mode": args.mode,
        "lens": args.lens,
        "jlens_layer": int(args.jlens_layer),
        "nla_hidden_state": NLA_HIDDEN_STATE,
        "topk": args.topk,
        "decomp_k": args.decomp_k,
        "vocab": vocab,
        "chance_rank_floor": vocab / 2.0,
        "seed": args.seed,
        "n_neutral": sum(1 for r in prompts if r["tag"] == "neutral"),
        "n_concept": sum(1 for r in prompts if r["tag"] == "concept"),
        "n_decomp": sum(1 for r in prompts if r.get("decomp")),
        "have_nla": have_nla,
    }
    # Concept-recovery cross-tab (Metric 3).
    concept = [r for r in prompts if r["tag"] == "concept"]
    summary["metric3_jlens_hit_rate"] = nanmean(
        [1.0 if r["jlens_hit"] else 0.0 for r in concept]
    )
    if not have_nla:
        return summary

    summary["metric3_nla_hit_rate"] = nanmean(
        [1.0 if r.get("nla_hit") else 0.0 for r in concept]
    )
    summary["metric3_both_hit_rate"] = nanmean(
        [1.0 if (r["jlens_hit"] and r.get("nla_hit")) else 0.0 for r in concept]
    )
    for tag in ("neutral", "concept"):
        grp = [r for r in prompts if r["tag"] == tag]
        summary[f"metric1_rank_median_{tag}"] = median(
            [cast(float, r["nla_rank_median"]) for r in grp]
        )
        summary[f"metric1_in_topk_{tag}"] = nanmean(
            [cast(float, r["nla_in_topk_frac"]) for r in grp]
        )
        summary[f"metric2_overlap_{tag}"] = nanmean(
            [cast(float, r["topk_overlap"]) for r in grp]
        )

    # ---- Mismatched-pairing null (post-hoc, no extra verbalizations) ----
    # A per-group null over an index set: matched = AV words scored against their
    # OWN prompt's J-lens readout; mismatched = against every OTHER prompt's in
    # the group. The concept-only variant is the informative one — the pooled
    # null is dominated by neutral prompts whose AV words rank near/below chance
    # regardless of pairing (so pooled matched ~= pooled mismatched, masking the
    # concept effect). Requested by the coordinator after the quick pass.
    def group_null(idxs: list[int]) -> dict[str, float]:
        matched: list[float] = []
        mismatched: list[float] = []
        for i in idxs:
            nw = cast(list[str], prompts[i]["nla_content"])
            for j in idxs:
                med = median([float(x) for x in nla_ranks(nw, readouts[j], tok)])
                (matched if i == j else mismatched).append(med)
        m_med = median(matched)
        mm = np.array([x for x in mismatched if not np.isnan(x)])
        return {
            "matched_rank_median": m_med,
            "mismatched_rank_median": median(mismatched),
            "matched_pctile_in_mismatched": (
                float((mm < m_med).mean())
                if mm.size and not np.isnan(m_med)
                else float("nan")
            ),
        }

    all_idx = list(range(len(prompts)))
    pooled = group_null(all_idx)
    summary["null_matched_rank_median"] = pooled["matched_rank_median"]
    summary["null_mismatched_rank_median"] = pooled["mismatched_rank_median"]
    summary["null_matched_pctile_in_mismatched"] = pooled[
        "matched_pctile_in_mismatched"
    ]
    for tag in ("neutral", "concept"):
        g = group_null([i for i, r in enumerate(prompts) if r["tag"] == tag])
        summary[f"null_{tag}_matched_rank_median"] = g["matched_rank_median"]
        summary[f"null_{tag}_mismatched_rank_median"] = g["mismatched_rank_median"]
        summary[f"null_{tag}_matched_pctile"] = g["matched_pctile_in_mismatched"]

    # ---- Experiment B ----
    dec = [(i, r) for i, r in enumerate(prompts) if r.get("decomp") and "c_text" in r]
    rand_words = (
        content_words(cast(str, control["rand_unit_text"]))
        if control is not None and "rand_unit_text" in control
        else []
    )
    b_carrier_comp: list[float] = []
    b_carrier_res: list[float] = []
    b_carrier_ctrl: list[float] = []
    b_delta: list[
        float
    ] = []  # damage_jspace - damage_ctrl = carrier_ctrl - carrier_res
    b_comp_rank: list[float] = []
    b_carrier_rand: list[float] = []
    for idx, r in dec:
        hw = content_words(cast(str, r["h_text"]))
        cw = content_words(cast(str, r["c_text"]))
        rw = content_words(cast(str, r["res_text"]))
        rcw = content_words(cast(str, r["resctrl_text"]))
        r["nla_content_component"] = cw
        r["nla_content_residual"] = rw
        r["nla_content_resctrl"] = rcw
        carrier_comp = jaccard(cw, hw)
        carrier_res = jaccard(rw, hw)
        carrier_ctrl = jaccard(rcw, hw)
        r["carrier_component"] = carrier_comp
        r["carrier_residual"] = carrier_res
        r["carrier_resctrl"] = carrier_ctrl
        r["carrier_delta"] = carrier_ctrl - carrier_res
        r["component_rank_median"] = median(
            [float(x) for x in nla_ranks(cw, readouts[idx], tok)]
        )
        b_carrier_comp.append(carrier_comp)
        b_carrier_res.append(carrier_res)
        b_carrier_ctrl.append(carrier_ctrl)
        b_delta.append(r["carrier_delta"])
        b_comp_rank.append(r["component_rank_median"])
        b_carrier_rand.append(jaccard(rand_words, hw))

    summary["expB_carrier_component_mean"] = nanmean(b_carrier_comp)
    summary["expB_carrier_residual_mean"] = nanmean(b_carrier_res)
    summary["expB_carrier_resctrl_mean"] = nanmean(b_carrier_ctrl)
    summary["expB_carrier_delta_mean"] = nanmean(
        b_delta
    )  # >0 supports J-space concentration
    summary["expB_component_rank_median"] = median(b_comp_rank)
    summary["expB_carrier_randunit_mean"] = nanmean(
        b_carrier_rand
    )  # floor for carrier_component
    return summary


def print_summary(summary: dict[str, Any]) -> None:
    print("\n=== Stage 6 NLA cross-tie summary ===")
    print(
        f"model={summary['model']} mode={summary['mode']} "
        f"jlens_layer={summary['jlens_layer']} (== hidden_states[{summary['nla_hidden_state']}]) "
        f"K={summary['topk']} k={summary['decomp_k']}"
    )
    print(
        f"prompts: neutral={summary['n_neutral']} concept={summary['n_concept']} "
        f"decomp={summary['n_decomp']}; chance rank floor={summary['chance_rank_floor']:.0f}"
    )
    print(
        f"[Metric 3] concept J-lens hit-rate: {summary['metric3_jlens_hit_rate']:.3f}"
    )
    if not summary["have_nla"]:
        print("(--skip-nla: Phase-1 only; no NLA metrics)")
        return
    print(
        f"[Metric 3] NLA hit-rate: {summary['metric3_nla_hit_rate']:.3f}  "
        f"both: {summary['metric3_both_hit_rate']:.3f}"
    )
    for tag in ("neutral", "concept"):
        print(
            f"[Metric 1/2] {tag:>7}: rank_median={summary[f'metric1_rank_median_{tag}']:.0f} "
            f"in_top{summary['topk']}={summary[f'metric1_in_topk_{tag}']:.3f} "
            f"overlap={summary[f'metric2_overlap_{tag}']:.3f}"
        )
    print(
        f"[Null pooled ] matched={summary['null_matched_rank_median']:.0f} vs "
        f"mismatched={summary['null_mismatched_rank_median']:.0f} "
        f"(pctile={summary['null_matched_pctile_in_mismatched']:.3f})"
    )
    for tag in ("neutral", "concept"):
        print(
            f"[Null {tag:>7}] matched={summary[f'null_{tag}_matched_rank_median']:.0f} vs "
            f"mismatched={summary[f'null_{tag}_mismatched_rank_median']:.0f} "
            f"(pctile={summary[f'null_{tag}_matched_pctile']:.3f})"
        )
    print(
        f"[Exp B] carrier: component={summary['expB_carrier_component_mean']:.3f} "
        f"residual={summary['expB_carrier_residual_mean']:.3f} "
        f"resctrl={summary['expB_carrier_resctrl_mean']:.3f} "
        f"randunit_floor={summary['expB_carrier_randunit_mean']:.3f}"
    )
    print(
        f"[Exp B] damage_jspace - damage_ctrl = {summary['expB_carrier_delta_mean']:+.3f} "
        f"(>0 supports J-space concentration); "
        f"component rank_median={summary['expB_component_rank_median']:.0f}"
    )


def _config_key(args: argparse.Namespace) -> dict[str, Any]:
    """Identity of a run's inputs — Phase-1/sidecar checkpoints are keyed to it."""
    return {
        k: getattr(args, k)
        for k in (
            "model",
            "mode",
            "lens",
            "jlens_layer",
            "prompts",
            "n_neutral",
            "n_concept",
            "n_decomp",
            "topk",
            "decomp_k",
            "max_seq_len",
            "seed",
        )
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    out = args.out or default_out(args.model, args.lens)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    phase1_ckpt = out + ".phase1.pt"
    partial_path = out + ".partial.jsonl"
    key = _config_key(args)

    # Phase 1 (GPU capture) is checkpointed so a Phase-2 death skips the base
    # reload + capture entirely. Reuse only if the stored config matches exactly;
    # a recompute invalidates the AV sidecar (its prompt indices would be stale).
    per_prompt: list[dict[str, Any]] | None = None
    readouts: list[Tensor] = []
    tok_name: str = args.model
    vocab: int = 0
    if os.path.exists(phase1_ckpt):
        ck = torch.load(phase1_ckpt, weights_only=False)
        if ck.get("config_key") == key:
            per_prompt = ck["per_prompt"]
            readouts = ck["readouts"]
            tok_name = ck["tok_name"]
            vocab = ck["vocab"]
            print(f"[resume] Phase-1 checkpoint reused ({phase1_ckpt})")
    if per_prompt is None:
        per_prompt, readouts, tok_name, vocab = phase1_capture(args)
        torch.save(
            {
                "config_key": key,
                "per_prompt": per_prompt,
                "readouts": readouts,
                "tok_name": tok_name,
                "vocab": vocab,
            },
            phase1_ckpt,
        )
        if os.path.exists(partial_path):
            os.remove(partial_path)  # stale sidecar from a different config
        print(f"[ckpt] Phase-1 saved -> {phase1_ckpt}")

    if not args.skip_nla:
        phase2_verbalize(per_prompt, args, partial_path)

    # Reload only the base tokenizer for rank lookups (base freed after Phase 1;
    # readout is over the base W_U, so the base tokenizer is authoritative).
    from transformers import AutoTokenizer

    from llm_surgeon import surgery

    tok = AutoTokenizer.from_pretrained(tok_name, cache_dir=surgery.MODEL_CACHE_DIR)
    summary = score(per_prompt, readouts, tok, vocab, args)

    torch.save({"summary": summary, "per_prompt": per_prompt}, out)
    print_summary(summary)
    print(f"\n[save] {out}")


if __name__ == "__main__":
    main()
