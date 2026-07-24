#!/usr/bin/env python3
"""Stage 5.2 of the jspace arc: entailed-property swap (paper §3.3 spider->ant).

The strongest causal form of the workspace claim
`[gurnee2026-workspace §3.3; kb/excerpts/gurnee2026-workspace#sec-3-3-spider-ant]`:
a descriptor prompt forces the model to infer an **unspoken** concept and read
out a **property** of it. The paper's example --- "The number of legs on the
animal that spins webs is" -> the model infers *spider* (never written) -> emits
*8*; swapping the spider J-lens vector for *ant* flips the top output "8"->"6".
The measured readout is a *property* of the swapped concept, which plain token
steering cannot supply on its own. THE key comparison here: does the
raw-unembedding (``logitlens``) token-steering control *also* flip the property?
If only ``jlens`` does, J-space carries relational structure; if ``logitlens``
does too, the token-steering account (`nanda-workspace-review.md`) extends to
entailed properties.

Design + protocol: ``research/arcs/04_jspace/plans/2026-07-21-stage52-entailed-property.md``.
Machinery is shared with stage 5.1 (``jspace_verbal_report.py``): ``normalize``,
``jlens_atom``, ``concept_ids``, ``topk_ids``, ``rank_of``, ``generate``,
``build_model``.

Three magnitude-equalized swap conditions at swap layer ``L``, applied at the
**concept-holding positions** (mid-prompt, NOT the report position --- editing
the concept upstream and letting the model re-derive the property is the whole
point):

  1. ``jlens``     -- u_s = norm(a_s),  u_t = norm(a_t)   (paper §3.3)
  2. ``logitlens`` -- u_s = norm(w_s),  u_t = norm(w_t)   (Nanda token-steering)
  3. ``random``    -- u_s = norm(a_s),  u_t = norm(g)     (floor)

Equalization uses the **live prefill residuals** (same generate forward as the
injection) so there is no fp32-capture-vs-bf16-prefill magnitude leak --- the
defect stage 5.1b's logitlens condition hit and part-A fixed.

Usage (full GPU run, 7B peak layer):
    python examples/jspace_entailed_swap.py --model Qwen/Qwen2.5-7B-Instruct \
        --mode nf4 --device cuda --layer 22 --prompt-style plain \
        --lens research/arcs/04_jspace/data/cache/jlens_qwen2.5-7b_nf4_n100.pt

CPU smoke (plumbing, 2 items):
    python examples/jspace_entailed_swap.py --device cpu --limit 2 \
        --strengths 1.0 --max-new-tokens 4 --prompt-style plain
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import Tensor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from jspace_readout_scan import build_model, slug  # noqa: E402
from jspace_verbal_report import (  # noqa: E402
    _chat_ids,
    concept_ids,
    generate,
    jlens_atom,
    normalize,
    rank_of,
    topk_ids,
)

CONDITIONS = ("jlens", "logitlens", "random")
TOP5 = 5
TOP8 = 8
K_DETECT = 10  # J-lens top-k used to auto-detect concept-holding positions
STEP_EPS = 1e-12

# Entailed-property item bank. Each descriptor implies the concept via a defining
# property OTHER than the queried one, and never names the concept or the answer.
# concept/swap_concept must be space-prefixed single-token; answer/swap_answer
# single-token in some case/space form and DISTINCT within the pair (enforced at
# load by ``validate_items``). Tokenizer-validated against Qwen2.5 2026-07-21
# (penguin/wasp/Brasilia dropped for lacking the required single-token form).
ITEMS: list[dict[str, str]] = [
    # ---- legs / number (property = a single-digit count) ----
    {
        "family": "legs",
        "answer_kind": "number",
        "descriptor": "The number of legs on the creature that spins silky webs to trap flies",
        "tail": " is",
        "concept": "spider",
        "answer": "8",
        "swap_concept": "ant",
        "swap_answer": "6",
    },
    {
        "family": "legs",
        "answer_kind": "number",
        "descriptor": "The number of legs on the tiny insect that lives in colonies and marches in lines",
        "tail": " is",
        "concept": "ant",
        "answer": "6",
        "swap_concept": "spider",
        "swap_answer": "8",
    },
    {
        "family": "legs",
        "answer_kind": "number",
        "descriptor": "The number of legs on the busy insect that makes honey inside a hive",
        "tail": " is",
        "concept": "bee",
        "answer": "6",
        "swap_concept": "spider",
        "swap_answer": "8",
    },
    {
        "family": "legs",
        "answer_kind": "number",
        "descriptor": "The number of legs on the loyal animal that barks and guards the house",
        "tail": " is",
        "concept": "dog",
        "answer": "4",
        "swap_concept": "spider",
        "swap_answer": "8",
    },
    {
        "family": "legs",
        "answer_kind": "number",
        "descriptor": "The number of legs on the small animal that says meow and purrs on laps",
        "tail": " is",
        "concept": "cat",
        "answer": "4",
        "swap_concept": "spider",
        "swap_answer": "8",
    },
    {
        "family": "legs",
        "answer_kind": "number",
        "descriptor": "The number of legs on the large animal ridden in races and pulling carriages",
        "tail": " is",
        "concept": "horse",
        "answer": "4",
        "swap_concept": "ant",
        "swap_answer": "6",
    },
    {
        "family": "legs",
        "answer_kind": "number",
        "descriptor": "The number of legs on the green amphibian that hops and croaks by the pond",
        "tail": " is",
        "concept": "frog",
        "answer": "4",
        "swap_concept": "spider",
        "swap_answer": "8",
    },
    {
        "family": "moons",
        "answer_kind": "number",
        "descriptor": "The number of moons orbiting the small red planet named after the god of war",
        "tail": " is",
        "concept": "Mars",
        "answer": "2",
        "swap_concept": "Earth",
        "swap_answer": "1",
    },
    {
        "family": "wheels",
        "answer_kind": "number",
        "descriptor": "The number of wheels on the machine a cyclist pedals along the road",
        "tail": " is",
        "concept": "bicycle",
        "answer": "2",
        "swap_concept": "car",
        "swap_answer": "4",
    },
    # ---- capital via landmark (two-hop: landmark -> country[unspoken] -> capital) ----
    {
        "family": "capital",
        "answer_kind": "word",
        "descriptor": "The capital city of the country whose most famous landmark is the Eiffel Tower",
        "tail": " is",
        "concept": "France",
        "answer": "Paris",
        "swap_concept": "Italy",
        "swap_answer": "Rome",
    },
    {
        "family": "capital",
        "answer_kind": "word",
        "descriptor": "The capital city of the country home to the ancient Colosseum and gondolas",
        "tail": " is",
        "concept": "Italy",
        "answer": "Rome",
        "swap_concept": "France",
        "swap_answer": "Paris",
    },
    {
        "family": "capital",
        "answer_kind": "word",
        "descriptor": "The capital city of the country that built the Great Wall",
        "tail": " is",
        "concept": "China",
        "answer": "Beijing",
        "swap_concept": "Japan",
        "swap_answer": "Tokyo",
    },
    {
        "family": "capital",
        "answer_kind": "word",
        "descriptor": "The capital city of the island country famous for Mount Fuji and sushi",
        "tail": " is",
        "concept": "Japan",
        "answer": "Tokyo",
        "swap_concept": "China",
        "swap_answer": "Beijing",
    },
    {
        "family": "capital",
        "answer_kind": "word",
        "descriptor": "The capital city of the country where the great Pyramids of Giza stand",
        "tail": " is",
        "concept": "Egypt",
        "answer": "Cairo",
        "swap_concept": "Russia",
        "swap_answer": "Moscow",
    },
    {
        "family": "capital",
        "answer_kind": "word",
        "descriptor": "The capital city of the largest country on Earth, spanning Europe and Asia",
        "tail": " is",
        "concept": "Russia",
        "answer": "Moscow",
        "swap_concept": "Egypt",
        "swap_answer": "Cairo",
    },
    {
        "family": "capital",
        "answer_kind": "word",
        "descriptor": "The capital city of the country known for flamenco, paella, and running bulls",
        "tail": " is",
        "concept": "Spain",
        "answer": "Madrid",
        "swap_concept": "France",
        "swap_answer": "Paris",
    },
    {
        "family": "capital",
        "answer_kind": "word",
        "descriptor": "The capital city of the country of Oktoberfest, autobahns, and the Brandenburg Gate",
        "tail": " is",
        "concept": "Germany",
        "answer": "Berlin",
        "swap_concept": "Spain",
        "swap_answer": "Madrid",
    },
    {
        "family": "capital",
        "answer_kind": "word",
        "descriptor": "The capital city of the country that is the birthplace of democracy and the Parthenon",
        "tail": " is",
        "concept": "Greece",
        "answer": "Athens",
        "swap_concept": "Italy",
        "swap_answer": "Rome",
    },
    {
        "family": "capital",
        "answer_kind": "word",
        "descriptor": "The capital city of the country home to the Taj Mahal",
        "tail": " is",
        "concept": "India",
        "answer": "Delhi",
        "swap_concept": "China",
        "swap_answer": "Beijing",
    },
    {
        "family": "capital",
        "answer_kind": "word",
        "descriptor": "The capital city of the country home to Big Ben and the River Thames",
        "tail": " is",
        "concept": "England",
        "answer": "London",
        "swap_concept": "France",
        "swap_answer": "Paris",
    },
    {
        "family": "capital",
        "answer_kind": "word",
        "descriptor": "The capital city of the country famous for maple syrup, moose, and ice hockey",
        "tail": " is",
        "concept": "Canada",
        "answer": "Ottawa",
        "swap_concept": "France",
        "swap_answer": "Paris",
    },
    # ---- language via country (two-hop: landmark -> country[unspoken] -> language) ----
    {
        "family": "language",
        "answer_kind": "word",
        "descriptor": "The main language spoken in the country famous for the Eiffel Tower",
        "tail": " is",
        "concept": "France",
        "answer": "French",
        "swap_concept": "Spain",
        "swap_answer": "Spanish",
    },
    {
        "family": "language",
        "answer_kind": "word",
        "descriptor": "The main language spoken in the country known for the Colosseum and pizza",
        "tail": " is",
        "concept": "Italy",
        "answer": "Italian",
        "swap_concept": "France",
        "swap_answer": "French",
    },
    {
        "family": "language",
        "answer_kind": "word",
        "descriptor": "The main language spoken in the country that built the Great Wall",
        "tail": " is",
        "concept": "China",
        "answer": "Chinese",
        "swap_concept": "Japan",
        "swap_answer": "Japanese",
    },
    {
        "family": "language",
        "answer_kind": "word",
        "descriptor": "The main language spoken in the island country famous for Mount Fuji",
        "tail": " is",
        "concept": "Japan",
        "answer": "Japanese",
        "swap_concept": "China",
        "swap_answer": "Chinese",
    },
    {
        "family": "language",
        "answer_kind": "word",
        "descriptor": "The main language spoken in the country of flamenco and paella",
        "tail": " is",
        "concept": "Spain",
        "answer": "Spanish",
        "swap_concept": "Germany",
        "swap_answer": "German",
    },
    {
        "family": "language",
        "answer_kind": "word",
        "descriptor": "The main language spoken in the country of Oktoberfest and the autobahn",
        "tail": " is",
        "concept": "Germany",
        "answer": "German",
        "swap_concept": "Italy",
        "swap_answer": "Italian",
    },
    {
        "family": "language",
        "answer_kind": "word",
        "descriptor": "The main language spoken in the largest country on Earth",
        "tail": " is",
        "concept": "Russia",
        "answer": "Russian",
        "swap_concept": "Spain",
        "swap_answer": "Spanish",
    },
    {
        "family": "language",
        "answer_kind": "word",
        "descriptor": "The main language spoken in the country home to the Taj Mahal",
        "tail": " is",
        "concept": "India",
        "answer": "Hindi",
        "swap_concept": "China",
        "swap_answer": "Chinese",
    },
    # ---- color via description (property = a single-token color) ----
    {
        "family": "color",
        "answer_kind": "word",
        "descriptor": "The color of the round fruit that supposedly keeps the doctor away",
        "tail": " is",
        "concept": "apple",
        "answer": "red",
        "swap_concept": "banana",
        "swap_answer": "yellow",
    },
    {
        "family": "color",
        "answer_kind": "word",
        "descriptor": "The color of the long curved fruit that monkeys are said to love",
        "tail": " is",
        "concept": "banana",
        "answer": "yellow",
        "swap_concept": "cherry",
        "swap_answer": "red",
    },
    {
        "family": "color",
        "answer_kind": "word",
        "descriptor": "The color of the small round fruit crushed to make wine",
        "tail": " is",
        "concept": "grape",
        "answer": "purple",
        "swap_concept": "lemon",
        "swap_answer": "yellow",
    },
    {
        "family": "color",
        "answer_kind": "word",
        "descriptor": "The color of the ripe fruit blended into ketchup for burgers",
        "tail": " is",
        "concept": "tomato",
        "answer": "red",
        "swap_concept": "lemon",
        "swap_answer": "yellow",
    },
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Entailed-property swap (stage 5.2).")
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--mode", default="bf16", choices=["nf4", "int8", "bf16", "fp16"])
    p.add_argument("--device", default="cpu", choices=["cuda", "cpu"])
    p.add_argument(
        "--lens",
        default="research/arcs/04_jspace/data/cache/jlens_qwen2.5-1.5b_bf16_n100.pt",
        help="Path to a fitted JacobianLens .pt (JacobianLens.save format).",
    )
    p.add_argument(
        "--layer",
        type=int,
        default=21,
        help="Swap + readout layer (1.5B varfrac peak=21; 7B=22; sweep {peak,+-3}).",
    )
    p.add_argument(
        "--swap-scope",
        default="auto",
        choices=["auto", "window", "all", "report"],
        help="auto = J-lens-detected concept positions (paper diagnostic, default);"
        " window = last --swap-window prompt positions; all = all prompt positions;"
        " report = final position only (control). None include the report position"
        " except 'report'.",
    )
    p.add_argument("--swap-window", type=int, default=4)
    p.add_argument(
        "--strengths",
        default="1.0,2.0",
        help="Comma-separated swap strengths (paper: 1=equal-magnitude, 2=doubled).",
    )
    p.add_argument(
        "--prompt-style",
        default="plain",
        choices=["plain", "chat"],
        help="plain = raw declarative completion (paper-faithful); chat = "
        "chat-templated fill-the-blank instruction (compliance-hardened).",
    )
    p.add_argument("--max-new-tokens", type=int, default=6)
    p.add_argument("--max-seq-len", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--limit", type=int, default=0, help="Cap number of items (0 = all)."
    )
    p.add_argument(
        "--out",
        default=None,
        help="Output .pt (default auto-named by style + layer + lens stem).",
    )
    p.add_argument(
        "--items-json",
        default=None,
        help="Path to a JSON list of item dicts replacing the built-in bank "
        "(same schema; passed through validate_items). For prompt-variant "
        "probes; pair with --out to avoid overwriting bank-run artifacts.",
    )
    return p.parse_args()


def default_out(model: str, lens: str, style: str, layer: int) -> str:
    stem = Path(lens).stem
    return (
        f"research/arcs/04_jspace/data/cache/"
        f"entailed_swap_{style}_L{layer}_{slug(model)}_{stem}.pt"
    )


def space_single_token(tok: Any, word: str) -> int | None:
    """Space-prefixed single-token id for ``word`` (the swap handle), else None."""
    ids = tok(" " + word, add_special_tokens=False).input_ids
    return int(ids[0]) if len(ids) == 1 else None


def validate_items(tok: Any, items: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Keep items whose concept/swap_concept are space-prefixed single tokens and
    whose answer/swap_answer have a single-token form and DIFFER. Attach resolved
    token ids + answer-form id lists. Drops (with a printed reason) otherwise."""
    out: list[dict[str, Any]] = []
    for it in items:
        src_tid = space_single_token(tok, it["concept"])
        tgt_tid = space_single_token(tok, it["swap_concept"])
        ans_ids = concept_ids(tok, it["answer"])
        swp_ids = concept_ids(tok, it["swap_answer"])
        reason = ""
        if src_tid is None:
            reason = f"concept {it['concept']!r} not space-single-token"
        elif tgt_tid is None:
            reason = f"swap_concept {it['swap_concept']!r} not space-single-token"
        elif not ans_ids:
            reason = f"answer {it['answer']!r} has no single-token form"
        elif not swp_ids:
            reason = f"swap_answer {it['swap_answer']!r} has no single-token form"
        elif set(ans_ids) & set(swp_ids):
            reason = "answer and swap_answer share a token form (must differ)"
        if reason:
            print(
                f"[drop] {it['family']:8} {it['concept']}->{it['swap_concept']}: {reason}"
            )
            continue
        out.append(
            {
                **it,
                "source_tok": src_tid,
                "swap_tok": tgt_tid,
                "answer_ids": ans_ids,
                "swap_answer_ids": swp_ids,
            }
        )
    return out


def build_prompt(tok: Any, item: dict[str, Any], style: str) -> Tensor:
    """Prompt ids. plain = raw ``descriptor + tail`` completion (no special
    tokens); chat = chat-templated fill-the-blank instruction. In both the
    descriptor sits mid-prompt so the concept is held before the report token."""
    if style == "chat":
        content = (
            f"{item['descriptor']}{item['tail']} ___.\n"
            f"Reply with only the single {item['answer_kind']} that fills the "
            f"blank, and nothing else."
        )
        return _chat_ids(tok, content)
    text = f"{item['descriptor']}{item['tail']}"
    ids = tok(text, add_special_tokens=False).input_ids
    return torch.tensor([ids], dtype=torch.long)


class FullPrefillCapture:
    """Passive hook: records the FIRST call's full output ``[T, d]`` (the prefill)
    at the swap layer, in the same generate forward the injection sees. Serves
    both concept-position detection and the live-coeff magnitude equalization."""

    def __init__(self) -> None:
        self.h: Tensor | None = None

    def __call__(self, module: Any, inputs: Any, output: Any) -> Any:
        if self.h is None:
            tensor = output if torch.is_tensor(output) else output[0]
            self.h = tensor[0].detach().float()  # [T, d]
        return output


class EntailedSwapHook:
    """Swap ``s*coeff_p*(u_t-u_s)`` at the concept positions ``P`` on the PREFILL
    forward only (``h.shape[1] == prompt_len``); generation steps (T=1) are left
    alone --- the concept is already in the KV cache. ``coeff_p = <h_p, u_s>`` is
    read live pre-injection; ``scale`` equalizes the total injected L2 over ``P``
    to the jlens reference. Records the total injected norm (first call) and the
    per-position norms for the audit."""

    def __init__(
        self,
        u_s: Tensor,
        u_t: Tensor,
        strength: float,
        positions: list[int],
        prompt_len: int,
        scale: float = 1.0,
    ) -> None:
        self.u_s = u_s
        self.u_t = u_t
        self.strength = strength
        self.positions = positions
        self.prompt_len = prompt_len
        self.scale = scale
        self.injected_norm = float("nan")
        self.perpos_norm: list[float] = []
        self.perpos_coeff: list[float] = []

    def __call__(self, module: Any, inputs: Any, output: Any) -> Any:
        tensor = output if torch.is_tensor(output) else output[0]
        h = tensor  # [B, T, d]
        if h.shape[1] != self.prompt_len or not self.positions:
            return output  # generation step or empty position set: no-op
        idx = torch.tensor(self.positions, device=h.device)
        u_s = self.u_s.to(torch.float32).to(h.device)
        u_t = self.u_t.to(torch.float32).to(h.device)
        seg = h[:, idx, :].to(torch.float32)  # [B, P, d]
        coeff = seg @ u_s  # [B, P]
        delta = self.scale * self.strength * coeff.unsqueeze(-1) * (u_t - u_s)
        h[:, idx, :] = (seg + delta).to(h.dtype)
        if np.isnan(self.injected_norm):
            flat = delta.reshape(-1, delta.shape[-1])  # [B*P, d]
            self.injected_norm = float(flat.norm().item())
            self.perpos_norm = [float(x.norm().item()) for x in delta[0]]
            self.perpos_coeff = [float(x.item()) for x in coeff[0]]
        if torch.is_tensor(output):
            return h
        return (h,) + tuple(output[1:])


def perlayer_jlens(
    m: Any, lens: Any, prompt_ids: Tensor, positions: list[int], k: int
) -> dict[int, dict[int, dict[str, Any]]]:
    """Per-layer J-lens top-k (ids+strings) at each concept position on the CLEAN
    prompt --- makes the 'concept visible at intermediate layers' claim checkable.
    Returns {position: {layer: {'ids':[...], 'strs':[...]}}}."""
    from jlens.hooks import ActivationRecorder

    layers = list(lens.source_layers)
    with ActivationRecorder(m.layers, at=layers) as rec, torch.no_grad():
        m.forward(prompt_ids.to(m.input_device))
        acts = {ell: rec.activations[ell][0].detach().float() for ell in layers}
    out: dict[int, dict[int, dict[str, Any]]] = {}
    for p in positions:
        out[p] = {}
        for ell in layers:
            jac = lens.jacobians[ell].to(acts[ell].device)
            h_p = acts[ell][p]  # [d]
            logits = m.unembed((h_p @ jac.t()).unsqueeze(0))[0].float()
            ids = topk_ids(logits, k)
            out[p][ell] = {
                "ids": ids,
                "strs": [m.tokenizer.decode([i]).strip() for i in ids],
            }
    return out


def detect_positions(
    m: Any,
    lens: Any,
    layer: int,
    h_prefill: Tensor,
    report_pos: int,
    src_forms: set[int],
    scope: str,
    window: int,
) -> tuple[list[int], str, dict[int, int]]:
    """Concept-holding positions per --swap-scope. 'auto' keeps positions p <
    report_pos whose swap-layer J-lens top-k contains a source-concept form
    (paper diagnostic); falls back to the 'window' set if none qualify. Returns
    (positions, scope_used, {position: source-concept J-lens rank})."""
    jac = lens.jacobians[layer].to(h_prefill.device)
    ranks: dict[int, int] = {}
    # source-concept J-lens rank at every pre-report position (for the record).
    for p in range(report_pos):
        logits = m.unembed((h_prefill[p] @ jac.t()).unsqueeze(0))[0].float()
        ranks[p] = min(rank_of(logits, f) for f in src_forms)
    window_set = list(range(max(0, report_pos - window), report_pos))
    if scope == "report":
        return [report_pos], "report", ranks
    if scope == "all":
        return list(range(report_pos)), "all", ranks
    if scope == "window":
        return window_set, "window", ranks
    # auto
    detected = [p for p in range(report_pos) if ranks[p] < K_DETECT]
    if detected:
        return detected, "auto", ranks
    return window_set, "auto->window_fallback", ranks


def leading_word(text: str) -> str:
    """First alphanumeric word of a decoded report, lowercased ("" if none).

    The PRIMARY answer/flip test matches this against the answer word, faithful
    to the paper's "top output" framing and robust to turn-start tokenization: an
    instruct assistant emits a multi-piece word ("Beijing" -> "B","eijing") whose
    first token id is NOT the space-prefixed single-token answer handle, so a
    top-1-token test false-negatives every multi-token city/word answer. Matching
    the decoded word instead is tokenization-invariant.
    """
    mt = re.match(r"\s*([A-Za-z0-9']+)", text)
    return mt.group(1).lower() if mt else ""


def word_hit(text: str, word: str) -> bool:
    """The report's leading word equals ``word`` (case-insensitive)."""
    return leading_word(text) == word.lower()


def topk_record(logits: Tensor, k: int, tok: Any) -> dict[str, Any]:
    """Top-k ids + decoded strings + probabilities at a position."""
    probs = torch.softmax(logits, dim=-1)
    ids = topk_ids(logits, k)
    return {
        "ids": ids,
        "strs": [tok.decode([i]).strip() for i in ids],
        "probs": [float(probs[i]) for i in ids],
    }


def main() -> None:  # noqa: C901
    args = parse_args()
    torch.manual_seed(args.seed)

    from jlens import JacobianLens

    t_load = time.time()
    m = build_model(args)
    lens = JacobianLens.load(args.lens)
    if args.layer not in lens.source_layers:
        raise SystemExit(f"--layer {args.layer} not in fitted {lens.source_layers}")
    print(f"[load] model + lens in {time.time() - t_load:.1f}s; {lens!r}")

    tok = m.tokenizer
    mm = cast(Any, m)
    w_u = cast(Tensor, mm._lm_head.weight)
    if not (torch.is_tensor(w_u) and w_u.dim() == 2 and w_u.is_floating_point()):
        raise SystemExit(
            "lm_head.weight must be a plain 2-D float tensor (unquantized W_U)."
        )
    device = w_u.device
    jac = lens.jacobians[args.layer].to(device)
    strengths = [float(x) for x in args.strengths.split(",") if x.strip()]

    bank: list[dict[str, str]] = ITEMS
    if args.items_json:
        bank = json.loads(Path(args.items_json).read_text())
    items = validate_items(tok, bank)
    if args.limit > 0:
        items = items[: args.limit]
    print(
        f"[data] {len(items)}/{len(bank)} items valid, layer={args.layer}, "
        f"scope={args.swap_scope}, strengths={strengths}, style={args.prompt_style}"
    )

    per_item: list[dict[str, Any]] = []
    gen_tokens = 0
    n_correct = 0
    t_gen = time.time()

    for ii, item in enumerate(items):
        prompt_ids = build_prompt(tok, item, args.prompt_style)
        prompt_len = int(prompt_ids.shape[1])
        report_pos = prompt_len - 1
        src_tid = int(item["source_tok"])
        tgt_tid = int(item["swap_tok"])
        ans_ids: list[int] = item["answer_ids"]
        swp_ids: list[int] = item["swap_answer_ids"]

        # Baseline generate + full prefill capture at the swap layer.
        cap = FullPrefillCapture()
        cap_handle = m.layers[args.layer].register_forward_hook(cap)
        try:
            base_tok0, base_scores, base_text = generate(
                m, prompt_ids, args.max_new_tokens
            )
        finally:
            cap_handle.remove()
        assert cap.h is not None, "FullPrefillCapture did not fire"
        h_prefill = cap.h.to(device)  # [T, d]
        gen_tokens += args.max_new_tokens

        base_top1 = topk_ids(base_scores, 1)[0]
        base_lead = leading_word(base_text)
        # PRIMARY correctness = decoded leading word matches the answer word
        # (tokenization-robust; see leading_word). base_top1_correct is the
        # strict single-token variant, kept for single-token (digit) answers.
        baseline_correct = word_hit(base_text, item["answer"])
        base_top1_correct = base_top1 in set(ans_ids)
        n_correct += int(baseline_correct)
        base_logp = torch.log_softmax(base_scores, dim=-1)
        base_logp_clean = max(float(base_logp[i]) for i in ans_ids)
        base_logp_swap = max(float(base_logp[i]) for i in swp_ids)

        src_forms = set(concept_ids(tok, item["concept"])) | {src_tid}
        positions, scope_used, perpos_rank = detect_positions(
            m,
            lens,
            args.layer,
            h_prefill,
            report_pos,
            src_forms,
            args.swap_scope,
            args.swap_window,
        )
        clean_perlayer = perlayer_jlens(m, lens, prompt_ids, positions, TOP8)

        # Directions.
        u_s_j = normalize(jlens_atom(w_u, jac, src_tid))
        u_t_j = normalize(jlens_atom(w_u, jac, tgt_tid))
        u_s_logit = normalize(w_u[src_tid].to(torch.float32))
        u_t_logit = normalize(w_u[tgt_tid].to(torch.float32))
        gen_r = torch.Generator(device="cpu").manual_seed(args.seed + tgt_tid)
        u_t_rand = normalize(torch.randn(w_u.shape[1], generator=gen_r).to(device))
        cond_dirs = {
            "jlens": (u_s_j, u_t_j),
            "logitlens": (u_s_logit, u_t_logit),
            "random": (u_s_j, u_t_rand),
        }

        # Equalize total injected L2 over the position set to the jlens reference,
        # using the LIVE prefill coeffs (same forward as injection -> no fp32/bf16
        # magnitude leak). total L2 = ||u_t-u_s|| * ||[<h_p,u_s>]_p||.
        pos_t = torch.tensor(positions, device=device)

        def total_l2(u_s: Tensor, u_t: Tensor) -> float:
            coeffs = h_prefill[pos_t] @ u_s  # [P]
            return float((u_t - u_s).norm()) * float(coeffs.norm())

        ref_l2 = total_l2(u_s_j, u_t_j)

        conds: dict[str, dict[str, Any]] = {}
        for cond in CONDITIONS:
            u_s, u_t = cond_dirs[cond]
            d_c = total_l2(u_s, u_t)
            scale = (ref_l2 / d_c) if d_c > STEP_EPS else 0.0
            for s in strengths:
                hook = EntailedSwapHook(u_s, u_t, s, positions, prompt_len, scale=scale)
                handle = m.layers[args.layer].register_forward_hook(hook)
                try:
                    tok0, scores0, text = generate(m, prompt_ids, args.max_new_tokens)
                finally:
                    handle.remove()
                gen_tokens += args.max_new_tokens
                top1 = topk_ids(scores0, 1)[0]
                top5 = topk_ids(scores0, TOP5)
                logp = torch.log_softmax(scores0, dim=-1)
                lead = leading_word(text)
                conds[f"{cond}@{s}"] = {
                    "condition": cond,
                    "strength": s,
                    "report_tok0": tok0,
                    "report_text": text,
                    "report_top1": top1,
                    "report_lead": lead,
                    "topk": topk_record(scores0, TOP8, tok),
                    # PRIMARY: leading decoded word == swap concept's property.
                    "property_flip": bool(word_hit(text, item["swap_answer"])),
                    # strict single-token variants (meaningful for digit answers).
                    "property_flip_tok1": bool(top1 in set(swp_ids)),
                    "swap_answer_in_top5": bool(any(t in top5 for t in swp_ids)),
                    "answer_changed": bool(lead != base_lead),
                    "clean_retained": bool(lead == base_lead),
                    "logp_swap_answer": max(float(logp[i]) for i in swp_ids),
                    "logp_clean_answer": max(float(logp[i]) for i in ans_ids),
                    "dlogp_swap_answer": max(float(logp[i]) for i in swp_ids)
                    - base_logp_swap,
                    "dlogp_clean_answer": max(float(logp[i]) for i in ans_ids)
                    - base_logp_clean,
                    "scale": scale,
                    "injected_norm": hook.injected_norm,
                    "perpos_norm": hook.perpos_norm,
                    "perpos_coeff": hook.perpos_coeff,
                }

        per_item.append(
            {
                "family": item["family"],
                "descriptor": item["descriptor"],
                "tail": item["tail"],
                "concept": item["concept"],
                "answer": item["answer"],
                "swap_concept": item["swap_concept"],
                "swap_answer": item["swap_answer"],
                "source_tok": src_tid,
                "swap_tok": tgt_tid,
                "answer_ids": ans_ids,
                "swap_answer_ids": swp_ids,
                "prompt_len": prompt_len,
                "report_pos": report_pos,
                "baseline_text": base_text,
                "baseline_tok0": base_tok0,
                "baseline_top1": base_top1,
                "baseline_lead": base_lead,
                "baseline_topk": topk_record(base_scores, TOP8, tok),
                "baseline_correct": bool(baseline_correct),
                "baseline_top1_correct": bool(base_top1_correct),
                "base_logp_clean": base_logp_clean,
                "base_logp_swap": base_logp_swap,
                "swap_positions": positions,
                "scope_used": scope_used,
                "source_jlens_rank": perpos_rank,
                "clean_perlayer_jlens": clean_perlayer,
                "conditions": conds,
            }
        )
        dt = time.time() - t_gen
        flip1 = per_item[-1]["conditions"].get("jlens@1.0", {}).get("property_flip")
        print(
            f"[item {ii + 1}/{len(items)}] {item['family']:8} "
            f"{item['concept']}->{item['swap_concept']} base={base_text!r} "
            f"correct={baseline_correct} scope={scope_used}({len(positions)}) "
            f"jlens@1_flip={flip1} ({dt:.1f}s cum)"
        )

    gen_elapsed = time.time() - t_gen
    tok_per_s = gen_tokens / gen_elapsed if gen_elapsed > 0 else float("nan")

    # ---- Aggregate over the baseline-correct subset ----
    correct = [it for it in per_item if it["baseline_correct"]]
    cond_keys = sorted({k for it in per_item for k in it["conditions"]})

    def rate(pred: Any, key: str) -> float:
        vals = [
            pred(it["conditions"][key]) for it in correct if key in it["conditions"]
        ]
        return float(np.mean(vals)) if vals else float("nan")

    metrics: dict[str, dict[str, Any]] = {}
    for key in cond_keys:
        metrics[key] = {
            "n_correct": len(correct),
            "property_flip_rate": rate(lambda c: c["property_flip"], key),
            "property_flip_tok1_rate": rate(lambda c: c["property_flip_tok1"], key),
            "swap_answer_top5_rate": rate(lambda c: c["swap_answer_in_top5"], key),
            "answer_changed_rate": rate(lambda c: c["answer_changed"], key),
            "clean_retention_rate": rate(lambda c: c["clean_retained"], key),
            "mean_dlogp_swap_answer": rate(lambda c: c["dlogp_swap_answer"], key),
            "mean_dlogp_clean_answer": rate(lambda c: c["dlogp_clean_answer"], key),
            "mean_injected_norm": rate(lambda c: c["injected_norm"], key),
        }

    # ---- Per-scope split over the baseline-correct subset (auto vs fallback) ----
    # 'auto' positions are the genuine J-lens-detected concept positions (>=1
    # source-concept form in the swap-layer top-k); 'auto->window_fallback' items
    # had none detected and fell back to a fixed positional window, which carries
    # near-zero entailed-property movement. Recorded additively so downstream
    # readers can read the auto-only aggregate (the true J-space-localized effect)
    # without re-deriving from per_item; existing keys are untouched.
    def rate_over(subset: list[dict[str, Any]], pred: Any, key: str) -> float:
        vals = [pred(it["conditions"][key]) for it in subset if key in it["conditions"]]
        return float(np.mean(vals)) if vals else float("nan")

    n_auto = sum(1 for it in per_item if it["scope_used"] == "auto")
    n_fallback = sum(
        1 for it in per_item if it["scope_used"] == "auto->window_fallback"
    )
    auto_correct = [it for it in correct if it["scope_used"] == "auto"]
    fb_correct = [it for it in correct if it["scope_used"] == "auto->window_fallback"]
    metrics_by_scope: dict[str, dict[str, dict[str, Any]]] = {}
    for scope_name, subset in (
        ("auto", auto_correct),
        ("auto->window_fallback", fb_correct),
    ):
        metrics_by_scope[scope_name] = {
            key: {
                "n": len(subset),
                "property_flip_rate": rate_over(
                    subset, lambda c: c["property_flip"], key
                ),
                "swap_answer_top5_rate": rate_over(
                    subset, lambda c: c["swap_answer_in_top5"], key
                ),
                "clean_retention_rate": rate_over(
                    subset, lambda c: c["clean_retained"], key
                ),
                "mean_dlogp_swap_answer": rate_over(
                    subset, lambda c: c["dlogp_swap_answer"], key
                ),
                "mean_dlogp_clean_answer": rate_over(
                    subset, lambda c: c["dlogp_clean_answer"], key
                ),
            }
            for key in cond_keys
        }

    summary: dict[str, Any] = {
        "model": args.model,
        "mode": args.mode,
        "lens": args.lens,
        "layer": args.layer,
        "swap_scope": args.swap_scope,
        "swap_window": args.swap_window,
        "strengths": strengths,
        "prompt_style": args.prompt_style,
        "n_items": len(per_item),
        "n_correct": len(correct),
        "n_auto": n_auto,
        "n_fallback": n_fallback,
        "n_auto_correct": len(auto_correct),
        "n_fallback_correct": len(fb_correct),
        "metrics_by_scope": metrics_by_scope,
        "clean_accuracy": (len(correct) / len(per_item)) if per_item else float("nan"),
        "n_correct_top1": int(sum(1 for it in per_item if it["baseline_top1_correct"])),
        "clean_accuracy_top1": (
            sum(1 for it in per_item if it["baseline_top1_correct"]) / len(per_item)
            if per_item
            else float("nan")
        ),
        "conditions": list(CONDITIONS),
        "metrics": metrics,
        "gen_tokens": gen_tokens,
        "gen_seconds": gen_elapsed,
        "tokens_per_second": tok_per_s,
        "swap_note": (
            "Entailed-property swap (paper §3.3): edit the concept direction at the "
            "concept-holding positions (NOT the report position), measure whether the "
            "single-token PROPERTY readout flips to the swap concept's property. "
            "jlens = J-lens vectors a_v=W_U[v]@J_l; logitlens = raw unembedding rows "
            "(token-steering control); random = Gaussian target. Total injected L2 "
            "equalized over the position set to the jlens reference per item/strength "
            "using live prefill coeffs. Flip metrics over baseline-correct subset."
        ),
    }

    out = args.out or default_out(args.model, args.lens, args.prompt_style, args.layer)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"per_item": per_item, "summary": summary}, out)

    print(
        f"\n=== Entailed-property swap (layer {args.layer}, style={args.prompt_style}, "
        f"scope={args.swap_scope}) ==="
    )
    print(
        f"clean accuracy (baseline precondition): {summary['clean_accuracy']:.3f} "
        f"({len(correct)}/{len(per_item)} items)"
    )
    print(
        f"{'condition@s':>16} {'flip':>7} {'swp_t5':>7} {'changed':>8} "
        f"{'retain':>7} {'dlp_swp':>8} {'dlp_cln':>8} {'inj':>7}"
    )
    for key in cond_keys:
        mk = metrics[key]
        print(
            f"{key:>16} {mk['property_flip_rate']:>7.3f} "
            f"{mk['swap_answer_top5_rate']:>7.3f} {mk['answer_changed_rate']:>8.3f} "
            f"{mk['clean_retention_rate']:>7.3f} {mk['mean_dlogp_swap_answer']:>8.2f} "
            f"{mk['mean_dlogp_clean_answer']:>8.2f} {mk['mean_injected_norm']:>7.2f}"
        )
    ja = metrics_by_scope["auto"].get("jlens@2.0", {})
    print(
        f"\n[scope] baseline-correct split: auto={len(auto_correct)} "
        f"fallback={len(fb_correct)} (all items: auto={n_auto} fallback={n_fallback}); "
        f"auto-only jlens@2 dlogp(swap)={ja.get('mean_dlogp_swap_answer', float('nan')):.3f}"
    )
    print(f"\n[save] {out}")
    print(
        f"[time] {gen_tokens} gen-tokens in {gen_elapsed:.1f}s "
        f"= {tok_per_s:.2f} tok/s ({args.device})"
    )


if __name__ == "__main__":
    main()
