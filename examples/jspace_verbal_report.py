#!/usr/bin/env python3
"""Stage 5.1 of the jspace arc: verbal report + J-lens swap (paper §3.1).

Protocol (`[gurnee2026-workspace §3.1; kb/notes/interpretability/j-space.md §2]`):
prompt the instruct model to "think of a {category}" and report it in one word;
read the J-lens at the workspace-band layer to confirm the pre-report readout
predicts the reported instance; then intervene on the residual stream along a
**swap** direction (subtract the projection onto the reported-source lens vector,
add an equal-magnitude projection onto a target lens vector) and measure whether
the swap target enters the top-5 of the model's report distribution.

Four swap conditions, magnitude-matched, all applied at the same layer ``l`` and
report position(s). For a residual ``h`` at the injection position, with unit
source/target directions ``u_s``/``u_t`` and strength ``s`` (paper's
"equal-magnitude" is ``s=1``; §3.4 reports higher rates at doubled strength):

    coeff = <h, u_s>                      (true source projection, live)
    h'    = h + s * coeff * (u_t - u_s)    (remove source, add equal onto target)

The four conditions differ only in how ``(u_s, u_t)`` are built from the source
token ``s`` (the instance the model actually reported) and target token ``t`` (a
different single-token instance of the same category). ``w_v = W_U[v]`` is the
unembedding row; the **J-lens vector** for token ``v`` at layer ``l`` is the
input direction ``a_v = J_l^T w_v = W_U[v] @ J_l`` `[j-space.md §1.1]` (same atom
as ``jspace_structure_scan``):

  1. ``jlens``      — u_s = norm(a_s),  u_t = norm(a_t)            [paper 88%]
  2. ``nonjspace``  — u_s = norm(a_s),  u_t = norm(w_t - Pi_J w_t) [paper 5% ctrl]
                      where ``Pi_J w_t`` is the k-sparse nonneg gradient-pursuit
                      J-space component of the target's unembedding row.
  3. ``random``     — u_s = norm(a_s),  u_t = norm(g), g ~ N(0,I) (per-item seed)
  4. ``logitlens``  — u_s = norm(w_s),  u_t = norm(w_t)  — the Nanda "just token
                      steering" control (`sources/forums/2026-07-06-nanda-
                      workspace-review.md`): if this matches ``jlens``, the swap
                      is plain unembedding-direction steering, not workspace.

Stage 5.1b adds two OPT-IN activation-contrastive conditions (``--contrastive``),
the paper's full triplet against the base ``jlens`` (88%) row. Per target
instance a **contrastive concept vector** ``v_t`` is captured at layer ``L``:
mean of ``h_L`` at the final prompt position over 3 instance paraphrases
(``INSTANCE_TEMPLATES``) minus the mean over 3 category paraphrases
(``CATEGORY_TEMPLATES``). Then:

  5. ``jspace_comp``    — u_s = norm(a_s), u_t = norm(Pi_J v_t)       [paper 59%]
  6. ``nonjspace_comp`` — u_s = norm(a_s), u_t = norm(v_t - Pi_J v_t) [paper 5%]
                          where ``Pi_J v_t`` is the k=25 gradient-pursuit J-space
                          component of ``v_t`` (activation-space, not ``w_t``).

``--prompt-style {plain,chat}`` (default ``plain`` = the committed stage-5.1
wording). ``chat`` adds an explicit one-word constraint to raise single-word
compliance (7B stage-5.1 produced many word-piece "fragment" reports). Both go
through the tokenizer chat template with ``add_generation_prompt=True``, so the
report locus is the final prefill token in either style; the exact templates are
documented in ``2026-07-20-stage5-design.md`` §8. Baseline single-word
compliance is measured as ``fragment_report_rate`` (fraction of items whose
baseline report matches no in-category instance).

Success metrics, per condition, over items (and over the subset where the
baseline J-lens top-1 predicts the report, the paper's "answers correctly"
filter): swap-target top-5 entry rate in the report distribution (paper metric);
report-changed rate; report-equals-target rate.

Artifacts follow the arc's ``{"summary", "per_item"}`` convention; output path is
auto-named by lens stem + prompt-style + condition-set (never hardcoded, never
clobbering the committed plain 4-condition artifacts — this arc had a clobber
incident).

Usage (full GPU run, stage 5.1b chat + 6 conditions):
    python examples/jspace_verbal_report.py --device cuda \
        --model Qwen/Qwen2.5-7B-Instruct --mode nf4 --layer 22 \
        --lens research/arcs/jspace/data/cache/jlens_qwen2.5-7b_nf4_n100.pt \
        --prompt-style chat --contrastive

CPU smoke (no CUDA):
    python examples/jspace_verbal_report.py --device cpu \
        --categories sport,fruit --targets-per-category 1 --strengths 1.0 \
        --max-new-tokens 4 --prompt-style chat --contrastive
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import Tensor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from jspace_readout_scan import build_model, slug  # noqa: E402

BASE_CONDITIONS = ("jlens", "nonjspace", "random", "logitlens")
# Stage 5.1b: activation-contrastive concept-vector conditions (opt-in via
# --contrastive). jspace_comp / nonjspace_comp map to the paper's 59% / 5% rows;
# the base jlens condition maps to the 88% row (paper §3.1, Fig. 8).
CONTRASTIVE_CONDITIONS = ("jspace_comp", "nonjspace_comp")
TOP5 = 5
TOP10 = 10
STEP_EPS = 1e-12
GP_K = 25  # gradient-pursuit sparsity for the non-J-space component (paper default)

# Paraphrase templates for the activation-contrastive concept vector (stage 5.1b):
# v_t = mean_L(instance prompts) - mean_L(category prompts) at the final position.
INSTANCE_TEMPLATES = (
    "Think of {w}.",
    "You are thinking about {w}.",
    "Picture {w} in your mind.",
)
CATEGORY_TEMPLATES = (
    "Think of a {w}.",
    "You are thinking about a {w}.",
    "Picture a {w} in your mind.",
)

# Candidate instances per category; the actual run keeps only those that encode
# to a single Qwen token (checked at load by ``single_token_instances``). Counts
# validated against the Qwen2.5 tokenizer 2026-07-20 (26 cats, 267 survivors).
CATEGORIES: dict[str, list[str]] = {
    "sport": [
        "Soccer",
        "Rugby",
        "Tennis",
        "Golf",
        "Boxing",
        "Cricket",
        "Hockey",
        "Baseball",
        "Basketball",
        "Swimming",
        "Cycling",
        "Football",
        "Wrestling",
    ],
    "fruit": [
        "Apple",
        "Banana",
        "Orange",
        "Mango",
        "Grape",
        "Cherry",
        "Lemon",
        "Peach",
        "Pear",
        "Plum",
        "Lime",
        "Berry",
        "Fig",
        "Date",
        "Coconut",
    ],
    "animal": [
        "Dog",
        "Cat",
        "Horse",
        "Cow",
        "Sheep",
        "Goat",
        "Pig",
        "Lion",
        "Tiger",
        "Bear",
        "Wolf",
        "Fox",
        "Deer",
        "Rabbit",
        "Mouse",
        "Elephant",
        "Monkey",
    ],
    "color": [
        "Red",
        "Blue",
        "Green",
        "Yellow",
        "Purple",
        "Orange",
        "Pink",
        "Black",
        "White",
        "Brown",
        "Gray",
        "Violet",
        "Cyan",
        "Gold",
        "Silver",
    ],
    "country": [
        "France",
        "China",
        "Japan",
        "Brazil",
        "Canada",
        "Egypt",
        "India",
        "Spain",
        "Italy",
        "Germany",
        "Russia",
        "Mexico",
        "Kenya",
        "Chile",
        "Peru",
        "Norway",
        "Poland",
        "Turkey",
    ],
    "metal": [
        "Iron",
        "Gold",
        "Silver",
        "Copper",
        "Zinc",
        "Lead",
        "Tin",
        "Nickel",
        "Steel",
        "Bronze",
        "Platinum",
        "Aluminum",
        "Titanium",
        "Mercury",
    ],
    "instrument": ["Piano", "Guitar", "Drum", "Organ"],
    "vegetable": ["Potato", "Onion", "Tomato", "Pepper", "Corn", "Bean", "Garlic"],
    "tree": [
        "Oak",
        "Pine",
        "Maple",
        "Birch",
        "Willow",
        "Cedar",
        "Elm",
        "Ash",
        "Palm",
        "Fir",
        "Cherry",
        "Walnut",
    ],
    "bird": [
        "Eagle",
        "Robin",
        "Owl",
        "Hawk",
        "Crow",
        "Dove",
        "Swan",
        "Duck",
        "Goose",
        "Falcon",
        "Penguin",
        "Raven",
        "Finch",
    ],
    "drink": [
        "Water",
        "Coffee",
        "Tea",
        "Juice",
        "Milk",
        "Wine",
        "Beer",
        "Soda",
        "Cocoa",
    ],
    "vehicle": [
        "Car",
        "Truck",
        "Bus",
        "Train",
        "Plane",
        "Boat",
        "Bike",
        "Ship",
        "Van",
        "Jeep",
        "Motorcycle",
    ],
    "planet": [
        "Mercury",
        "Venus",
        "Earth",
        "Mars",
        "Jupiter",
        "Saturn",
        "Neptune",
        "Pluto",
    ],
    "gemstone": [
        "Diamond",
        "Ruby",
        "Emerald",
        "Sapphire",
        "Pearl",
        "Amber",
        "Jade",
        "Quartz",
    ],
    "flower": ["Rose", "Daisy", "Lily", "Iris", "Violet", "Jasmine", "Lotus"],
    "tool": ["Hammer", "Saw", "Drill", "Axe", "File", "Clamp"],
    "insect": ["Ant", "Bee", "Beetle", "Fly", "Butterfly", "Spider", "Cricket"],
    "fish": ["Salmon", "Trout", "Cod", "Bass", "Shark", "Carp", "Pike"],
    "shape": [
        "Circle",
        "Square",
        "Triangle",
        "Rectangle",
        "Oval",
        "Diamond",
        "Pentagon",
        "Star",
        "Cube",
        "Sphere",
        "Cone",
    ],
    "emotion": ["Joy", "Fear", "Love", "Hate", "Hope", "Pride", "Shame", "Surprise"],
    "language": [
        "English",
        "French",
        "Spanish",
        "German",
        "Chinese",
        "Japanese",
        "Arabic",
        "Russian",
        "Italian",
        "Hindi",
        "Korean",
        "Latin",
    ],
    "dance": ["Ballet", "Tango", "Jazz", "Swing"],
    "weather": [
        "Rain",
        "Snow",
        "Wind",
        "Storm",
        "Fog",
        "Sunshine",
        "Thunder",
        "Frost",
        "Cloud",
    ],
    "occupation": [
        "Doctor",
        "Teacher",
        "Lawyer",
        "Nurse",
        "Engineer",
        "Farmer",
        "Chef",
        "Pilot",
        "Artist",
        "Writer",
        "Judge",
        "Baker",
    ],
    "weapon": [
        "Sword",
        "Knife",
        "Gun",
        "Spear",
        "Bow",
        "Axe",
        "Dagger",
        "Rifle",
        "Cannon",
        "Pistol",
        "Bomb",
    ],
    "furniture": [
        "Chair",
        "Table",
        "Sofa",
        "Bed",
        "Desk",
        "Shelf",
        "Bench",
        "Cabinet",
        "Couch",
    ],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="J-lens verbal report + swap (stage 5.1).")
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--mode", default="bf16", choices=["nf4", "int8", "bf16", "fp16"])
    p.add_argument("--device", default="cpu", choices=["cuda", "cpu"])
    p.add_argument(
        "--lens",
        default="research/arcs/jspace/data/cache/jlens_qwen2.5-1.5b_bf16_n100.pt",
        help="Path to a fitted JacobianLens .pt (JacobianLens.save format).",
    )
    p.add_argument(
        "--layer",
        type=int,
        default=21,
        help="Injection + readout layer (1.5B varfrac peak=21; 7B peak=22; "
        "workspace band L19-24).",
    )
    p.add_argument(
        "--categories",
        default=None,
        help="Comma-separated subset of categories (default: all validated cats).",
    )
    p.add_argument(
        "--targets-per-category",
        type=int,
        default=3,
        help="Swap targets per category.",
    )
    p.add_argument(
        "--target-mode",
        default="random",
        choices=["random", "leastfav"],
        help="random = seeded plausible-sibling instances (paper-comparable, "
        "default); leastfav = least-favoured baseline instances (conservative).",
    )
    p.add_argument(
        "--strengths",
        default="1.0,2.0",
        help="Comma-separated swap strengths s (paper: 1=equal-magnitude, 2=doubled).",
    )
    p.add_argument(
        "--scope",
        default="last",
        choices=["last", "all"],
        help="Inject at the last position of each forward chunk (report locus) "
        "or at all positions (persistent steer).",
    )
    p.add_argument("--max-new-tokens", type=int, default=6)
    p.add_argument("--max-seq-len", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--prompt-style",
        default="plain",
        choices=["plain", "chat"],
        help="plain = committed stage-5.1 wording; chat = adds an explicit "
        "one-word constraint (higher single-word compliance).",
    )
    p.add_argument(
        "--contrastive",
        action="store_true",
        help="Add the activation-contrastive jspace_comp / nonjspace_comp "
        "conditions (paper's 59%%/5%% rows); off = committed 4-condition set.",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Output .pt (default auto-named by style + condition-set + lens stem).",
    )
    return p.parse_args()


def default_out(model: str, lens: str, style: str, contrastive: bool) -> str:
    stem = Path(lens).stem
    condset = "6c" if contrastive else "4c"
    return (
        f"research/arcs/jspace/data/cache/"
        f"verbal_report_{style}_{condset}_{slug(model)}_{stem}.pt"
    )


def single_token_instances(tok: Any, words: list[str]) -> list[tuple[str, int]]:
    """Keep words whose space-prefixed form encodes to exactly one token.

    The J-lens vectors are token-indexed `[gurnee2026-workspace §9.1]`, so only
    single-token instances have a well-defined swap direction. The space-prefixed
    form (`" Soccer"`) is the canonical mid-text encoding; a per-position id
    variant (unspaced/lowercase) is resolved separately by ``concept_ids``.
    """
    out: list[tuple[str, int]] = []
    for w in words:
        ids = tok(" " + w, add_special_tokens=False).input_ids
        if len(ids) == 1:
            out.append((w, int(ids[0])))
    return out


def concept_ids(tok: Any, word: str) -> list[int]:
    """Single-token ids for ``word`` across case + leading-space variants.

    A concept surfaces in several English token forms: space-prefixed vs bare
    (turn-start vs mid-text), capitalised vs lowercase. The top-5 metric and the
    "predicts" diagnostic accept any of them. Cross-lingual synonyms (e.g. 足球
    for *soccer*) are a KNOWN uncounted form — see the design addendum's
    multilingual-confound note; this stays English-only by design.
    """
    ids: list[int] = []
    forms = {
        " " + word,
        word,
        " " + word.lower(),
        word.lower(),
        " " + word.capitalize(),
        word.capitalize(),
    }
    for form in forms:
        enc = tok(form, add_special_tokens=False).input_ids
        if len(enc) == 1 and int(enc[0]) not in ids:
            ids.append(int(enc[0]))
    return ids


def match_instance(text: str, insts: list[tuple[str, int]]) -> tuple[str, int] | None:
    """Which category instance the report names (case-insensitive substring).

    At an assistant-turn start Qwen emits words unspaced/multi-piece
    (``"Soccer"`` -> ``"S","occer"``), so the first generated token id is not a
    reliable instance handle. We instead match the decoded report text against
    the category vocab and use the instance's canonical space-prefixed id (a
    well-defined J-lens vector ``a_v = W_U[v] @ J_l``) as the source handle.
    Longest word first, so ``"Basketball"`` wins over a spurious ``"Ball"``.
    """
    low = text.lower()
    for word, tid in sorted(insts, key=lambda wt: -len(wt[0])):
        if word.lower() in low:
            return word, tid
    return None


def normalize(v: Tensor) -> Tensor:
    n = v.norm()
    return v / n if float(n) > 0 else v


def jlens_atom(w_u: Tensor, jac: Tensor, tid: int) -> Tensor:
    """J-lens vector a_v = W_U[v] @ J_l in layer-l space (fp32)."""
    return w_u[tid].to(torch.float32) @ jac


def jspace_component(target: Tensor, jac: Tensor, w_u: Tensor, k: int) -> Tensor:
    """k-sparse nonneg gradient-pursuit J-space component of ``target`` (fp32).

    Single-vector form of ``jspace_structure_scan.gradient_pursuit_layer``: greedy
    atom selection by correlation ``c = W_U (J_l r)`` (eq. 1 there), one gradient
    step per atom, nonnegativity projection; returns the reconstruction ``A x``
    (the nearest sparse-nonneg cone point). Atoms materialised only when selected,
    so peak extra memory is O(k*d).
    """
    device = target.device
    d = target.shape[0]
    jac_t = jac.t()
    wu_t = w_u.t()
    wu_dtype = w_u.dtype
    y = target
    r = y.clone()
    sel: list[int] = []
    atoms = target.new_zeros((0, d))
    coeffs = target.new_zeros((0,))
    for _ in range(k):
        c = (r @ jac_t).to(wu_dtype) @ wu_t  # [|V|]
        c = c.float()
        if sel:
            c[torch.tensor(sel, device=device)] = float("-inf")
        j = int(c.argmax().item())
        if float(c[j]) <= 0.0:
            break
        a_j = (w_u[j].to(torch.float32) @ jac).unsqueeze(0)
        atoms = torch.cat([atoms, a_j], dim=0)
        coeffs = torch.cat([coeffs, coeffs.new_zeros(1)])
        sel.append(j)
        g = atoms @ r  # [m]
        ag = g @ atoms  # [d]
        denom = float((ag * ag).sum())
        alpha = float((g * g).sum()) / denom if denom > STEP_EPS else 0.0
        coeffs = torch.clamp(coeffs + alpha * g, min=0.0)
        r = y - coeffs @ atoms
    return coeffs @ atoms if coeffs.numel() else target.new_zeros(d)


class SwapHook:
    """Forward hook that applies the magnitude-equalized swap at ``layer``.

    The raw swap direction is ``s * coeff * (u_t - u_s)`` with ``coeff = <h,u_s>``
    read live; ``scale`` rescales it so the injected L2 at the report position
    matches the jlens-condition reference for this item/strength (ruling 3).
    Records the injection at the **first** call (the prefill last position = the
    report/metric locus, equal across conditions by construction) and the last
    call, for the per-trial sanity audit. ``scope='last'`` edits only the final
    position of each forward chunk (report locus, and the single new token each
    generation step); ``scope='all'`` edits every position.
    """

    def __init__(
        self, u_s: Tensor, u_t: Tensor, strength: float, scope: str, scale: float = 1.0
    ) -> None:
        self.u_s = u_s
        self.u_t = u_t
        self.strength = strength
        self.scope = scope
        self.scale = scale
        self.first_coeff = float("nan")
        self.first_norm = float("nan")
        self.last_coeff = float("nan")
        self.last_norm = float("nan")

    def __call__(self, module: Any, inputs: Any, output: Any) -> Any:
        tensor = output if torch.is_tensor(output) else output[0]
        h = tensor  # [B, T, d]
        u_s = self.u_s.to(h.dtype).to(h.device)
        u_t = self.u_t.to(h.dtype).to(h.device)
        idx = slice(None) if self.scope == "all" else slice(h.shape[1] - 1, h.shape[1])
        seg = h[:, idx, :].to(torch.float32)
        coeff = seg @ u_s.to(torch.float32)  # [B, T']
        delta = (
            self.scale
            * self.strength
            * coeff.unsqueeze(-1)
            * (u_t.to(torch.float32) - u_s.to(torch.float32))
        )
        h[:, idx, :] = (seg + delta).to(h.dtype)
        c = float(coeff.reshape(-1)[-1].item())
        n = float(delta.reshape(coeff.numel(), -1)[-1].norm().item())
        if np.isnan(self.first_norm):
            self.first_coeff, self.first_norm = c, n
        self.last_coeff, self.last_norm = c, n
        if torch.is_tensor(output):
            return h
        return (h,) + tuple(output[1:])


class PrefillCapture:
    """Passive forward hook: records the layer output at the last position of the
    FIRST call (the prefill), i.e. the report locus, in the *same* generate
    forward whose numerics the injection hooks see. Used to equalize the
    ``logitlens`` injected L2 with the live coeff (stage-5.2 norm fix): its
    ``u_s = norm(w_s)`` differs from the five J-lens-source conditions' shared
    ``u_s = a_s``, so the fp32-capture-vs-bf16-prefill discrepancy does not
    cancel in its scale (up to 78% per-item at 7B; design §8 note)."""

    def __init__(self) -> None:
        self.h: Tensor | None = None

    def __call__(self, module: Any, inputs: Any, output: Any) -> Any:
        if self.h is None:
            tensor = output if torch.is_tensor(output) else output[0]
            self.h = tensor[0, -1, :].detach().float()
        return output


def _chat_ids(tok: Any, content: str) -> Tensor:
    ids = tok.apply_chat_template(
        [{"role": "user", "content": content}],
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=False,
    )
    # Some transformers versions return a BatchEncoding even for return_dict=False.
    if not torch.is_tensor(ids):
        ids = ids["input_ids"] if hasattr(ids, "__getitem__") else ids.input_ids
    return cast(Tensor, ids)


def build_prompt(tok: Any, category: str, style: str) -> Tensor:
    """Task prompt (both styles chat-templated; report locus = final token).

    plain: the committed stage-5.1 wording (preserved verbatim). chat: adds an
    explicit "Answer with exactly one word." constraint to raise single-word
    compliance. Templates mirrored in 2026-07-20-stage5-design.md §8.
    """
    if style == "chat":
        content = (
            f"Think of a specific {category}. Answer with exactly one word: the "
            f"{category} you thought of. Do not write anything else."
        )
    else:
        content = (
            f"Think of a specific {category}. Reply with only that one word — "
            f"the {category} you thought of — and nothing else."
        )
    return _chat_ids(tok, content)


def concept_vector(
    m: Any, layer: int, words: list[str], templates: tuple[str, ...], max_seq_len: int
) -> Tensor:
    """Mean layer-``layer`` residual at the final prompt position over
    ``templates`` x ``words`` (fp32, [d]). The building block of the contrastive
    concept vector v_t = concept_vector(instance) - concept_vector(category)."""
    accs: list[Tensor] = []
    for w in words:
        for tmpl in templates:
            ids = _chat_ids(m.tokenizer, tmpl.format(w=w))
            if ids.shape[1] > max_seq_len:
                ids = ids[:, :max_seq_len]
            accs.append(capture_last_residual(m, ids, layer))
    return torch.stack(accs, dim=0).mean(dim=0)


def generate(m: Any, input_ids: Tensor, max_new_tokens: int) -> tuple[int, Tensor, str]:
    """Greedy generate; return (first_new_token_id, step0_logits, decoded_text)."""
    hf = m._hf_model
    ids = input_ids.to(m.input_device)
    attn = torch.ones_like(ids)
    with torch.no_grad():
        out = hf.generate(
            ids,
            attention_mask=attn,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
            pad_token_id=m.tokenizer.eos_token_id,
        )
    seq = out.sequences[0]
    new = seq[input_ids.shape[1] :]
    tok0 = int(new[0].item())
    scores0 = cast(Tensor, out.scores[0][0]).float()  # [|V|]
    text = m.tokenizer.decode(new, skip_special_tokens=True).strip()
    return tok0, scores0, text


def capture_last_residual(m: Any, input_ids: Tensor, layer: int) -> Tensor:
    """Raw residual at the last prefill position of ``layer`` (fp32, [d])."""
    from jlens.hooks import ActivationRecorder

    with ActivationRecorder(m.layers, at=[layer]) as rec, torch.no_grad():
        m.forward(input_ids.to(m.input_device))
        h = rec.activations[layer][0, -1].detach().float()
    return h


def topk_ids(logits: Tensor, k: int) -> list[int]:
    return [int(i) for i in logits.topk(k).indices.tolist()]


def rank_of(logits: Tensor, tid: int) -> int:
    return int((logits > logits[tid]).sum().item())


def main() -> None:
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
    jac = lens.jacobians[args.layer].to(device)  # [d, d] fp32
    strengths = [float(x) for x in args.strengths.split(",") if x.strip()]
    active_conditions = BASE_CONDITIONS + (
        CONTRASTIVE_CONDITIONS if args.contrastive else ()
    )

    cats = (
        [c.strip() for c in args.categories.split(",")]
        if args.categories
        else list(CATEGORIES)
    )
    cat_insts: dict[str, list[tuple[str, int]]] = {}
    for c in cats:
        if c not in CATEGORIES:
            raise SystemExit(f"unknown category {c!r}; known: {sorted(CATEGORIES)}")
        insts = single_token_instances(tok, CATEGORIES[c])
        if len(insts) >= 2:
            cat_insts[c] = insts
    print(
        f"[data] {len(cat_insts)} categories, layer={args.layer}, "
        f"strengths={strengths}, scope={args.scope}, "
        f"targets/cat={args.targets_per_category}, style={args.prompt_style}, "
        f"conditions={list(active_conditions)}"
    )

    per_item: list[dict[str, Any]] = []
    gen_tokens = 0
    n_fragment = 0  # baseline reports matching no in-category instance (compliance)
    t_gen = time.time()

    for ci, (cat, insts) in enumerate(cat_insts.items()):
        input_ids = build_prompt(tok, cat, args.prompt_style)
        inst_ids = [tid for _, tid in insts]

        # Baseline report (no intervention). Identify the named instance from the
        # decoded text; its canonical space-prefixed id is the source handle.
        # A passive PrefillCapture records the report-position residual from THIS
        # generate-prefill forward (h_live) -- the live coeff the injection hooks
        # will read -- so the logitlens injected L2 is equalized on the same
        # forward as the injection (stage-5.2 fix), not the fp32 capture below.
        cap = PrefillCapture()
        cap_handle = m.layers[args.layer].register_forward_hook(cap)
        try:
            base_tok0, base_scores, base_text = generate(
                m, input_ids, args.max_new_tokens
            )
        finally:
            cap_handle.remove()
        assert cap.h is not None, "PrefillCapture did not fire during baseline generate"
        h_live = cap.h
        gen_tokens += args.max_new_tokens
        matched = match_instance(base_text, insts)
        if matched is None:
            # Off-vocab / word-piece report: fall back to the raw first token.
            src_word, src_tid = tok.decode([base_tok0]).strip(), base_tok0
            n_fragment += 1
        else:
            src_word, src_tid = matched

        # Contrastive category baseline mean (once per category), if enabled.
        cat_concept_base = (
            concept_vector(m, args.layer, [cat], CATEGORY_TEMPLATES, args.max_seq_len)
            if args.contrastive
            else None
        )

        # Pre-report J-lens readout at the injection layer / last prefill position.
        h_last = capture_last_residual(m, input_ids, args.layer)
        read_logits = m.unembed((h_last @ jac.t()).unsqueeze(0))[0].float()  # [|V|]
        jlens_top10 = topk_ids(read_logits, TOP10)
        jlens_top5 = jlens_top10[:TOP5]
        # "predicts the report": any English token form of the reported instance in
        # the J-lens top-5. Strict-English by design (ruling 1); cross-lingual
        # synonyms are a KNOWN underestimate -- jlens_top10 is stored per item so
        # post-hoc aliasing needs no rerun.
        src_forms = set(concept_ids(tok, src_word)) | {src_tid}
        predicts = bool(src_forms & set(jlens_top5))

        # Targets (ruling 2): default = seeded random plausible-sibling instances of
        # this category (paper-comparable); leastfav = least-favoured baseline
        # instances (conservative). Both exclude the reported source token.
        pool = [t for t in inst_ids if t != src_tid]
        if args.target_mode == "leastfav":
            targets = sorted(pool, key=lambda t: float(base_scores[t]))[
                : args.targets_per_category
            ]
        else:
            rng = np.random.default_rng(args.seed + ci)
            n_take = min(args.targets_per_category, len(pool))
            targets = [int(pool[i]) for i in rng.permutation(len(pool))[:n_take]]

        u_s_j = normalize(jlens_atom(w_u, jac, src_tid))
        u_s_logit = normalize(w_u[src_tid].to(torch.float32))

        for tgt_tid in targets:
            tgt_word = tok.decode([tgt_tid]).strip()
            tgt_report = concept_ids(tok, tgt_word) or [tgt_tid]
            base_tgt_rank = rank_of(base_scores, tgt_tid)

            # Target directions per condition.
            u_t_j = normalize(jlens_atom(w_u, jac, tgt_tid))
            w_t = w_u[tgt_tid].to(torch.float32)
            comp = jspace_component(w_t, jac, w_u, GP_K)
            u_t_nonj = normalize(w_t - comp)
            gen_r = torch.Generator(device="cpu").manual_seed(args.seed + tgt_tid)
            u_t_rand = normalize(torch.randn(w_t.shape[0], generator=gen_r).to(device))
            u_t_logit = normalize(w_t)

            cond_dirs = {
                "jlens": (u_s_j, u_t_j),
                "nonjspace": (u_s_j, u_t_nonj),
                "random": (u_s_j, u_t_rand),
                "logitlens": (u_s_logit, u_t_logit),
            }

            # Stage 5.1b: activation-contrastive concept vector v_t and its
            # J-space / non-J-space components (paper's 59% / 5% rows).
            if args.contrastive:
                assert cat_concept_base is not None
                v_inst = concept_vector(
                    m, args.layer, [tgt_word], INSTANCE_TEMPLATES, args.max_seq_len
                )
                v_t = v_inst - cat_concept_base
                comp_v = jspace_component(v_t, jac, w_u, GP_K)
                cond_dirs["jspace_comp"] = (u_s_j, normalize(comp_v))
                cond_dirs["nonjspace_comp"] = (u_s_j, normalize(v_t - comp_v))

            # Ruling 3: equalize injected L2 across conditions per item/strength.
            # Raw injection L2 at the report position for residual ``h`` is
            # s*|<h,u_s>|*||u_t-u_s||; scale every condition to the jlens
            # reference so all inject the same magnitude at step 0.
            def raw_l2(u_s: Tensor, u_t: Tensor, h: Tensor) -> float:
                return abs(float(h @ u_s)) * float((u_t - u_s).norm())

            # The five J-lens-source conditions share u_s = u_s_j, so |h.u_s|
            # cancels in ref_l2/d_c -- their scale is invariant to the choice of
            # h (fp32 capture h_last used, unchanged, so they reproduce exactly).
            ref_l2 = raw_l2(u_s_j, u_t_j, h_last)
            # logitlens (u_s = w_s) does NOT share u_s_j, so its |h.u_s| factor
            # does not cancel: it must be equalized on the SAME generate-prefill
            # forward as the injection (h_live), else the fp32-capture-vs-bf16-
            # prefill mismatch leaks into its magnitude (stage-5.2 fix). Both
            # numerator and denominator use h_live, giving live norm s*ref_live,
            # exactly the five J-conditions' shared live norm.
            ref_l2_live = raw_l2(u_s_j, u_t_j, h_live)

            conds: dict[str, dict[str, Any]] = {}
            for cond in active_conditions:
                u_s, u_t = cond_dirs[cond]
                if cond == "logitlens":
                    d_c = raw_l2(u_s, u_t, h_live)
                    scale = (ref_l2_live / d_c) if d_c > STEP_EPS else 0.0
                else:
                    d_c = raw_l2(u_s, u_t, h_last)
                    scale = (ref_l2 / d_c) if d_c > STEP_EPS else 0.0
                for s in strengths:
                    hook = SwapHook(u_s, u_t, s, args.scope, scale=scale)
                    handle = m.layers[args.layer].register_forward_hook(hook)
                    try:
                        tok0, scores0, text = generate(
                            m, input_ids, args.max_new_tokens
                        )
                    finally:
                        handle.remove()
                    gen_tokens += args.max_new_tokens
                    swap_top5 = topk_ids(scores0, TOP5)
                    tgt_in_top5 = any(t in swap_top5 for t in tgt_report)
                    tgt_rank = min(rank_of(scores0, t) for t in tgt_report)
                    swap_match = match_instance(text, insts)
                    swap_word = swap_match[0] if swap_match else text
                    conds[f"{cond}@{s}"] = {
                        "condition": cond,
                        "strength": s,
                        "report_tok0": tok0,
                        "report_text": text,
                        "report_word": swap_word,
                        "target_in_top5": bool(tgt_in_top5),
                        "target_rank": int(tgt_rank),
                        "report_changed": bool(swap_word != src_word),
                        "report_is_target": bool(swap_word == tgt_word),
                        "swap_top5": swap_top5,
                        "scale": scale,
                        "source_coeff": hook.first_coeff,
                        # injected_norm = step-0 (report locus) L2, equal across
                        # conditions by construction; laststep is the last gen step.
                        "injected_norm": hook.first_norm,
                        "injected_norm_laststep": hook.last_norm,
                    }

            per_item.append(
                {
                    "category": cat,
                    "source_word": src_word,
                    "source_tok": src_tid,
                    "target_word": tgt_word,
                    "target_tok": tgt_tid,
                    "target_report_ids": tgt_report,
                    "baseline_text": base_text,
                    "baseline_tok0": base_tok0,
                    "baseline_target_rank": base_tgt_rank,
                    "jlens_top5": jlens_top5,
                    "jlens_top10": jlens_top10,
                    "jlens_predicts_report": bool(predicts),
                    "conditions": conds,
                }
            )
        dt = time.time() - t_gen
        print(
            f"[cat {ci + 1}/{len(cat_insts)}] {cat:12} src={src_word!r} "
            f"predicts={predicts} targets={[tok.decode([t]).strip() for t in targets]} "
            f"({dt:.1f}s cumulative)"
        )

    gen_elapsed = time.time() - t_gen
    tok_per_s = gen_tokens / gen_elapsed if gen_elapsed > 0 else float("nan")

    # ---- Aggregate rates per condition@strength, all items and predicts-subset ----
    cond_keys = sorted({k for it in per_item for k in it["conditions"]})

    def rate(pred: Any, subset: list[dict[str, Any]], key: str) -> float:
        vals = [pred(it["conditions"][key]) for it in subset if key in it["conditions"]]
        return float(np.mean(vals)) if vals else float("nan")

    subset_pred = [it for it in per_item if it["jlens_predicts_report"]]
    metrics: dict[str, dict[str, Any]] = {}
    for key in cond_keys:
        metrics[key] = {
            "n_all": len(per_item),
            "n_predicts": len(subset_pred),
            "target_top5_rate_all": rate(lambda c: c["target_in_top5"], per_item, key),
            "target_top5_rate_predicts": rate(
                lambda c: c["target_in_top5"], subset_pred, key
            ),
            "report_changed_rate_all": rate(
                lambda c: c["report_changed"], per_item, key
            ),
            "report_is_target_rate_all": rate(
                lambda c: c["report_is_target"], per_item, key
            ),
            "mean_target_rank_all": rate(
                lambda c: float(c["target_rank"]), per_item, key
            ),
            "mean_injected_norm": rate(lambda c: c["injected_norm"], per_item, key),
        }

    summary: dict[str, Any] = {
        "model": args.model,
        "mode": args.mode,
        "lens": args.lens,
        "layer": args.layer,
        "scope": args.scope,
        "strengths": strengths,
        "targets_per_category": args.targets_per_category,
        "target_mode": args.target_mode,
        "gp_k": GP_K,
        "n_categories": len(cat_insts),
        "n_items": len(per_item),
        "n_predicts": len(subset_pred),
        "prompt_style": args.prompt_style,
        "contrastive": bool(args.contrastive),
        "n_fragment": n_fragment,
        "fragment_report_rate": (
            n_fragment / len(cat_insts) if cat_insts else float("nan")
        ),
        "jlens_predicts_report_rate": (
            float(np.mean([it["jlens_predicts_report"] for it in per_item]))
            if per_item
            else float("nan")
        ),
        "conditions": list(active_conditions),
        "metrics": metrics,
        "gen_tokens": gen_tokens,
        "gen_seconds": gen_elapsed,
        "tokens_per_second": tok_per_s,
        "swap_note": (
            "h' = h + scale*s*<h,u_s>*(u_t-u_s) at layer L, report-position(s). "
            "u_s/u_t per condition (jlens: J-lens vectors a_v=W_U[v]@J_l; "
            "nonjspace: target = w_t minus its k-sparse nonneg J-space component; "
            "random: Gaussian target; logitlens: raw unembedding rows u_s=norm(w_s), "
            "u_t=norm(w_t) -- Nanda token-steering control; jspace_comp / "
            "nonjspace_comp: J-space / non-J-space components of the activation-"
            "contrastive concept vector v_t=mean_L(instance)-mean_L(category)). "
            "scale equalizes the step-0 injected L2 across all conditions to the "
            "jlens reference per item/strength (injected_norm equal within fp tol)."
        ),
    }

    out = args.out or default_out(
        args.model, args.lens, args.prompt_style, args.contrastive
    )
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"per_item": per_item, "summary": summary}, out)

    print(
        f"\n=== Verbal-report swap rates "
        f"(layer {args.layer}, style={args.prompt_style}, "
        f"conds={len(active_conditions)}) ==="
    )
    print(
        f"baseline fragment-report rate: {summary['fragment_report_rate']:.3f} "
        f"({n_fragment}/{len(cat_insts)} categories)"
    )
    print(
        f"J-lens top-1 predicts report: "
        f"{summary['jlens_predicts_report_rate']:.3f} "
        f"({len(subset_pred)}/{len(per_item)} items)"
    )
    print(
        f"{'condition@s':>16} {'top5_all':>9} {'top5_pred':>10} "
        f"{'changed':>8} {'=target':>8} {'inj_norm':>9}"
    )
    for key in cond_keys:
        mk = metrics[key]
        print(
            f"{key:>16} {mk['target_top5_rate_all']:>9.3f} "
            f"{mk['target_top5_rate_predicts']:>10.3f} "
            f"{mk['report_changed_rate_all']:>8.3f} "
            f"{mk['report_is_target_rate_all']:>8.3f} "
            f"{mk['mean_injected_norm']:>9.2f}"
        )
    print(f"\n[save] {out}")
    print(
        f"[time] {gen_tokens} gen-tokens in {gen_elapsed:.1f}s "
        f"= {tok_per_s:.2f} tok/s ({args.device})"
    )


if __name__ == "__main__":
    main()
