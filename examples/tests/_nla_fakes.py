"""Tiny stand-ins for the NLA AV model and tokenizer.

``nla_verbalize`` touches the model only through ``get_input_embeddings`` and
``generate`` and the tokenizer only through ``apply_chat_template``,
``decode`` and ``eos_token_id``, so these fakes exercise its whole numeric
path (normalise, scale, cast, embed splice) without loading a checkpoint.
"""

from __future__ import annotations

from typing import Any

import torch

D_MODEL = 16
VOCAB = 32
INJ_ID, LEFT_ID, RIGHT_ID = 7, 5, 9

META: dict[str, Any] = {
    "d_model": D_MODEL,
    "prompt_templates": {"av": "explain {injection_char}"},
    "tokens": {
        "injection_char": "X",
        "injection_token_id": INJ_ID,
        "injection_left_neighbor_id": LEFT_ID,
        "injection_right_neighbor_id": RIGHT_ID,
    },
    "extraction": {"injection_scale": 3.5},
}


class FakeTok:
    eos_token_id = 0

    def __init__(self, ids: list[int]) -> None:
        self.ids = ids

    def apply_chat_template(self, *_: Any, **_kw: Any) -> dict[str, torch.Tensor]:
        ids = torch.tensor([self.ids])
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    def decode(self, *_: Any, **_kw: Any) -> str:
        return "prefix <explanation> a fake explanation </explanation>"


class FakeAV:
    """Records the ``inputs_embeds`` and other arguments ``generate`` receives."""

    def __init__(self) -> None:
        g = torch.Generator().manual_seed(0)
        self.embed = torch.nn.Embedding(VOCAB, D_MODEL, dtype=torch.bfloat16)
        with torch.no_grad():
            self.embed.weight.copy_(torch.randn(VOCAB, D_MODEL, generator=g))
        self.seen: torch.Tensor | None = None
        self.generate_kwargs: dict[str, Any] = {}

    def get_input_embeddings(self) -> torch.nn.Embedding:
        return self.embed

    def generate(self, *, inputs_embeds: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        self.seen = inputs_embeds.detach().clone()
        self.generate_kwargs = kwargs
        return torch.zeros(1, 1, dtype=torch.long)


def prompt_ids(position: str = "middle") -> list[int]:
    """Token ids with the injection token at ``position`` (middle/first/last)."""
    if position == "first":
        return [INJ_ID, RIGHT_ID, 1]
    if position == "last":
        return [1, LEFT_ID, INJ_ID]
    return [1, LEFT_ID, INJ_ID, RIGHT_ID, 2]
