"""Migration gate for issue #94: in-repo loaders must match llm_surgeon exactly.

``examples/_hf_models.py`` and ``examples/_nla_probe.py`` replace the sibling
``llm_surgeon`` install. Committed artifacts were produced through the
original, so the replacement must load bit-identical weights and tokenizers
and compute identical scores. Both loaders read the same cache for the
comparison, so the test measures the loading code, not the cache location.

Needs the sibling ``llm_surgeon`` install (skips without it) and network or a
warm cache for ``Qwen/Qwen2.5-0.5B-Instruct`` (skips without them). The nf4
case needs CUDA. Run with:

    python -m pytest examples/tests/test_hf_models_parity.py
"""

from __future__ import annotations

import gc
import inspect
import sys
from pathlib import Path
from typing import Any, Callable

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from llm_surgeon import probe as orig_probe
    from llm_surgeon import surgery as orig_surgery
except ImportError:
    pytest.skip(
        "parity gate needs the sibling llm_surgeon install (pip install -e ../llm-surgeon)",
        allow_module_level=True,
    )

import _hf_models  # noqa: E402
import _nla_probe  # noqa: E402

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
TEXT = "The capital of France is Paris. 1 + 1 = 2; naïve café, 東京."


@pytest.fixture
def shared_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the original at the new module's cache so both read the same files."""
    monkeypatch.setattr(orig_surgery, "MODEL_CACHE_DIR", _hf_models.MODEL_CACHE_DIR)


def _load_both(mode: str, device_map: Any) -> tuple[dict[str, torch.Tensor], list[int], dict[str, torch.Tensor], list[int]]:
    out: list[Any] = []
    for loader in (orig_surgery.load_model, _hf_models.load_model):
        try:
            model, tok = loader(MODEL_ID, mode=mode, revision=REVISION, device_map=device_map)
        except OSError as e:  # no network and a cold cache
            pytest.skip(f"cannot fetch {MODEL_ID}@{REVISION[:8]}: {e}")
        out += [{k: v.detach().clone() for k, v in model.state_dict().items()}, tok(TEXT)["input_ids"]]
        del model, tok
        gc.collect()
    return out[0], out[1], out[2], out[3]


def _assert_state_dicts_equal(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> None:
    assert a.keys() == b.keys()
    for k in a:
        assert a[k].dtype == b[k].dtype, k
        assert torch.equal(a[k], b[k]), k


def test_bf16_cpu_parity(shared_cache: None) -> None:
    sd_orig, ids_orig, sd_new, ids_new = _load_both("bf16", "cpu")
    assert all(v.dtype == torch.bfloat16 for v in sd_new.values() if v.is_floating_point())
    _assert_state_dicts_equal(sd_orig, sd_new)
    assert ids_orig == ids_new


@pytest.mark.skipif(not torch.cuda.is_available(), reason="nf4 needs CUDA")
def test_nf4_cuda_parity(shared_cache: None) -> None:
    # bitsandbytes stores packed uint8 weights plus absmax / quant_map /
    # quant_state tensors; every one present must match.
    sd_orig, ids_orig, sd_new, ids_new = _load_both("nf4", {"": 0})
    assert any(v.dtype == torch.uint8 for v in sd_new.values())
    _assert_state_dicts_equal(sd_orig, sd_new)
    assert ids_orig == ids_new


def test_nla_score_parity() -> None:
    g = torch.Generator().manual_seed(0)
    for d, scale in ((3584, None), (64, None), (64, 10.0)):
        h = torch.randn(d, generator=g, dtype=torch.bfloat16)
        p = torch.randn(d, generator=g)
        assert orig_probe.nla_score(h, p, mse_scale=scale) == _nla_probe.nla_score(h, p, mse_scale=scale)
    same = torch.randn(64, generator=g)
    assert orig_probe.nla_score(same, same) == _nla_probe.nla_score(same, same)


def test_nla_constants() -> None:
    assert _nla_probe.AV_ID == orig_probe.AV_ID
    assert _nla_probe.AR_ID == orig_probe.AR_ID


def _params(fn: Callable[..., Any]) -> list[tuple[str, Any, Any]]:
    # Names, kinds and defaults only: surgery.py has no `from __future__ import
    # annotations`, so its annotations are objects where ours are strings.
    return [(p.name, p.kind, p.default) for p in inspect.signature(fn).parameters.values()]


@pytest.mark.parametrize(
    "name",
    ["load_av", "load_ar", "nla_verbalize", "nla_reconstruct", "nla_score"],
)
def test_nla_signatures(name: str) -> None:
    assert _params(getattr(_nla_probe, name)) == _params(getattr(orig_probe, name))


def test_load_model_signature() -> None:
    assert _params(_hf_models.load_model) == _params(orig_surgery.load_model)
