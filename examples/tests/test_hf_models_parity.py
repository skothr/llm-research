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
import importlib.util
import inspect
import sys
from pathlib import Path
from typing import Any, Callable

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Skip only when the sibling package is absent. A present install whose
# import fails (a broken transitive dependency) must fail the gate, not skip.
if importlib.util.find_spec("llm_surgeon") is None:
    pytest.skip(
        "parity gate needs the sibling llm_surgeon install (pip install -e ../llm-surgeon)",
        allow_module_level=True,
    )
from llm_surgeon import probe as orig_probe  # noqa: E402
from llm_surgeon import surgery as orig_surgery  # noqa: E402
from llm_surgeon.probe import _nla as orig_nla  # noqa: E402

import _hf_models  # noqa: E402
import _nla_probe  # noqa: E402

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
TEXT = "The capital of France is Paris. 1 + 1 = 2; naïve café, 東京."


@pytest.fixture
def shared_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the original at the new module's cache so both read the same files."""
    monkeypatch.setattr(orig_surgery, "MODEL_CACHE_DIR", _hf_models.MODEL_CACHE_DIR)


def _snapshot(model: Any, tok: Any) -> dict[str, Any]:
    """Everything a loader can change that moves numerics or tokenisation."""
    return {
        # Clones go to the CPU so the original's weights do not stay on the
        # GPU while the new model loads.
        "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        "config": model.config.to_dict(),
        "attn": getattr(model.config, "_attn_implementation", None),
        "generation_config": model.generation_config.to_dict(),
        "ids": tok(TEXT)["input_ids"],
        "tok_attrs": {
            "pad_token_id": tok.pad_token_id,
            "padding_side": tok.padding_side,
            "model_max_length": tok.model_max_length,
            "special_tokens_map": tok.special_tokens_map,
            "chat_template": tok.chat_template,
        },
    }


def _load_both(mode: str, device_map: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    # Only the original loader may skip the test (no network and a cold
    # cache). Once it has loaded, the files are present, so a failure of the
    # new loader is a defect in the code under test and must fail the gate.
    try:
        model, tok = orig_surgery.load_model(MODEL_ID, mode=mode, revision=REVISION, device_map=device_map)
    except OSError as e:
        pytest.skip(f"cannot fetch {MODEL_ID}@{REVISION[:8]}: {e}")
    orig = _snapshot(model, tok)
    del model, tok
    gc.collect()
    model, tok = _hf_models.load_model(MODEL_ID, mode=mode, revision=REVISION, device_map=device_map)
    new = _snapshot(model, tok)
    del model, tok
    gc.collect()
    return orig, new


def _assert_parity(orig: dict[str, Any], new: dict[str, Any]) -> None:
    a, b = orig["state_dict"], new["state_dict"]
    assert a.keys() == b.keys()
    for k in a:
        assert a[k].dtype == b[k].dtype, k
        assert torch.equal(a[k], b[k]), k
    for key in ("config", "attn", "generation_config", "ids", "tok_attrs"):
        assert orig[key] == new[key], key


@pytest.mark.parametrize(
    ("mode", "dtype"),
    [("bf16", torch.bfloat16), ("fp16", torch.float16), ("fp32", torch.float32)],
)
def test_float_cpu_parity(shared_cache: None, mode: str, dtype: torch.dtype) -> None:
    orig, new = _load_both(mode, "cpu")
    assert all(v.dtype == dtype for v in new["state_dict"].values() if v.is_floating_point())
    _assert_parity(orig, new)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="nf4 needs CUDA")
@pytest.mark.parametrize("device_map", [{"": 0}, None], ids=["explicit", "default-auto"])
def test_nf4_cuda_parity(shared_cache: None, device_map: Any) -> None:
    # bitsandbytes stores packed uint8 weights plus absmax / quant_map /
    # quant_state tensors; every one present must match. Both loaders default
    # device_map to None and map None to "auto" internally, so passing None
    # is the same call as omitting it; the call sites pass explicit maps.
    orig, new = _load_both("nf4", device_map)
    assert any(v.dtype == torch.uint8 for v in new["state_dict"].values())
    _assert_parity(orig, new)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="int8 needs CUDA")
def test_int8_cuda_parity(shared_cache: None) -> None:
    orig, new = _load_both("int8", {"": 0})
    assert any(v.dtype == torch.int8 for v in new["state_dict"].values())
    _assert_parity(orig, new)


def test_nla_score_parity() -> None:
    g = torch.Generator().manual_seed(0)
    for d, scale in ((3584, None), (64, None), (64, 10.0)):
        h = torch.randn(d, generator=g, dtype=torch.bfloat16)
        p = torch.randn(d, generator=g)
        assert orig_probe.nla_score(h, p, mse_scale=scale) == _nla_probe.nla_score(h, p, mse_scale=scale)
    same = torch.randn(64, generator=g)
    assert orig_probe.nla_score(same, same) == _nla_probe.nla_score(same, same)


_CACHE_REFS = ("surgery.MODEL_CACHE_DIR", "_hf_models.MODEL_CACHE_DIR")


@pytest.mark.parametrize(
    "name",
    ["load_av_meta", "load_ar_meta", "load_av", "load_ar", "nla_verbalize", "nla_reconstruct", "nla_score"],
)
def test_nla_probe_source_identical(name: str) -> None:
    """The probe is a verbatim port: the only permitted difference is where the
    cache-dir constant comes from. Arc-01 artifacts store nla_verbalize's
    greedy-decoded text, so any other source change is a numerics change."""
    orig_src = inspect.getsource(getattr(orig_probe, name)).replace(_CACHE_REFS[0], "CACHE")
    new_src = inspect.getsource(getattr(_nla_probe, name)).replace(_CACHE_REFS[1], "CACHE")
    assert orig_src == new_src


def test_nla_constants() -> None:
    assert _nla_probe.AV_ID == orig_probe.AV_ID
    assert _nla_probe.AR_ID == orig_probe.AR_ID
    # The one module-level helper the functions use besides the constants.
    assert _nla_probe._EXPLANATION_RE.pattern == orig_nla._EXPLANATION_RE.pattern
    assert _nla_probe._EXPLANATION_RE.flags == orig_nla._EXPLANATION_RE.flags


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
