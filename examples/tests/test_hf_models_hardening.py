"""Issue #105 hardening of the in-repo loader and NLA probe.

Nothing here loads a checkpoint or touches the network: the HuggingFace entry
points are monkeypatched and the AV is the fake in ``_nla_fakes``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import _hf_models  # noqa: E402
import _nla_probe  # noqa: E402
from _nla_fakes import META, FakeAV, FakeTok, prompt_ids  # noqa: E402


def _clear_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        monkeypatch.delenv(var, raising=False)


# --- nla_verbalize guards -------------------------------------------------


@pytest.mark.parametrize(
    "activation",
    [
        torch.zeros(META["d_model"]),
        torch.full((META["d_model"],), float("nan")),
        torch.full((META["d_model"],), float("inf")),
    ],
    ids=["zero", "nan", "inf"],
)
def test_verbalize_rejects_degenerate_norm(activation: torch.Tensor) -> None:
    av = FakeAV()
    with pytest.raises(ValueError, match="norm"):
        _nla_probe.nla_verbalize(
            activation, model=av, tok=FakeTok(prompt_ids()), meta=META
        )
    assert av.seen is None


@pytest.mark.parametrize("position", ["first", "last"])
def test_verbalize_rejects_injection_at_edge(position: str) -> None:
    with pytest.raises(RuntimeError, match="sequence edge"):
        _nla_probe.nla_verbalize(
            torch.ones(META["d_model"]),
            model=FakeAV(),
            tok=FakeTok(prompt_ids(position)),
            meta=META,
        )


def test_verbalize_valid_input_still_runs() -> None:
    av = FakeAV()
    text = _nla_probe.nla_verbalize(
        torch.arange(1.0, META["d_model"] + 1),
        model=av,
        tok=FakeTok(prompt_ids()),
        meta=META,
    )
    assert text == "a fake explanation"
    assert av.seen is not None


# --- pinned revisions -----------------------------------------------------


class _Recorder:
    def __init__(self, yaml_path: Path, head_path: Path) -> None:
        self.calls: list[tuple[str, str, Any]] = []
        self.yaml_path, self.head_path = yaml_path, head_path

    def hub_download(self, repo: str, filename: str, **kw: Any) -> str:
        self.calls.append(("hf_hub_download:" + filename, repo, kw.get("revision")))
        return str(self.yaml_path if filename.endswith(".yaml") else self.head_path)

    def tokenizer(self, repo: str, **kw: Any) -> object:
        self.calls.append(("AutoTokenizer", repo, kw.get("revision")))
        return object()

    def model(self, repo: str, **kw: Any) -> Any:
        self.calls.append(("AutoModelForCausalLM", repo, kw.get("revision")))

        class _M:
            config = type("C", (), {"hidden_size": 4})()

            def eval(self) -> None:
                pass

        return _M()


@pytest.fixture
def recorder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _Recorder:
    from safetensors.torch import save_file

    yaml_path = tmp_path / "nla_meta.yaml"
    yaml_path.write_text("d_model: 4\n")
    head_path = tmp_path / "value_head.safetensors"
    save_file({"weight": torch.zeros(4, 4, dtype=torch.bfloat16)}, str(head_path))
    rec = _Recorder(yaml_path, head_path)
    monkeypatch.setattr(_nla_probe, "hf_hub_download", rec.hub_download)
    monkeypatch.setattr(_nla_probe.AutoTokenizer, "from_pretrained", rec.tokenizer)
    monkeypatch.setattr(_nla_probe.AutoModelForCausalLM, "from_pretrained", rec.model)
    _clear_offline(monkeypatch)
    return rec


def test_load_av_pins_every_fetch(recorder: _Recorder) -> None:
    _nla_probe.load_av()
    assert {c[0] for c in recorder.calls} == {
        "hf_hub_download:nla_meta.yaml",
        "AutoTokenizer",
        "AutoModelForCausalLM",
    }
    assert all(
        c[1:] == (_nla_probe.AV_ID, _nla_probe.AV_REVISION) for c in recorder.calls
    )


def test_load_ar_pins_every_fetch(recorder: _Recorder) -> None:
    _nla_probe.load_ar()
    assert {c[0] for c in recorder.calls} == {
        "hf_hub_download:nla_meta.yaml",
        "hf_hub_download:value_head.safetensors",
        "AutoTokenizer",
        "AutoModelForCausalLM",
    }
    assert all(
        c[1:] == (_nla_probe.AR_ID, _nla_probe.AR_REVISION) for c in recorder.calls
    )


def test_revisions_are_full_shas() -> None:
    for rev in (_nla_probe.AV_REVISION, _nla_probe.AR_REVISION):
        assert len(rev) == 40 and all(ch in "0123456789abcdef" for ch in rev)


# --- load_model: offline miss, max_memory, pickle opt-in ------------------


def test_offline_cache_miss_names_the_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(_hf_models, "_is_cached", lambda *_, **__: False)
    with pytest.raises(OSError, match="LLM_RESEARCH_MODEL_CACHE"):
        _hf_models.load_model("org/absent-model", mode="bf16")


@pytest.mark.parametrize("var", ["HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"])
@pytest.mark.parametrize(
    ("loader", "revision"),
    [
        (_nla_probe.load_av_meta, _nla_probe.AV_REVISION),
        (_nla_probe.load_ar_meta, _nla_probe.AR_REVISION),
    ],
    ids=["av", "ar"],
)
def test_offline_probe_load_names_the_variable(
    monkeypatch: pytest.MonkeyPatch, var: str, loader: Any, revision: str
) -> None:
    _clear_offline(monkeypatch)
    monkeypatch.setenv(var, "1")
    probed: list[str] = []

    def is_cached(*_: Any, filename: str = "config.json", **__: Any) -> bool:
        probed.append(filename)
        return False

    monkeypatch.setattr(_hf_models, "_is_cached", is_cached)
    with pytest.raises(OSError) as err:
        loader()
    assert "LLM_RESEARCH_MODEL_CACHE" in str(err.value)
    assert revision in str(err.value)
    assert probed == ["nla_meta.yaml"]


def test_online_miss_is_not_blocked(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_offline(monkeypatch)
    monkeypatch.setattr(_hf_models, "_is_cached", lambda *_, **__: False)
    _hf_models.require_cached_when_offline("org/absent-model")


@pytest.mark.parametrize(
    ("mode", "device_map"),
    [("bf16", None), ("bf16", "cpu"), ("fp32", {"": 0}), ("nf4", {"": 0})],
)
def test_max_memory_without_strategy_map_is_rejected(mode: str, device_map: Any) -> None:
    with pytest.raises(ValueError, match="strategy device_map"):
        _hf_models.load_model(
            "org/any", mode=mode, device_map=device_map, max_memory={"cpu": "8GiB"}
        )


class _FakeLoads:
    """Stands in for from_pretrained; fails the safetensors attempt."""

    def __init__(self) -> None:
        self.model_kwargs: list[dict[str, Any]] = []

    def model(self, _id: str, **kw: Any) -> str:
        self.model_kwargs.append(kw)
        if kw.get("use_safetensors"):
            raise OSError("no file named model.safetensors")
        return "pickle-model"

    def tokenizer(self, _id: str, **_: Any) -> str:
        return "tok"


@pytest.fixture
def fake_loads(monkeypatch: pytest.MonkeyPatch) -> _FakeLoads:
    fl = _FakeLoads()
    monkeypatch.setattr(_hf_models.AutoModelForCausalLM, "from_pretrained", fl.model)
    monkeypatch.setattr(_hf_models.AutoTokenizer, "from_pretrained", fl.tokenizer)
    monkeypatch.setattr(_hf_models, "_is_cached", lambda *_, **__: False)
    _clear_offline(monkeypatch)
    return fl


def test_pickle_fallback_needs_opt_in(fake_loads: _FakeLoads) -> None:
    with pytest.raises(OSError, match="allow_pickle"):
        _hf_models.load_model("org/legacy", mode="bf16")
    assert len(fake_loads.model_kwargs) == 1


def test_pickle_fallback_with_opt_in(fake_loads: _FakeLoads) -> None:
    model, tok = _hf_models.load_model("org/legacy", mode="bf16", allow_pickle=True)
    assert (model, tok) == ("pickle-model", "tok")
    assert "use_safetensors" not in fake_loads.model_kwargs[-1]


def test_float_max_memory_forwarded_with_device_map(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_offline(monkeypatch)
    seen: dict[str, Any] = {}

    def model(_id: str, **kw: Any) -> str:
        seen.update(kw)
        return "m"

    monkeypatch.setattr(_hf_models.AutoModelForCausalLM, "from_pretrained", model)
    monkeypatch.setattr(
        _hf_models.AutoTokenizer, "from_pretrained", lambda *_, **__: "t"
    )
    monkeypatch.setattr(_hf_models, "_is_cached", lambda *_, **__: False)
    budget: dict[int | str, str] = {0: "6GiB", "cpu": "20GiB"}
    _hf_models.load_model("org/any", mode="bf16", device_map="auto", max_memory=budget)
    assert seen["max_memory"] == budget and seen["device_map"] == "auto"
