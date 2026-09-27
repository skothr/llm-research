"""In-repo Hugging Face model loader: ``MODEL_CACHE_DIR`` and ``load_model``.

Replaces ``llm_surgeon.surgery.{MODEL_CACHE_DIR, load_model}`` from the
sibling llm-surgeon checkout (issue #94). The sibling was an editable install
that is not on PyPI, so a clean clone could not run the capture scripts; the
dependency policy is in ``research/ARC_PROCESS.md`` § The non-negotiables.
Only the code paths this repo calls are ported: the Ollama/GGUF branch, the
mode aliases and the ``fp32-cpu`` mode are dropped.

Behaviour that committed artifacts depend on, and so must not change:

* ``nf4``: ``BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
  bnb_4bit_compute_dtype=torch.float16)``. Compute dtype is fp16 (not bf16)
  and double quantisation is off (the bitsandbytes default, not passed).
  ``device_map="auto"`` unless the caller overrides it.
* ``int8``: ``BitsAndBytesConfig(load_in_8bit=True)``, ``device_map="auto"``
  unless overridden.
* ``bf16`` / ``fp16`` / ``fp32``: ``torch_dtype`` only, plus ``device_map``
  and ``max_memory`` when the caller passes them. transformers applies
  ``max_memory`` only under a strategy ``device_map`` ("auto", "balanced",
  "balanced_low_0", "sequential"), so any other combination is rejected.
* ``revision`` is forwarded to both model and tokenizer; the emb_* captures
  pin it to a commit SHA.
* ``use_safetensors=True``. A legacy pickle ``.bin`` retry happens only when
  the caller passes ``allow_pickle=True`` and the error names safetensors;
  every model this repo loads ships safetensors.

``MODEL_CACHE_DIR`` is ``$LLM_RESEARCH_MODEL_CACHE`` when set, else ``None``
(the Hugging Face default cache). With ``HF_HUB_OFFLINE`` set, a checkpoint
missing from that cache raises ``OSError`` naming the directory probed and the
variable that sets it (``require_cached_when_offline``).
"""

from __future__ import annotations

import logging
import os
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logger = logging.getLogger(__name__)

MODEL_CACHE_DIR: str | None = os.environ.get("LLM_RESEARCH_MODEL_CACHE") or None

VALID_MODES = {"nf4", "int8", "bf16", "fp16", "fp32"}

_TRUE_VALUES = {"1", "ON", "YES", "TRUE"}  # huggingface_hub's parse of the flag
_DEVICE_MAP_STRATEGIES = {"auto", "balanced", "balanced_low_0", "sequential"}


def _is_cached(
    model_id: str,
    cache_dir: str | None = None,
    revision: str | None = None,
    filename: str = "config.json",
) -> bool:
    """True if the cache holds ``filename`` in model_id's snapshot at revision.

    Probing the same revision the load will request keeps ``local_files_only``
    honest: a cached ``main`` with an uncached pinned SHA must go to the
    network, and a cached pinned SHA with no ``main`` ref must not.
    """
    from huggingface_hub import try_to_load_from_cache

    path = try_to_load_from_cache(
        model_id,
        filename=filename,
        cache_dir=cache_dir or MODEL_CACHE_DIR,
        revision=revision,
    )
    # A str is a cached file. None is unknown. The _CACHED_NO_EXIST sentinel
    # means the file is known to be absent, which is not "cached".
    return isinstance(path, str)


def _offline() -> bool:
    # Read at call time: huggingface_hub captures HF_HUB_OFFLINE at import, and
    # the offline scripts set it before importing, so both agree in practice.
    # transformers also honours TRANSFORMERS_OFFLINE.
    return any(
        os.environ.get(var, "").upper() in _TRUE_VALUES
        for var in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    )


def require_cached_when_offline(
    model_id: str, revision: str | None = None, filename: str = "config.json"
) -> None:
    """Raise a descriptive ``OSError`` for an offline load of an uncached model.

    Without this, the load fails inside HuggingFace with a generic offline
    cache-miss error that names neither the cache directory nor
    ``LLM_RESEARCH_MODEL_CACHE``. Local directories and online loads pass.
    It probes one file (``filename``), so it catches a missing snapshot, not
    an incomplete one: a cache holding that file but missing weight shards
    still reaches HuggingFace's own error.
    """
    if (
        not _offline()
        or os.path.isdir(model_id)
        or _is_cached(model_id, revision=revision, filename=filename)
    ):
        return
    where = MODEL_CACHE_DIR or "the HuggingFace default cache"
    at = f"@{revision}" if revision else ""
    raise OSError(
        f"{model_id}{at} ({filename}) is not in {where} and offline mode is set "
        "(HF_HUB_OFFLINE or TRANSFORMERS_OFFLINE). Export "
        "LLM_RESEARCH_MODEL_CACHE to the directory that holds the checkpoint, "
        "or run once online to download it."
    )


def load_model(
    model_id: str,
    mode: str = "nf4",
    *,
    revision: str | None = None,
    max_memory: dict[int | str, str] | None = None,
    device_map: str | dict[str, int | str] | None = None,
    allow_pickle: bool = False,
) -> tuple:
    """Load a causal LM and its tokenizer; return ``(model, tokenizer)``.

    Args:
        model_id: Hugging Face Hub ID or local directory.
        mode: One of ``nf4``, ``int8``, ``bf16``, ``fp16``, ``fp32``.
        revision: Hub commit SHA / branch / tag to pin the snapshot.
        max_memory: accelerate budget (e.g. ``{0: "5.5GiB", "cpu": "20GiB"}``).
            Needs a strategy device map ("auto", the quantized modes' default,
            or "balanced" / "balanced_low_0" / "sequential").
        device_map: Overrides the device map (``{"": 0}`` forces the whole
            model onto GPU 0; ``"cpu"`` for the CPU bf16 paths).
        allow_pickle: Permit the legacy pickle ``.bin`` fallback when no
            safetensors file is available. Unpickling can execute code, so it
            is off unless the caller opts in.
    """
    if mode not in VALID_MODES:
        raise ValueError(f"Unknown mode: '{mode}'. Must be one of {sorted(VALID_MODES)}.")
    effective_map = device_map
    if effective_map is None and mode in ("nf4", "int8"):
        effective_map = "auto"
    is_strategy = isinstance(effective_map, str) and effective_map in _DEVICE_MAP_STRATEGIES
    if max_memory is not None and not is_strategy:
        raise ValueError(
            f"max_memory needs a strategy device_map {sorted(_DEVICE_MAP_STRATEGIES)}; "
            f"with {effective_map!r} transformers would ignore it."
        )
    require_cached_when_offline(model_id, revision)

    is_local = os.path.isdir(model_id)
    cached = (not is_local) and _is_cached(model_id, revision=revision)

    common_kwargs: dict[str, Any] = {"use_safetensors": True, "revision": revision}
    if not is_local:
        common_kwargs["cache_dir"] = MODEL_CACHE_DIR
        common_kwargs["local_files_only"] = cached

    mode_kwargs: dict[str, Any]
    if mode == "nf4":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        mode_kwargs = {"quantization_config": bnb_config, "device_map": "auto"}
        if max_memory is not None:
            mode_kwargs["max_memory"] = max_memory
    elif mode == "int8":
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        mode_kwargs = {"quantization_config": bnb_config, "device_map": "auto"}
        if max_memory is not None:
            mode_kwargs["max_memory"] = max_memory
    elif mode == "bf16":
        mode_kwargs = {"torch_dtype": torch.bfloat16}
    elif mode == "fp16":
        mode_kwargs = {"torch_dtype": torch.float16}
    else:  # fp32
        mode_kwargs = {"torch_dtype": torch.float32}
    if device_map is not None:
        mode_kwargs["device_map"] = device_map
        if max_memory is not None:
            mode_kwargs["max_memory"] = max_memory

    # Safetensors first: pickle-format .bin can execute code on load. Fall
    # back to .bin only when the caller opted in and the failure is about
    # safetensors (older Hub models, or a stale
    # `.no_exist/<sha>/model.safetensors` cache marker).
    # Before that, a load barred from the network by the cache probe (which
    # checks config.json only) is retried online once, still safetensors-only,
    # so a partial cache downloads its missing shards instead of failing.
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, **common_kwargs, **mode_kwargs)
    except OSError as first_error:
        error: OSError = first_error
        model = None
        if common_kwargs.get("local_files_only"):
            online_kwargs = {k: v for k, v in common_kwargs.items() if k != "local_files_only"}
            try:
                model = AutoModelForCausalLM.from_pretrained(model_id, **online_kwargs, **mode_kwargs)
            except OSError as second_error:
                error = second_error
        if model is None:
            # Decide the fallback on either error: offline, the online retry
            # fails with a connection error that hides the safetensors cause.
            if "safetensors" not in (str(first_error) + str(error)).lower():
                raise error
            if not allow_pickle:
                hint = (
                    " Offline mode is set, so the cache may be incomplete; a run "
                    "online fills it." if _offline() else ""
                )
                raise OSError(
                    f"Model '{model_id}' has no safetensors file accessible. Pass "
                    "allow_pickle=True to load the legacy .bin weights (unpickling "
                    f"can execute code).{hint}"
                ) from error
            logger.warning(
                "Model '%s' has no safetensors file accessible; falling back to .bin",
                model_id,
            )
            retry_kwargs = {
                k: v for k, v in common_kwargs.items()
                if k not in ("use_safetensors", "local_files_only")
            }
            model = AutoModelForCausalLM.from_pretrained(model_id, **retry_kwargs, **mode_kwargs)

    # AutoTokenizer does not accept use_safetensors. The cache probe checks
    # config.json only, so a partial cache can hold the model files and not the
    # tokenizer's; retry without local_files_only in that case.
    tok_kwargs = {k: v for k, v in common_kwargs.items() if k != "use_safetensors"}
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, **tok_kwargs)
    except OSError:
        if not tok_kwargs.get("local_files_only"):
            raise
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, **{k: v for k, v in tok_kwargs.items() if k != "local_files_only"}
        )

    return model, tokenizer
