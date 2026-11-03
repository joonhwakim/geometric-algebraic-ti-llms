"""Tests for ``load_model_config`` and the committed roster (no model loading)."""

from __future__ import annotations

from pathlib import Path

import pytest

from geo_alg_ti_llms.models import DEFAULT_MODELS_DIR, load_model_config, model_config_path

ROSTER = [
    "deepseek-r1-distill-llama-8b", "deepseek-r1-distill-qwen-14b", "deepseek-r1-distill-qwen-32b",
    "gemma-2-9b-it", "glm-4-32b-0414", "glm-4-9b-0414", "gpt-oss-20b",
    "llama-3.1-70b-instruct", "llama-3.1-8b-instruct", "llama-3.2-1b", "llama-3.2-3b",
    "mistral-7b-instruct-v0.3", "olmo-2-32b-instruct", "olmo-2-7b-instruct", "phi-4-reasoning",
    "qwen2.5-32b-instruct", "qwen2.5-7b-instruct",
    "qwen3-1.7b", "qwen3-14b", "qwen3-32b", "qwen3-4b", "qwen3-8b",
]
NATIVE_THINKING = {m for m in ROSTER if m.startswith("qwen3") or "deepseek" in m
                   or "phi-4" in m or "gpt-oss" in m}


def _write_cfg(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "m.yaml"
    p.write_text(body, encoding="utf-8")
    return p


def test_roster_is_the_22_models_of_the_paper():
    assert sorted(p.stem for p in DEFAULT_MODELS_DIR.glob("*.yaml")) == ROSTER


@pytest.mark.parametrize("short_name", ROSTER)
def test_every_roster_config_loads_bf16(short_name):
    cfg = load_model_config(model_config_path(short_name))
    assert cfg.short_name == short_name
    assert cfg.dtype == "bfloat16"
    assert cfg.chat_template is True
    assert bool(cfg.thinking_mode) == (short_name in NATIVE_THINKING)
    assert cfg.revision and len(cfg.revision) == 40 and int(cfg.revision, 16) >= 0   # a Hub commit


def test_chat_generation_suffix_read_from_yaml(tmp_path: Path) -> None:
    cfg = load_model_config(_write_cfg(tmp_path, (
        "name: x/Y\nshort_name: y\ndtype: bfloat16\nquantization: null\n"
        "chat_template: true\nthinking_mode: null\nexpected_vram_gb: 1\n"
        'chat_generation_suffix: "\\n"\n'
    )))
    assert cfg.chat_generation_suffix == "\n"


def test_defaults_when_fields_are_omitted(tmp_path: Path) -> None:
    cfg = load_model_config(_write_cfg(tmp_path, "name: x/Y\nshort_name: y\n"))
    assert cfg.dtype == "bfloat16"
    assert cfg.chat_template is True
    assert cfg.thinking_mode is None
    assert cfg.revision is None
    assert cfg.chat_generation_suffix is None
    assert cfg.thinking_suppress_prefix is None
    assert cfg.thinking_stop_string == "</think>"
    assert cfg.n_gpus == 1
    assert cfg.device_map == "auto"
    assert cfg.experts_implementation is None


@pytest.mark.parametrize("fname", ["glm-4-9b-0414", "glm-4-32b-0414"])
def test_glm_configs_set_newline_suffix(fname: str) -> None:
    """The committed GLM configs must request the trailing newline so True/False
    is scored at the right position."""
    assert load_model_config(model_config_path(fname)).chat_generation_suffix == "\n"


@pytest.mark.parametrize("fname", ["qwen2.5-7b-instruct", "llama-3.2-3b", "qwen3-32b"])
def test_llama_qwen_configs_have_no_suffix(fname: str) -> None:
    """Llama/Qwen templates already bake the trailing newline in, so they
    must NOT add one (doing so would shift the scored position)."""
    assert load_model_config(model_config_path(fname)).chat_generation_suffix is None


def test_thinking_suppress_prefix_read_and_default(tmp_path: Path) -> None:
    with_pfx = load_model_config(_write_cfg(tmp_path, (
        "name: x/Y\nshort_name: y\ndtype: bfloat16\n"
        "chat_template: true\nthinking_mode: true\nexpected_vram_gb: 20\n"
        'thinking_suppress_prefix: "</think>\\n\\n"\n'
    )))
    assert with_pfx.thinking_suppress_prefix == "</think>\n\n"


def test_nonqwen_reasoning_configs_set_suppression() -> None:
    """The committed no-toggle reasoning configs must carry a suppression
    prefix (their templates have no enable_thinking), while Qwen3 must NOT
    (it uses enable_thinking=False instead)."""
    ds = load_model_config(model_config_path("deepseek-r1-distill-llama-8b"))
    phi = load_model_config(model_config_path("phi-4-reasoning"))
    oss = load_model_config(model_config_path("gpt-oss-20b"))
    q3 = load_model_config(model_config_path("qwen3-32b"))
    assert ds.thinking_mode and ds.thinking_suppress_prefix == "</think>\n\n"
    assert phi.thinking_mode and phi.thinking_suppress_prefix == "<think>\n\n</think>\n\n"
    assert oss.thinking_mode and oss.thinking_suppress_prefix == "<|channel|>final<|message|>"
    assert oss.thinking_stop_string == "<|end|>"
    assert oss.device_map == "cuda" and oss.experts_implementation == "eager"
    assert q3.thinking_mode and q3.thinking_suppress_prefix is None


def test_llama_70b_needs_two_gpus() -> None:
    assert load_model_config(model_config_path("llama-3.1-70b-instruct")).n_gpus == 2
