"""The model roster: one YAML per model under ``configs/models/`` and its loading.

``load_model_and_tokenizer`` places single-GPU models on cuda:0 and shards
multi-GPU ones with ``device_map="auto"`` (``n_gpus`` in the model config), and
fails loudly if any module is left off the GPU. ``load_tokenizer`` wraps
``AutoTokenizer`` in a whitespace round-trip guard (see its docstring: the
DeepSeek-R1-Distill checkpoints load with a whitespace-dropping pre-tokenizer
under transformers 5.6.x and must be rebuilt from their own tokenizer.json).

``thinking_mode`` marks the native thinking models (Qwen3, the DeepSeek
distills, Phi-4-reasoning, gpt-oss); how the block is suppressed on the direct
pass and closed on the reasoning pass lives in :mod:`geo_alg_ti_llms.scoring`
and the config keys ``thinking_suppress_prefix`` / ``thinking_stop_string``.
torch and transformers are imported inside the loaders only, so the package
imports without them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


DEFAULT_MODELS_DIR: Path = Path(__file__).resolve().parents[2] / "configs" / "models"
STUB_NAME = "stub/char"
"""The model name that loads the stub of :mod:`geo_alg_ti_llms.stub` (no torch)."""
"""The repository's ``configs/models`` directory (the 22-model roster)."""


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Loaded from ``configs/models/<short_name>.yaml``."""

    name: str               # full HuggingFace id, e.g. "Qwen/Qwen3-32B"
    revision: str | None    # the Hub commit the published runs loaded (None: the Hub default)
    short_name: str
    dtype: str              # "bfloat16" | "float16" | "float32"
    chat_template: bool
    thinking_mode: bool | None
    expected_vram_gb: float | None
    # Text appended to the chat-template generation prompt (after
    # add_generation_prompt=True) before scoring. Needed for models whose
    # template omits the post-header newline that the model saw in
    # training: GLM-4 renders assistant turns as ``<|assistant|>\n{content}``
    # but its generation prompt ends at ``<|assistant|>``, so the model's
    # first emitted token is ``\n`` and the answer (True/False) lands one
    # position later. Setting this to ``"\n"`` moves the scored position
    # onto the answer token. ``None``/empty for Llama/Qwen (their templates
    # already bake the trailing newline into add_generation_prompt).
    chat_generation_suffix: str | None
    # Text appended to the chat-template prompt on the DIRECT/no-reasoning pass
    # to SUPPRESS native reasoning on models whose template has no
    # ``enable_thinking`` toggle (DeepSeek-R1-distill, Phi-4-reasoning — they
    # always open a <think> block). Set to the model's empty-thinking string
    # (e.g. ``"<think>\n\n</think>\n\n"``) so the scored next-token lands on the
    # answer instead of an opening <think>. ``None`` for Qwen3 (which uses
    # ``enable_thinking=False`` instead) and all non-reasoning models.
    thinking_suppress_prefix: str | None
    # String that closes the native reasoning block, used as the generation stop
    # on the CoT pass when ``thinking_mode`` is true. Default ``"</think>"``
    # (Qwen3, DeepSeek-R1-distill, Phi-4-reasoning); gpt-oss ends its
    # ``analysis`` channel with ``<|end|>`` instead.
    thinking_stop_string: str
    # Number of GPUs the bf16 weights need (device_map="auto" shards across them).
    n_gpus: int
    # Placement passed to from_pretrained: ``"auto"`` (default; accelerate shards over
    # the visible GPUs) or ``"cuda"`` (= ``{"": 0}``: every module on GPU 0, no CPU
    # offload). gpt-oss-20b needs the latter -- under "auto" transformers 5.6 left part
    # of its dequantized MoE experts on the CPU, where ``aten::_grouped_mm`` does not
    # exist. Either way, load_model_and_tokenizer refuses to return a model with any
    # parameter off the GPU.
    device_map: str
    # MoE experts kernel passed to from_pretrained (``None`` = transformers' default,
    # grouped_mm). gpt-oss-20b needs ``"eager"`` on Blackwell: torch 2.8's
    # ``torch._grouped_mm`` runs only on compute capability 9.0 and transformers 5.6
    # dispatches to it anyway. Same implementation for every run of that model, so it
    # is uniform within the model.
    experts_implementation: str | None


def model_config_path(short_name: str, models_dir: str | Path = DEFAULT_MODELS_DIR) -> Path:
    """``configs/models/<short_name>.yaml``."""
    return Path(models_dir) / f"{short_name}.yaml"


def load_model_config(path: str | Path) -> ModelConfig:
    """Load and validate a model config YAML."""
    with Path(path).open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return ModelConfig(
        name=str(raw["name"]),
        revision=str(raw["revision"]) if raw.get("revision") else None,
        short_name=str(raw["short_name"]),
        dtype=str(raw.get("dtype", "bfloat16")),
        chat_template=bool(raw.get("chat_template", True)),
        thinking_mode=raw.get("thinking_mode"),
        expected_vram_gb=raw.get("expected_vram_gb"),
        chat_generation_suffix=raw.get("chat_generation_suffix"),
        thinking_suppress_prefix=raw.get("thinking_suppress_prefix"),
        thinking_stop_string=str(raw.get("thinking_stop_string") or "</think>"),
        n_gpus=int(raw.get("n_gpus") or 1),
        device_map=str(raw.get("device_map") or "auto"),
        experts_implementation=raw.get("experts_implementation"),
    )


ROUNDTRIP_PROBE = "a b\nc"


def tokenizer_roundtrips(tokenizer, probe: str = ROUNDTRIP_PROBE) -> bool:
    """True iff ``decode(encode(probe))`` reproduces the probe's spaces and newlines."""
    return tokenizer.decode(tokenizer.encode(probe, add_special_tokens=False)) == probe


def load_tokenizer(name: str, revision: str | None = None):
    """``AutoTokenizer.from_pretrained`` with a whitespace round-trip guard.

    Returns ``(tokenizer, rebuilt)``.

    Why the guard: under transformers 5.6.2 the three DeepSeek-R1-Distill checkpoints
    (``tokenizer_class: LlamaTokenizerFast``) load as a SentencePiece-style
    ``LlamaTokenizer`` whose ``Metaspace`` pre-tokenizer drops every space and newline
    (``"a b\nc"`` -> ``"abc"``) and folds ``" True"`` onto ``"True"``, although the
    checkpoint's ``tokenizer.json`` declares a byte-level pipeline.

    If the probe does not round-trip, the tokenizer is rebuilt from the checkpoint's own
    ``tokenizer.json`` (the pipeline the model was trained with), keeping the chat
    template and the bos/eos/pad tokens, re-checked, and the rebuild is printed so it
    appears in every job log. Tokenizers that round-trip (all byte-level BPE families,
    and SentencePiece ones such as Mistral that map spaces to a marker) are returned as
    loaded. The chat template of the rebuilt tokenizer must render byte-identically.
    """
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name, revision=revision)
    if tokenizer_roundtrips(tok):
        return tok, False
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer
    from transformers import PreTrainedTokenizerFast
    path = hf_hub_download(name, "tokenizer.json", revision=revision)
    fixed = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer.from_file(path), chat_template=tok.chat_template,
        bos_token=tok.bos_token, eos_token=tok.eos_token, pad_token=tok.pad_token or tok.eos_token,
    )
    if not tokenizer_roundtrips(fixed):
        raise RuntimeError(f"{name}: tokenizer drops whitespace even when rebuilt from tokenizer.json")
    probe_msgs = [{"role": "user", "content": "probe"}]
    kw = dict(tokenize=False, add_generation_prompt=True)
    if tok.apply_chat_template(probe_msgs, **kw) != fixed.apply_chat_template(probe_msgs, **kw):
        raise RuntimeError(f"{name}: rebuilt tokenizer renders a different chat template")
    print(f"[tokenizer] {name}: AutoTokenizer gave a {type(tok).__name__} with a "
          f"{type(tok.backend_tokenizer.pre_tokenizer).__name__} pre-tokenizer that drops "
          f"whitespace; REBUILT from tokenizer.json "
          f"({type(fixed.backend_tokenizer.pre_tokenizer).__name__}); eos={fixed.eos_token_id} "
          f"bos={fixed.bos_token_id}", flush=True)
    return fixed, True


def load_model_and_tokenizer(cfg: ModelConfig):
    """Load model + tokenizer per the config.

    Returns ``(model, tokenizer)``. Model is set to ``eval()`` and placed
    on the GPU(s) per ``cfg.device_map`` / ``cfg.n_gpus``.
    """
    if cfg.name == STUB_NAME:
        from geo_alg_ti_llms import stub
        return stub.StubModel(), stub.StubTokenizer()
    # Lazy import: keeps `--help` paths fast and isolates heavy deps.
    import torch
    from transformers import AutoModelForCausalLM

    print(f"loading tokenizer: {cfg.name}")
    tokenizer, _rebuilt = load_tokenizer(cfg.name, cfg.revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if cfg.dtype not in dtype_map:
        raise ValueError(f"unsupported dtype: {cfg.dtype!r}")

    kwargs: dict = {"torch_dtype": dtype_map[cfg.dtype], "revision": cfg.revision}

    # Single-GPU models go straight to cuda:0 -- the same placement "auto" produces
    # when the weights fit, without its silent CPU offload when it thinks they do not
    # (a second load in one process can land on the CPU).
    # Multi-GPU models (n_gpus > 1) keep "auto" sharding.
    single = cfg.device_map == "cuda" or (cfg.device_map == "auto" and cfg.n_gpus == 1)
    kwargs["device_map"] = {"": 0} if single else cfg.device_map
    if cfg.experts_implementation:
        kwargs["experts_implementation"] = cfg.experts_implementation

    print(f"loading model:     {cfg.name}@{cfg.revision}  dtype={cfg.dtype}")
    model = AutoModelForCausalLM.from_pretrained(cfg.name, **kwargs)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    device = next(model.parameters()).device
    # Uniformity guard: a module silently offloaded to the CPU (or left on the meta
    # device) would run a different kernel path from the other runs -- or crash
    # later, deep inside a forward pass. Fail here, naming the offenders.
    off = sorted({n.rsplit(".", 2)[0] for n, prm in model.named_parameters()
                  if prm.device.type != "cuda"})
    if off:
        raise RuntimeError(f"{len(off)} module(s) not on a GPU after loading {cfg.name} "
                           f"(device_map={cfg.device_map!r}): {off[:8]}{' ...' if len(off) > 8 else ''}")
    print(f"loaded:            {n_params:.2f}B params on {device}")
    return model, tokenizer
