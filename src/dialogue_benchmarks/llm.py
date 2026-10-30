"""The one entry point every method/metric plugin should use for LLM access.

Never call `anthropic`/`openai` directly from a plugin -- go through complete()
(chat) or embed() (embeddings) here instead. Reasons this is shared instead of
left to each plugin:

- disk cache under DATA_ROOT/llm_cache/, shared by everyone on the team --
  re-running an eval (or two people evaluating the same method+model) costs
  zero API calls the second time
- uniform retry/backoff instead of every baseline reinventing it
- cost/latency tracked on every call so results_store can report what a run
  cost without each method having to track it itself
"""
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from dialogue_benchmarks import usage
from dialogue_benchmarks._paths import DATA_ROOT, PROJECT_ROOT

CACHE_DIR = DATA_ROOT / "llm_cache"


def _main_checkout_root() -> Path:
    """The main checkout this one belongs to. A git worktree's `.git` is a file
    ("gitdir: <main>/.git/worktrees/<name>"), so worktrees resolve to the main
    repo -- that's where the one shared `.env` lives."""
    git = PROJECT_ROOT / ".git"
    if git.is_file():
        gitdir = git.read_text().split("gitdir:", 1)[-1].strip()
        if "/.git/worktrees/" in gitdir:
            return Path(gitdir.split("/.git/worktrees/", 1)[0])
    return PROJECT_ROOT


def _load_dotenv() -> None:
    """API keys live in a gitignored `.env` at the repo root (KEY=VALUE lines),
    shared by every worktree -- project-scoped instead of exported from a
    shell profile, and picked up both by terminal `db run`s and by Claude Code
    sessions. Real environment variables win over the file. Keys used:
    OPENAI_API_KEY (default for every purpose), and optionally
    OPENAI_API_KEY_METHOD / _EMBED / _JUDGE to bill baseline answering,
    embeddings and judge metrics to separate keys (see _openai_client)."""
    for root in dict.fromkeys([PROJECT_ROOT, _main_checkout_root()]):
        path = root / ".env"
        if not path.is_file():
            continue
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip().removeprefix("export ").strip(), value.strip().strip("\"'"))


_load_dotenv()

# huggingface_hub/transformers freeze their cache-location constants (from
# these env vars) at first import anywhere in the process -- has to be set
# before that happens, so set it here since every plugin using
# complete()/embed() imports this module first. Exists because a root-run
# docker container once left root-owned files under the default
# ~/.cache/huggingface/, which a normal user can't even delete (no write
# permission on their own parent dirs, only root can) -- redirect the whole
# thing rather than depend on that ever getting chown'd back. HF_HOME alone
# isn't enough: nvidia/NV-Embed-v2's custom (trust_remote_code) modeling code
# does its own internal AutoTokenizer.from_pretrained() call that ignores
# SentenceTransformer's cache_folder kwarg and falls back to
# transformers.utils.hub's cached_file(), which reads HF_HOME but not
# cache_folder -- hence setting the env var too, not just the kwarg.
os.environ.setdefault("HF_HOME", str(DATA_ROOT / "hf_cache"))
os.environ.setdefault("HF_MODULES_CACHE", str(DATA_ROOT / "hf_cache" / "modules"))
# Moving HF_HOME also moves where huggingface_hub looks for the login token
# (HF_HOME/token), so gated models (meta-llama/*) would 403 -- point it back
# at the token `huggingface-cli login` actually wrote, without copying the
# secret anywhere new.
_HF_LOGIN_TOKEN = Path.home() / ".cache" / "huggingface" / "token"
if "HF_TOKEN" not in os.environ and "HF_TOKEN_PATH" not in os.environ and _HF_LOGIN_TOKEN.is_file():
    os.environ["HF_TOKEN_PATH"] = str(_HF_LOGIN_TOKEN)

# $ per 1M tokens (in, out). Best-effort, for run-cost bookkeeping only -- not
# billing-accurate. Add a model here when you start using it; unknown models
# just report cost=0.0 rather than failing.
_PRICING = {
    "claude-sonnet-5": (3.0, 15.0),
    "claude-opus-5": (15.0, 75.0),
    "claude-haiku-4-5-20251001": (0.8, 4.0),
    # OpenAI standard tier, checked against developers.openai.com/api/docs/pricing
    # on 2026-10-16. Cached-input rates are in _CACHED_INPUT_PRICING below.
    "gpt-4o": (2.5, 10.0),
    "gpt-4o-mini": (0.15, 0.6),
    "gpt-4.1": (2.0, 8.0),
    "gpt-4.1-mini": (0.4, 1.6),
    "gpt-5": (1.25, 10.0),
    "gpt-5-mini": (0.25, 2.0),
    # embedding models have no output tokens, so price_out is always 0
    "text-embedding-3-small": (0.02, 0.0),
    "text-embedding-3-large": (0.13, 0.0),
    # self-hosted on this machine's own GPUs (~/work/vllm_server) -- no billing
    "openai/gpt-oss-120b": (0.0, 0.0),
    "google/gemma-4-31B-it": (0.0, 0.0),
    # embedding models below: also self-hosted / local, no billing
    "nvidia/NV-Embed-v2": (0.0, 0.0),
    "Qwen/Qwen3-Embedding-4B": (0.0, 0.0),
    "all-MiniLM-L6-v2": (0.0, 0.0),
}

# $ per 1M *cached* input tokens. OpenAI applies prompt caching automatically
# (prompts >= 1024 tokens sharing a prefix, e.g. full_context re-sending the
# same transcript for every question) and bills those tokens at this rate --
# without it the ledger would overstate spend against the invoice.
_CACHED_INPUT_PRICING = {
    "gpt-4o": 1.25,
    "gpt-4o-mini": 0.075,
    "gpt-4.1": 0.5,
    "gpt-4.1-mini": 0.1,
    "gpt-5": 0.125,
    "gpt-5-mini": 0.025,
}

# Local vLLM gateway (~/work/vllm_server, `tmux attach -t vllm_gateway`) -- an
# OpenAI-compatible proxy per model that lazily starts/stops the actual vLLM
# docker container on first/idle request (cold start up to ~10 min), so these
# need a longer client timeout than the hosted APIs. No API key: the gateway
# doesn't check one, openai.OpenAI() just requires some non-empty string.
#
# openai/gpt-oss-120b is a reasoning model -- it spends completion tokens on an
# internal reasoning trace before the final answer, which the vLLM server
# surfaces as message.content=None (finish_reason="stop_reason": length) if
# max_tokens runs out before it gets there (confirmed: max_tokens=10 -> None,
# max_tokens=200 -> real content). complete()'s default max_tokens=1024 is
# comfortably past that, but a caller passing a small max_tokens (e.g. a judge
# metric expecting a one-word verdict) needs to budget for the reasoning trace
# too if it ever points at this model.
#
# Reasoning is controllable very differently per model (pass as extra
# complete(**params), forwarded straight through to the API call -- also part
# of the cache key, so different settings get separate cache entries):
# - openai/gpt-oss-120b: `reasoning_effort="low"/"medium"/"high"` (its Harmony
#   format). There is no way to fully disable it -- `reasoning_effort="none"`
#   is rejected outright (400 "Harmony does not support reasoning_effort=
#   'none'"), and even "low" still leaves a short trace in message.reasoning.
# - google/gemma-4-31B-it: `chat_template_kwargs={"enable_thinking": True/
#   False}`, default False. Unlike gpt-oss, this one genuinely turns thinking
#   off. When True, the raw "thought\n..." scratchpad is emitted inline as
#   part of message.content itself (not a separate reasoning field like
#   gpt-oss) -- budget max_tokens accordingly if enabling it.
#
# Where the gateway runs is DIALOGUE_BENCHMARKS_VLLM_HOST (scheme + host, no
# port; default http://localhost), set in .env. GPU split in the lab: both chat
# models are served by a separate GPU server's gateway (4x A6000, reached
# directly over the LAN -- no SSH tunnel needed), so this machine's
# GPU 0,1 stay free for retrieval/local models: in-process NV-Embed-v2 /
# MiniLM / memalpha / Family E checkpoints on GPU 0, and the Qwen3-Embedding
# gateway below on GPU 1. Pointing a chat model at this machine's own
# gateway (localhost:8002/8003) would put an 80%-utilization vLLM on GPU 0,1
# and either OOM those or fail to start next to them.
#
# On that server both models are TP4 at 0.8 utilization on the same 4 GPUs, so
# only one can be up at a time: calling the other within the gateway's 5-min
# idle window fails its startup ("Free memory ... less than desired GPU
# memory utilization"). E.g. predict with gpt-oss, then run the gemma judge
# metric (`--metrics qa_llm_judge`) as a separate `db run` 5+ minutes later --
# the predictions are cached, so only scoring re-runs.
_VLLM_HOST = (os.environ.get("DIALOGUE_BENCHMARKS_VLLM_HOST") or "http://localhost").rstrip("/")
_VLLM_GATEWAY = {
    "openai/gpt-oss-120b": f"{_VLLM_HOST}:8002/v1",
    "google/gemma-4-31B-it": f"{_VLLM_HOST}:8003/v1",
}
_VLLM_COLD_START_TIMEOUT = 660.0

# Embedding-serving vLLM instance (--runner pooling), same lazy-start gateway
# as above -- exists so plugins/methods/hypermem's embed_model default can
# match HyperMem's own paper default (Qwen/Qwen3-Embedding-4B, see
# vendor/hypermem/config.py's embedding_config) instead of settling for
# text-embedding-3-small just because that was the only embed() provider
# available. nvidia/NV-Embed-v2 was meant to go here too (hipporag's own
# default) but this machine's pinned vLLM (0.19.0) doesn't support its
# NVEmbedModel architecture at all ("not supported for now", checked directly
# against `vllm serve --help=ModelConfig`'s architecture list) -- so it's
# below in _LOCAL_EMBED_MODELS instead, loaded in-process via
# sentence-transformers like all-MiniLM-L6-v2, just heavier (7.85B, needs a
# real GPU in whatever process calls embed() with it -- no lazy start/stop).
_VLLM_EMBED_GATEWAY = {
    "Qwen/Qwen3-Embedding-4B": "http://localhost:8008/v1",
}

# model name -> extra kwargs for SentenceTransformer(...). Local/in-process,
# no GPU/docker/gateway management -- fine for all-MiniLM-L6-v2 (~90MB, CPU-
# friendly, plugins/methods/a_mem's and plugins/methods/memoryos's real paper
# default). nvidia/NV-Embed-v2 (7.85B, plugins/methods/hipporag's real paper
# default) is also local/in-process but NOT loaded through
# SentenceTransformer -- see _call_nvembed_direct's docstring for why -- so
# it isn't listed here; _EMBED_PROVIDERS routes it to that function
# directly instead of _call_local_embedding.
_LOCAL_EMBED_MODELS = {
    "all-MiniLM-L6-v2": {},
}
_LOCAL_ST_INSTANCES: dict = {}
# Caps how many texts _call_nvembed_direct's own encode() call handles at
# once -- see that function's docstring for why an unbounded batch OOMs
# against a co-resident vLLM container.
_NVEMBED_BATCH_SIZE = 8


@dataclass
class LLMResponse:
    text: str
    model: str
    tokens_in: int = 0
    tokens_out: int = 0
    cost: float = 0.0
    latency: float = 0.0
    cached: bool = False


@dataclass
class EmbeddingResponse:
    vectors: list
    model: str
    tokens_in: int = 0
    cost: float = 0.0
    latency: float = 0.0
    cached: bool = False


def _cache_key(model: str, payload: dict) -> str:
    blob = json.dumps({"model": model, **payload}, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


def _cache_path(model: str, key: str) -> Path:
    return CACHE_DIR / model.replace("/", "_") / f"{key}.json"


def _price_key(model: str) -> str:
    """Dated snapshots ("gpt-4o-mini-2024-07-18") bill like their alias."""
    if model in _PRICING:
        return model
    base = re.sub(r"-\d{4}-\d{2}-\d{2}$", "", model)
    return base


def _estimate_cost(model: str, tokens_in: int, tokens_out: int, cached_tokens: int = 0) -> float:
    key = _price_key(model)
    price = _PRICING.get(key)
    if not price:
        return 0.0
    price_in, price_out = price
    cached_price = _CACHED_INPUT_PRICING.get(key, price_in)
    uncached = max(tokens_in - cached_tokens, 0)
    return (uncached * price_in + cached_tokens * cached_price + tokens_out * price_out) / 1_000_000


def _call_anthropic(model, prompt, system, max_tokens, temperature, **extra):
    import anthropic

    client = anthropic.Anthropic()
    kwargs = dict(model=model, max_tokens=max_tokens, messages=[{"role": "user", "content": prompt}])
    if system:
        kwargs["system"] = system
    # Claude's API dropped temperature in favor of output_config.effort; silently
    # ignored here rather than passed through, since messages.create() now rejects
    # an unknown "temperature" kwarg outright.
    kwargs.update(extra)
    resp = client.messages.create(**kwargs)
    text = "".join(block.text for block in resp.content if hasattr(block, "text"))
    return text, resp.usage.input_tokens, resp.usage.output_tokens


def _openai_client(purpose: str):
    """OPENAI_API_KEY_<PURPOSE> (METHOD / EMBED / JUDGE) if set, else
    OPENAI_API_KEY -- separate keys let the OpenAI dashboard itself split spend
    by purpose; with one key, the usage ledger still does."""
    import openai

    key = os.environ.get(f"OPENAI_API_KEY_{purpose.upper()}") or os.environ.get("OPENAI_API_KEY")
    return openai.OpenAI(api_key=key) if key else openai.OpenAI()


# reasoning models: take max_completion_tokens instead of max_tokens and only
# the default temperature
_OPENAI_REASONING_PREFIXES = ("gpt-5", "o1", "o3", "o4")


def _call_openai(model, prompt, system, max_tokens, temperature, **extra):
    client = _openai_client(usage.purpose_now())
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    kwargs = dict(model=model, messages=messages)
    if model.startswith(_OPENAI_REASONING_PREFIXES):
        kwargs["max_completion_tokens"] = max_tokens
    else:
        kwargs["max_tokens"] = max_tokens
        if temperature is not None:
            kwargs["temperature"] = temperature
    kwargs.update(extra)
    resp = client.chat.completions.create(**kwargs)
    details = getattr(resp.usage, "prompt_tokens_details", None)
    cached = getattr(details, "cached_tokens", 0) or 0
    return resp.choices[0].message.content, resp.usage.prompt_tokens, resp.usage.completion_tokens, cached


def _call_openai_embedding(model, texts):
    client = _openai_client("embed")
    resp = client.embeddings.create(model=model, input=texts)
    return [d.embedding for d in resp.data], resp.usage.prompt_tokens


def _call_vllm_embedding(model, texts):
    import openai

    client = openai.OpenAI(base_url=_VLLM_EMBED_GATEWAY[model], api_key="EMPTY", timeout=_VLLM_COLD_START_TIMEOUT)
    resp = client.embeddings.create(model=model, input=texts)
    return [d.embedding for d in resp.data], resp.usage.prompt_tokens


def _transformers_major_version() -> int:
    import transformers

    return int(transformers.__version__.split(".")[0])


_TORCH_TIED_WEIGHTS_PATCHED = False


def _patch_torch_tied_weights_compat():
    """transformers>=5.0.0 renamed the old _tied_weights_keys class attribute
    to an all_tied_weights_keys property, computed by PreTrainedModel's own
    init/post_init. Custom trust_remote_code model classes written before
    that change (nvidia/NV-Embed-v2's modeling_nvembed.py included) never
    gained it, and crash with AttributeError inside transformers' own
    generic loading code -- confirmed the same gap hits several other
    pre-5.0 custom models this way too (MolmoForCausalLM: huggingface/
    transformers#43883; moondream3-preview: HF discussion #30). Same fix
    applies here: fall back to an empty dict (no *additional* tied weights
    beyond what the model already declares) for exactly this one missing-
    attribute case. Every model that already defines the attribute normally
    is untouched -- __getattr__ only fires when normal lookup already
    failed. Deliberately not the alternative of pulling a community-patched
    weights repo via trust_remote_code from an unofficial HF account -- that
    trades a version-compat gap for arbitrary third-party code execution,
    a worse tradeoff than patching this ourselves.
    """
    global _TORCH_TIED_WEIGHTS_PATCHED
    if _TORCH_TIED_WEIGHTS_PATCHED or _transformers_major_version() < 5:
        return
    import torch

    _orig_getattr = torch.nn.Module.__getattr__

    def _patched_getattr(self, name):
        if name == "all_tied_weights_keys":
            return {}
        return _orig_getattr(self, name)

    torch.nn.Module.__getattr__ = _patched_getattr
    _TORCH_TIED_WEIGHTS_PATCHED = True


_TRANSFORMERS_LEGACY_CACHE_PATCHED = False


def _patch_transformers_legacy_cache_compat():
    """transformers>=5.0.0 dropped the "legacy" tuple-based KV-cache format
    entirely, taking DynamicCache.from_legacy_cache and .get_usable_length
    with it. nvidia/NV-Embed-v2's forward() still calls both unconditionally
    whenever use_cache is truthy (its config's default -- embedding usage
    never actually passes a real past_key_values, so from_legacy_cache is
    always called with None here). Restores just the None-input case
    from_legacy_cache needs (build an empty DynamicCache; there's nothing to
    replay in), and get_usable_length's plain-DynamicCache behavior (no
    bound on cache size, so "usable length" is just the current sequence
    length regardless of the requested new length -- that parameter only
    ever mattered for bounded cache subclasses like StaticCache, not this
    one). Only adds either if genuinely missing, same as the tied-weights
    patch above -- a future transformers reintroducing them takes priority.
    """
    global _TRANSFORMERS_LEGACY_CACHE_PATCHED
    if _TRANSFORMERS_LEGACY_CACHE_PATCHED or _transformers_major_version() < 5:
        return
    from transformers.cache_utils import DynamicCache

    if not hasattr(DynamicCache, "from_legacy_cache"):

        @classmethod
        def _from_legacy_cache(cls, past_key_values=None):
            cache = cls()
            if past_key_values is not None:
                for layer_idx, (key_states, value_states) in enumerate(past_key_values):
                    cache.update(key_states, value_states, layer_idx)
            return cache

        DynamicCache.from_legacy_cache = _from_legacy_cache

    if not hasattr(DynamicCache, "get_usable_length"):

        def _get_usable_length(self, new_seq_length, layer_idx=0):
            return self.get_seq_length(layer_idx)

        DynamicCache.get_usable_length = _get_usable_length

    _TRANSFORMERS_LEGACY_CACHE_PATCHED = True


def _patch_nvembed_position_embeddings(nvembed_model):
    """nvidia/NV-Embed-v2's BidirectionalMistralModel.forward() calls each
    Mistral decoder layer without position_embeddings (older transformers
    computed RoPE cos/sin inside each attention layer from position_ids;
    newer transformers computes it once at the model level via a
    rotary_emb module and threads it through instead) -- MistralAttention.
    forward() then crashes trying to unpack `None`. Confirmed via a
    community fork (abhaybd/NV-Embed-v2_fixed) that this exact gap is the
    fix needed; replicated here against the official nvidia weights rather
    than trusting that fork's own weights/repo. Sets use_cache=False too
    (matches the fork -- embedding usage never benefits from KV-cache
    anyway, one-shot forward pass every call) so the legacy-cache patches
    above don't even need to engage for this model.

    Wraps only the decoder-layer *instances* belonging to this one loaded
    model, not the shared MistralDecoderLayer class -- everything else
    about each layer's real forward() (attention, MLP, norms) is untouched,
    and no other model sharing that class elsewhere in the process is
    affected. Also fills in config.head_dim when null (see the comment
    below) -- newer transformers' rotary embedding sizing needs it explicit,
    and NV-Embed-v2's text_config.json predates that requirement.

    `nvembed_model` is the raw NVEmbedModel instance (from
    transformers.AutoModel.from_pretrained(..., trust_remote_code=True) --
    not loaded via SentenceTransformer, see _call_local_embedding's
    docstring for why), so `nvembed_model.embedding_model` is directly the
    BidirectionalMistralModel this patches.
    """
    if _transformers_major_version() < 5:
        # Pre-5.0 transformers' MistralDecoderLayer computes rotary
        # embeddings internally from position_ids (the behavior
        # modeling_nvembed.py was written against) -- nothing to patch, and
        # MistralRotaryEmbedding's __init__ doesn't even accept a `config`
        # kwarg on this version (confirmed: TypeError, not just a no-op).
        return
    embedding_model = getattr(nvembed_model, "embedding_model", None)
    if embedding_model is None or getattr(embedding_model, "_position_embeddings_patched", False):
        return

    from transformers.models.mistral.modeling_mistral import MistralRotaryEmbedding

    embedding_model.config.use_cache = False
    # NV-Embed-v2's text_config.json predates transformers requiring an
    # explicit head_dim (it's left null, expecting hidden_size //
    # num_attention_heads to be inferred as before) -- newer
    # MistralRotaryEmbedding reads config.head_dim directly instead of
    # computing it, and silently uses the wrong number when it's None,
    # producing rotary embeddings sized for num_attention_heads (32) instead
    # of the real per-head dimension (128) -- a shape-mismatch RuntimeError
    # a couple layers later in apply_rotary_pos_emb. Fill in the value it
    # was always implicitly using.
    if embedding_model.config.head_dim is None:
        embedding_model.config.head_dim = (
            embedding_model.config.hidden_size // embedding_model.config.num_attention_heads
        )
    embedding_model.rotary_emb = MistralRotaryEmbedding(config=embedding_model.config)

    for layer in embedding_model.layers:
        orig_forward = layer.forward

        def _wrapped_forward(hidden_states, position_ids=None, _orig=orig_forward, _model=embedding_model, **kwargs):
            position_embeddings = _model.rotary_emb(hidden_states, position_ids)
            return _orig(hidden_states, position_ids=position_ids, position_embeddings=position_embeddings, **kwargs)

        layer.forward = _wrapped_forward

    embedding_model._position_embeddings_patched = True


def _call_nvembed_direct(model, texts):
    """nvidia/NV-Embed-v2 loaded via plain transformers.AutoModel, calling its
    own encode() directly -- not through SentenceTransformer(model_name).
    That generic wrapper's Transformer module expects the underlying HF
    model's forward() to return a standard last_hidden_state (so its own
    Pooling/Normalize modules -- 1_Pooling/, 2_Normalize/, shipped in the
    repo -- can do the pooling), but NVEmbedModel.forward() already does its
    own internal pooling (a LatentAttentionModel) and returns
    {"sentence_embeddings": ...} instead -- a KeyError, not a version-compat
    issue like the other three patches above. Its own repo's
    config_sentence_transformers.json pins sentence_transformers==2.7.0 /
    transformers==4.37.2, both far older than either package installed
    here; rather than chase a second, separate compat gap in
    sentence-transformers' own Transformer-module output-key extraction,
    just call the model's own public .encode(prompts, instruction=...) --
    documented in its model card as the direct-usage path, doing its own
    tokenization/pooling without sentence-transformers involved at all.

    encode() itself has no internal batch cap -- it runs every text in one
    forward pass through this 7B-parameter decoder + cross-attention
    pooling head, so peak GPU memory scales with however many texts the
    caller happens to pass in one call (e.g. hipporag's index() step, which
    embeds every chunk of a conversation in a single batch_encode() call).
    Live-measured this to swing from ~30GB to ~54GB between conversations
    of different chunk counts, OOM-ing against whatever else shares the GPU
    (a co-resident vLLM container). Chunking into fixed-size sub-batches
    below bounds peak memory to roughly what _NVEMBED_BATCH_SIZE texts cost,
    independent of how many texts the caller passes in total.
    """
    nvembed_model = _LOCAL_ST_INSTANCES.get(model)
    if nvembed_model is None:
        _patch_torch_tied_weights_compat()
        _patch_transformers_legacy_cache_compat()
        import torch
        from transformers import AutoModel

        nvembed_model = AutoModel.from_pretrained(model, trust_remote_code=True)
        if torch.cuda.is_available():
            nvembed_model = nvembed_model.cuda()
        _patch_nvembed_position_embeddings(nvembed_model)
        _LOCAL_ST_INSTANCES[model] = nvembed_model
    texts = list(texts)
    all_embeds = []
    for start in range(0, len(texts), _NVEMBED_BATCH_SIZE):
        batch = texts[start : start + _NVEMBED_BATCH_SIZE]
        embeds = nvembed_model.encode(batch)
        all_embeds.extend(embeds.cpu().float().numpy().tolist())
    return all_embeds, 0


def _call_local_embedding(model, texts):
    """sentence-transformers, run in-process -- no server, no cost, no token
    count (ST doesn't report one; _estimate_cost is $0 for this model anyway).
    nvidia/NV-Embed-v2 doesn't go through here -- see _call_nvembed_direct."""
    st_model = _LOCAL_ST_INSTANCES.get(model)
    if st_model is None:
        _patch_torch_tied_weights_compat()
        _patch_transformers_legacy_cache_compat()
        from sentence_transformers import SentenceTransformer

        st_model = SentenceTransformer(model, **_LOCAL_EMBED_MODELS[model])
        _LOCAL_ST_INSTANCES[model] = st_model
    vectors = st_model.encode(list(texts), convert_to_numpy=True).tolist()
    return vectors, 0


def _call_vllm(model, prompt, system, max_tokens, temperature, **extra):
    import openai

    client = openai.OpenAI(base_url=_VLLM_GATEWAY[model], api_key="EMPTY", timeout=_VLLM_COLD_START_TIMEOUT)
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    kwargs = dict(model=model, messages=messages, max_tokens=max_tokens)
    if temperature is not None:
        kwargs["temperature"] = temperature
    # vLLM-specific fields (e.g. chat_template_kwargs) aren't real parameters
    # of openai's Completions.create() -- it raises TypeError on unknown
    # kwargs unless they're routed through extra_body, which merges them
    # into the outgoing JSON body instead of the SDK's own validated params.
    if extra:
        kwargs["extra_body"] = extra
    resp = client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content, resp.usage.prompt_tokens, resp.usage.completion_tokens


# model name prefix -> provider call. Add an entry here for a new provider.
_PROVIDERS = {
    "claude": _call_anthropic,
    "gpt": _call_openai,
    "o1": _call_openai,
    "o3": _call_openai,
    "o4": _call_openai,
    **{m: _call_vllm for m in _VLLM_GATEWAY},
}
# providers that bill per call -- the only ones written to the usage ledger
_BILLED = {_call_anthropic: "anthropic", _call_openai: "openai", _call_openai_embedding: "openai"}
_EMBED_PROVIDERS = {
    "text-embedding": _call_openai_embedding,
    **{m: _call_vllm_embedding for m in _VLLM_EMBED_GATEWAY},
    **{m: _call_local_embedding for m in _LOCAL_EMBED_MODELS},
    "nvidia/NV-Embed-v2": _call_nvembed_direct,
}


def _provider_for(model: str):
    for prefix, fn in _PROVIDERS.items():
        if model.startswith(prefix):
            return fn
    raise ValueError(
        f"don't know which provider serves model '{model}'. Known prefixes: {list(_PROVIDERS)}. "
        f"Add a case to dialogue_benchmarks.llm._PROVIDERS."
    )


def _embed_provider_for(model: str):
    for prefix, fn in _EMBED_PROVIDERS.items():
        if model.startswith(prefix):
            return fn
    raise ValueError(
        f"don't know which provider serves embedding model '{model}'. Known prefixes: {list(_EMBED_PROVIDERS)}. "
        f"Add a case to dialogue_benchmarks.llm._EMBED_PROVIDERS."
    )


def complete(
    prompt: str,
    model: str,
    system: Optional[str] = None,
    max_tokens: int = 1024,
    temperature: Optional[float] = 0.0,
    use_cache: bool = True,
    max_retries: int = 3,
    **params,
) -> LLMResponse:
    key_params = {"max_tokens": max_tokens, "temperature": temperature, **params}
    key = _cache_key(model, {"prompt": prompt, "system": system, "params": key_params})
    path = _cache_path(model, key)

    if use_cache and path.exists():
        with open(path) as f:
            return LLMResponse(**json.load(f), cached=True)

    call_fn = _provider_for(model)
    last_err = None
    for attempt in range(max_retries):
        try:
            start = time.monotonic()
            text, tokens_in, tokens_out, *rest = call_fn(model, prompt, system, max_tokens, temperature, **params)
            cached_tokens = rest[0] if rest else 0
            latency = time.monotonic() - start
            break
        except Exception as e:  # noqa: BLE001 -- deliberately broad: retry any transient API/network error
            last_err = e
            time.sleep(min(2**attempt, 20))
    else:
        raise RuntimeError(f"LLM call to '{model}' failed after {max_retries} retries") from last_err

    result = {
        "text": text,
        "model": model,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "cost": _estimate_cost(model, tokens_in, tokens_out, cached_tokens),
        "latency": latency,
    }
    if call_fn in _BILLED:
        usage.record(
            provider=_BILLED[call_fn],
            model=model,
            kind="chat",
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cached_tokens=cached_tokens,
            cost=result["cost"],
            latency=latency,
        )
    if use_cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(result, f)
        tmp.rename(path)
    return LLMResponse(**result, cached=False)


def embed(
    texts: Union[str, list],
    model: str = "text-embedding-3-small",
    use_cache: bool = True,
    max_retries: int = 3,
) -> EmbeddingResponse:
    texts = [texts] if isinstance(texts, str) else texts
    key = _cache_key(model, {"texts": texts})
    path = _cache_path(model, key)

    if use_cache and path.exists():
        with open(path) as f:
            return EmbeddingResponse(**json.load(f), cached=True)

    call_fn = _embed_provider_for(model)
    last_err = None
    for attempt in range(max_retries):
        try:
            start = time.monotonic()
            vectors, tokens_in = call_fn(model, texts)
            latency = time.monotonic() - start
            break
        except Exception as e:  # noqa: BLE001 -- deliberately broad: retry any transient API/network error
            last_err = e
            time.sleep(min(2**attempt, 20))
    else:
        raise RuntimeError(f"embedding call to '{model}' failed after {max_retries} retries") from last_err

    result = {
        "vectors": vectors,
        "model": model,
        "tokens_in": tokens_in,
        "cost": _estimate_cost(model, tokens_in, 0),
        "latency": latency,
    }
    if call_fn in _BILLED:
        usage.record(
            provider=_BILLED[call_fn],
            model=model,
            kind="embed",
            tokens_in=tokens_in,
            tokens_out=0,
            cached_tokens=0,
            cost=result["cost"],
            latency=latency,
        )
    if use_cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(result, f)
        tmp.rename(path)
    return EmbeddingResponse(**result, cached=False)
