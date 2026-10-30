"""embed()'s cache/retry/cost-tracking mirrors complete()'s -- these tests mostly
cover embed() since it's new; complete() has no broader test coverage and
backfilling it is out of scope here, beyond the regression tests below for the
temperature/anthropic-SDK bug and the local vLLM gateway routing (both chat and
embedding models).
"""
import os

import pytest

from dialogue_benchmarks import llm
from dialogue_benchmarks._paths import DATA_ROOT


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "CACHE_DIR", tmp_path / "llm_cache")


def test_hf_cache_env_vars_redirected_off_default():
    """Regression for the root-owned-cache-files incident: HF_HOME/
    HF_MODULES_CACHE must be set (module-import-time side effect, by the
    time this test file even imports `llm`) to something under DATA_ROOT,
    not left at huggingface's own ~/.cache/huggingface default."""
    assert os.environ.get("HF_HOME") == str(DATA_ROOT / "hf_cache")
    assert os.environ.get("HF_MODULES_CACHE") == str(DATA_ROOT / "hf_cache" / "modules")


def test_embed_unknown_model_raises():
    with pytest.raises(ValueError):
        llm.embed("hello", model="claude-sonnet-5")


def test_embed_single_string_and_caching(monkeypatch):
    calls = []

    def fake_call(model, texts):
        calls.append(list(texts))
        return [[0.1, 0.2, 0.3] for _ in texts], 5

    monkeypatch.setitem(llm._EMBED_PROVIDERS, "text-embedding", fake_call)

    resp = llm.embed("hello world", model="text-embedding-3-small")
    assert resp.vectors == [[0.1, 0.2, 0.3]]
    assert resp.tokens_in == 5
    assert resp.cost == pytest.approx(5 / 1_000_000 * 0.02)
    assert resp.cached is False

    resp2 = llm.embed("hello world", model="text-embedding-3-small")
    assert resp2.vectors == resp.vectors
    assert resp2.cached is True
    assert len(calls) == 1  # second call was served from cache, no provider call


def test_embed_batch_preserves_order(monkeypatch):
    def fake_call(model, texts):
        return [[float(i)] for i, _ in enumerate(texts)], 3

    monkeypatch.setitem(llm._EMBED_PROVIDERS, "text-embedding", fake_call)

    resp = llm.embed(["a", "b", "c"], model="text-embedding-3-large")
    assert resp.vectors == [[0.0], [1.0], [2.0]]


def test_complete_claude_does_not_pass_temperature(monkeypatch):
    """The installed anthropic SDK dropped `temperature` from messages.create()
    entirely -- passing it raises TypeError. complete()'s default temperature=0.0
    must not reach the anthropic client."""
    import anthropic

    captured = {}

    class FakeContentBlock:
        text = "hi"

    class FakeUsage:
        input_tokens = 1
        output_tokens = 1

    class FakeResponse:
        content = [FakeContentBlock()]
        usage = FakeUsage()

    class FakeMessages:
        def create(self, **kwargs):
            captured.update(kwargs)
            return FakeResponse()

    class FakeAnthropic:
        def __init__(self):
            self.messages = FakeMessages()

    monkeypatch.setattr(anthropic, "Anthropic", FakeAnthropic)

    resp = llm.complete("hello", model="claude-sonnet-5")
    assert resp.text == "hi"
    assert "temperature" not in captured


@pytest.mark.parametrize(
    "model, base_url",
    [
        ("openai/gpt-oss-120b", f"{llm._VLLM_HOST}:8002/v1"),
        ("google/gemma-4-31B-it", f"{llm._VLLM_HOST}:8003/v1"),
    ],
)
def test_complete_routes_to_local_vllm_gateway(monkeypatch, model, base_url):
    """openai/gpt-oss-120b is served locally by the vLLM gateway
    (~/work/vllm_server, OpenAI-compatible) -- one fixed port, no API key.
    Both are served by the gateway at DIALOGUE_BENCHMARKS_VLLM_HOST (see
    llm.py's _VLLM_GATEWAY comment). This only checks
    routing/wiring via a fake openai.OpenAI, not a real call (see the
    gateway's own README for a live curl check)."""
    import openai

    captured = {}

    class FakeUsage:
        prompt_tokens = 3
        completion_tokens = 1

    class FakeChoice:
        message = type("Message", (), {"content": "hi"})()

    class FakeResponse:
        choices = [FakeChoice()]
        usage = FakeUsage()

    class FakeCompletions:
        def create(self, **kwargs):
            captured["call_kwargs"] = kwargs
            return FakeResponse()

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured["client_kwargs"] = kwargs
            self.chat = type("Chat", (), {"completions": FakeCompletions()})()

    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)

    resp = llm.complete("hello", model=model)
    assert resp.text == "hi"
    assert resp.cost == 0.0
    assert captured["client_kwargs"]["base_url"] == base_url
    assert captured["call_kwargs"]["model"] == model


def test_embed_routes_to_local_vllm_gateway(monkeypatch):
    """HyperMem's own paper-default embedding model (Qwen/Qwen3-Embedding-4B),
    served the same lazy-start way as the chat models above but via vLLM's
    --runner pooling. Routing/wiring only, no real call.

    nvidia/NV-Embed-v2 (HippoRAG's own default) was meant to be here too, but
    this machine's pinned vLLM (0.19.0) doesn't support the NVEmbedModel
    architecture at all -- see test_embed_local_nv_embed_runs_in_process
    below, where it's loaded in-process via plain transformers.AutoModel
    instead."""
    import openai

    captured = {}

    class FakeUsage:
        prompt_tokens = 3

    class FakeDatum:
        embedding = [0.1, 0.2]

    class FakeResponse:
        data = [FakeDatum()]
        usage = FakeUsage()

    class FakeEmbeddings:
        def create(self, **kwargs):
            captured["call_kwargs"] = kwargs
            return FakeResponse()

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured["client_kwargs"] = kwargs
            self.embeddings = FakeEmbeddings()

    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)

    resp = llm.embed("hello", model="Qwen/Qwen3-Embedding-4B")
    assert resp.vectors == [[0.1, 0.2]]
    assert resp.cost == 0.0
    assert captured["client_kwargs"]["base_url"] == "http://localhost:8008/v1"
    assert captured["call_kwargs"]["model"] == "Qwen/Qwen3-Embedding-4B"


def test_embed_local_minilm_runs_in_process(monkeypatch):
    """a_mem and memoryos both natively default to this (sentence-transformers,
    not OpenAI) -- runs in-process, no gateway/API key involved. Mocks
    SentenceTransformer itself so this stays fast/offline like the other
    provider-routing tests, rather than actually downloading the model."""
    import dialogue_benchmarks.llm as llm_module

    captured = {}

    class FakeSTModel:
        def __init__(self, model_name):
            captured["model_name"] = model_name

        def encode(self, texts, convert_to_numpy=True):
            import numpy as np

            captured["texts"] = texts
            return np.array([[float(len(t))] for t in texts])

    monkeypatch.setattr(
        "sentence_transformers.SentenceTransformer", FakeSTModel, raising=False
    )
    monkeypatch.setattr(llm_module, "_LOCAL_ST_INSTANCES", {})

    resp = llm.embed(["hi", "hello"], model="all-MiniLM-L6-v2")
    assert resp.vectors == [[2.0], [5.0]]
    assert resp.cost == 0.0
    assert captured["model_name"] == "all-MiniLM-L6-v2"


def test_embed_local_nv_embed_runs_in_process(monkeypatch):
    """HippoRAG's own paper-default embedding model (nvidia/NV-Embed-v2) --
    7.85B, needs trust_remote_code for its custom modeling code, loaded via
    plain transformers.AutoModel (not SentenceTransformer -- see
    _call_nvembed_direct's docstring for why) and its own .encode()."""
    import dialogue_benchmarks.llm as llm_module

    captured = {}

    class FakeTensor:
        def __init__(self, values):
            self.values = values

        def cpu(self):
            return self

        def float(self):
            return self

        def numpy(self):
            return self

        def tolist(self):
            return self.values

    class FakeNVEmbedModel:
        def cuda(self):
            return self

        def encode(self, texts):
            captured["texts"] = texts
            return FakeTensor([[1.0] for _ in texts])

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(model_name, **kwargs):
            captured["model_name"] = model_name
            captured["kwargs"] = kwargs
            return FakeNVEmbedModel()

    import transformers

    monkeypatch.setattr(transformers, "AutoModel", FakeAutoModel)
    monkeypatch.setattr(llm_module, "_patch_nvembed_position_embeddings", lambda m: None)
    monkeypatch.setattr(llm_module, "_LOCAL_ST_INSTANCES", {})

    resp = llm.embed("hi", model="nvidia/NV-Embed-v2")
    assert resp.vectors == [[1.0]]
    assert resp.cost == 0.0
    assert captured["model_name"] == "nvidia/NV-Embed-v2"
    assert captured["kwargs"] == {"trust_remote_code": True}


def test_torch_tied_weights_patch_only_fills_the_missing_attribute(monkeypatch):
    """Regression for AttributeError: '<Model>' object has no attribute
    'all_tied_weights_keys' -- transformers>=5.0.0 expects every
    PreTrainedModel subclass to have this (a property normally set up by
    PreTrainedModel's own init), but nvidia/NV-Embed-v2's custom
    (trust_remote_code) modeling_nvembed.py predates that and never gained
    it (same gap hits other pre-5.0 custom models -- see the comment on
    _patch_torch_tied_weights_compat). The patch must only kick in for that
    exact missing attribute -- a module that already has a real attribute
    of any other name, or that attribute set to a real value, must be
    unaffected. Forces the version gate on regardless of which transformers
    is actually installed in whatever venv runs this -- this tests the
    patch logic itself, not which environment happens to need it."""
    import torch

    from dialogue_benchmarks import llm as llm_module

    monkeypatch.setattr(llm_module, "_transformers_major_version", lambda: 5)
    llm_module._TORCH_TIED_WEIGHTS_PATCHED = False
    original_getattr = torch.nn.Module.__getattr__
    try:
        llm_module._patch_torch_tied_weights_compat()

        class Bare(torch.nn.Module):
            pass

        m = Bare()
        assert m.all_tied_weights_keys == {}

        class WithAttr(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.all_tied_weights_keys = {"a": "b"}

        assert WithAttr().all_tied_weights_keys == {"a": "b"}

        with pytest.raises(AttributeError):
            Bare().some_other_missing_attr
    finally:
        torch.nn.Module.__getattr__ = original_getattr
        llm_module._TORCH_TIED_WEIGHTS_PATCHED = False


def test_transformers_legacy_cache_patch_restores_removed_methods():
    """Regression for AttributeError: type object 'DynamicCache' has no
    attribute 'from_legacy_cache' -- transformers>=5.0.0 dropped the legacy
    tuple-based KV-cache format, but nvidia/NV-Embed-v2's forward() still
    calls DynamicCache.from_legacy_cache(None) and .get_usable_length()
    unconditionally (see _patch_transformers_legacy_cache_compat's
    docstring). Only checks the None-input path this codepath actually
    hits, not a general from_legacy_cache reimplementation."""
    from transformers.cache_utils import DynamicCache

    from dialogue_benchmarks import llm as llm_module

    llm_module._TRANSFORMERS_LEGACY_CACHE_PATCHED = False
    had_from_legacy = hasattr(DynamicCache, "from_legacy_cache")
    had_get_usable = hasattr(DynamicCache, "get_usable_length")
    try:
        llm_module._patch_transformers_legacy_cache_compat()

        cache = DynamicCache.from_legacy_cache(None)
        assert isinstance(cache, DynamicCache)
        assert cache.get_usable_length(5) == cache.get_seq_length()
    finally:
        if not had_from_legacy:
            del DynamicCache.from_legacy_cache
        if not had_get_usable:
            del DynamicCache.get_usable_length
        llm_module._TRANSFORMERS_LEGACY_CACHE_PATCHED = False


def test_nvembed_position_embeddings_patch_threads_rotary_through_layers(monkeypatch):
    """Regression for TypeError: cannot unpack non-iterable NoneType object
    at `cos, sin = position_embeddings` -- nvidia/NV-Embed-v2's
    BidirectionalMistralModel.forward() calls each Mistral decoder layer
    without position_embeddings (newer transformers computes RoPE once at
    the model level via rotary_emb and threads it through, rather than each
    attention layer computing it internally). Checks the wrapped layer
    computes position_embeddings via rotary_emb and passes it to the
    original forward, and that use_cache gets forced off. Forces the
    version gate on regardless of which transformers is actually installed
    in whatever venv runs this -- this tests the patch logic itself, not
    which environment happens to need it."""
    import types

    import torch

    from dialogue_benchmarks import llm as llm_module

    monkeypatch.setattr(llm_module, "_transformers_major_version", lambda: 5)

    class FakeRotaryEmbedding:
        def __init__(self, config):
            self.config = config

        def __call__(self, hidden_states, position_ids):
            return ("cos", "sin")

    monkeypatch.setattr(
        "transformers.models.mistral.modeling_mistral.MistralRotaryEmbedding",
        FakeRotaryEmbedding,
    )

    captured_calls = []

    class FakeLayer(torch.nn.Module):
        def forward(self, hidden_states, **kwargs):
            captured_calls.append(kwargs)
            return (hidden_states,)

    class FakeConfig:
        use_cache = True
        head_dim = None
        hidden_size = 4096
        num_attention_heads = 32

    embedding_model = types.SimpleNamespace(
        config=FakeConfig(), layers=[FakeLayer(), FakeLayer()]
    )
    nvembed_model = types.SimpleNamespace(embedding_model=embedding_model)

    llm_module._patch_nvembed_position_embeddings(nvembed_model)

    assert embedding_model.config.use_cache is False
    assert embedding_model.config.head_dim == 128
    for layer in embedding_model.layers:
        layer.forward(torch.zeros(1), position_ids=None, attention_mask=None)
    assert len(captured_calls) == 2
    for call_kwargs in captured_calls:
        assert call_kwargs["position_embeddings"] == ("cos", "sin")
