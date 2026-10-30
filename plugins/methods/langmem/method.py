"""langmem -- family C baseline (BASELINE_METRIC_PLAN.md §4): wraps
langchain-ai/langmem's `create_memory_store_manager`, which ingests a
conversation, decides insert/update/delete over existing memories via
tool-calling, and persists the result into a LangGraph `BaseStore` with a
semantic index -- then answers questions from the memories `manager.search()`
retrieves for that query.

Two integration points, both extension seams langmem/LangGraph officially
support (unlike mem0, which has no bring-your-own-callable provider -- see
plugins/methods/mem0/method.py):
- `create_memory_store_manager(model, ...)` accepts any `BaseChatModel`
  instance in place of a model name string.
- `InMemoryStore(index={"embed": ...})` accepts a plain
  `Callable[[Sequence[str]], list[list[float]]]` in place of a LangChain
  `Embeddings` object.

`_DBChatModel(BaseChatModel)` routes every plain chat call through
`dialogue_benchmarks.llm.complete()`. The one real wrinkle: langmem's
extraction pipeline (via the `trustcall` library) always drives the model
through `.bind_tools(...)` + tool-calling, even for the default unstructured
`Memory` schema (a single `content: str` field) -- there's no non-tool-calling
extraction path to opt into. `complete()` has no tool-calling support (by
design -- CLAUDE.md routes all chat through it), so `_DBChatModel` emulates
tool-calling in text: `bind_tools()` records the tool schemas, and
`_generate()` prompts the model to reply with a JSON array of
`{"name": ..., "arguments": {...}}` calls instead of using a real
function-calling API, then parses that back into `AIMessage.tool_calls`. This
is the same kind of shim community LangChain chat-model integrations use for
backends without native tool-calling.
"""
import json
import re
import uuid
from typing import Any, List, Optional, Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.store.memory import InMemoryStore
from langmem import create_memory_store_manager

from dialogue_benchmarks.contracts import Method as _MethodBase
from dialogue_benchmarks.contracts import MethodOutput
from dialogue_benchmarks.llm import complete, embed

_EMBED_MODEL = "text-embedding-3-small"
_EMBED_DIMS = 1536


def _split_system(messages: Sequence[BaseMessage]):
    system = None
    turns = []
    for m in messages:
        if m.type == "system" and system is None:
            system = m.content
        else:
            turns.append(m)
    return system, turns


def _format_turns(turns: Sequence[BaseMessage]) -> str:
    return "\n\n".join(f"{m.type}: {m.content}" for m in turns)


def _extract_json_array(text: str):
    """Tool-call responses are asked for as a bare JSON array; strip markdown
    code fences some models add anyway before parsing."""
    text = (text or "").strip()
    match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if match:
        text = match.group(1).strip()
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return []
    return parsed if isinstance(parsed, list) else []


class _DBChatModel(BaseChatModel):
    """Minimal BaseChatModel routing chat calls through
    dialogue_benchmarks.llm.complete(), with a text-based tool-calling shim
    for trustcall's structured extraction (see module docstring)."""

    model_name: str = "claude-sonnet-5"
    max_tokens: int = 1024

    @property
    def _llm_type(self) -> str:
        return "dialogue_benchmarks"

    def bind_tools(self, tools: Sequence, *, tool_choice: Optional[Any] = None, **kwargs):
        # trustcall passes tool specs as OpenAI-format dicts on the first call but as
        # raw pydantic model classes on its own internal retry/patch tools (PatchDoc,
        # PatchFunctionErrors, ...) -- normalize everything through LangChain's own
        # converter so _generate_tool_call always sees the same {"type": "function",
        # "function": {...}} shape regardless of which path called bind_tools().
        formatted = [convert_to_openai_tool(t) for t in tools]
        return self.bind(tools=formatted, tool_choice=tool_choice, **kwargs)

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager=None,
        **kwargs,
    ) -> ChatResult:
        system, turns = _split_system(messages)
        tools = kwargs.get("tools")
        if tools:
            message = self._generate_tool_call(system, turns, tools)
        else:
            resp = complete(_format_turns(turns), model=self.model_name, system=system, max_tokens=self.max_tokens)
            message = AIMessage(content=resp.text, id=f"run-{uuid.uuid4()}")
        return ChatResult(generations=[ChatGeneration(message=message)])

    def _generate_tool_call(self, system, turns, tools) -> AIMessage:
        tool_specs = []
        for t in tools:
            fn = t["function"]
            tool_specs.append(
                {"name": fn.get("name"), "description": fn.get("description"), "parameters": fn.get("parameters")}
            )
        prompt = (
            "You must decide which of the following tools (if any) to call based on the conversation "
            "below. You may call zero, one, or multiple tools.\n\n"
            f"### Available tools\n{json.dumps(tool_specs, indent=2)}\n\n"
            f"### Conversation\n{_format_turns(turns)}\n\n"
            "Respond with ONLY a JSON array, no other text. Each element must be "
            '{"name": "<tool name>", "arguments": {<arguments matching that tool\'s "parameters" schema>}}. '
            "Return [] if no tool call is appropriate."
        )
        resp = complete(prompt, model=self.model_name, system=system, max_tokens=self.max_tokens)
        calls = _extract_json_array(resp.text)
        tool_calls = [
            {"name": c.get("name"), "args": c.get("arguments", {}), "id": f"call_{uuid.uuid4().hex[:8]}"}
            for c in calls
            if isinstance(c, dict) and c.get("name")
        ]
        return AIMessage(content="", tool_calls=tool_calls, id=f"run-{uuid.uuid4()}")


def _embed_texts(texts: Sequence[str]) -> list:
    return embed(list(texts), model=_EMBED_MODEL).vectors


def _extract_chunks(conversation) -> list:
    """Same chunking convention as running_summary/mem0: one chunk per session
    if session-grouped, else fixed-size windows over flat `.turns`."""
    if hasattr(conversation, "sessions"):
        return [
            [{"role": "user", "content": f"{turn.speaker}: {turn.text}"} for turn in conversation.sessions[idx]]
            for idx in sorted(conversation.sessions)
        ]
    if hasattr(conversation, "turns"):
        window = 20
        turns = conversation.turns
        return [
            [{"role": "user", "content": f"{t.speaker}: {t.text}"} for t in turns[start : start + window]]
            for start in range(0, len(turns), window)
        ]
    raise ValueError(
        "langmem: conversation has neither `.sessions` (dict of session_idx -> turns) nor a flat "
        "`.turns` list -- can't extract chunks for this dataset's shape"
    )


class Method(_MethodBase):
    def __init__(
        self,
        model: str = "claude-sonnet-5",
        embed_model: str = "text-embedding-3-small",
        top_k: int = 10,
        max_tokens: int = 512,
        **config,
    ):
        super().__init__(model=model, embed_model=embed_model, top_k=top_k, max_tokens=max_tokens, **config)
        self.model = model
        self.embed_model = embed_model
        self.top_k = top_k
        self.max_tokens = max_tokens
        self._manager = None

    def prepare(self, conversation) -> None:
        store = InMemoryStore(index={"dims": _EMBED_DIMS, "embed": _embed_texts})
        chat_model = _DBChatModel(model_name=self.model)
        self._manager = create_memory_store_manager(
            chat_model, namespace=("memories", "conversation"), store=store
        )
        for chunk in _extract_chunks(conversation):
            self._manager.invoke({"messages": chunk})

    def answer(self, item) -> MethodOutput:
        results = self._manager.search(query=item.question, limit=self.top_k)
        retrieved = []
        for r in results:
            value = r.value if hasattr(r, "value") else r.get("value", {})
            # `value` is the extracted Memory pydantic object itself (its default
            # __repr__ happens to print like a dict, which is misleading) -- not a
            # plain dict, so attribute access has to come first.
            content = getattr(value, "content", None)
            if content is None and isinstance(value, dict):
                content = value.get("content")
            retrieved.append(content if content is not None else str(value))

        question = item.question
        if item.options:
            question += "\n\nChoose one of: " + ", ".join(item.options)
        passages = "\n".join(f"- {m}" for m in retrieved) or "(no memories retrieved)"
        prompt = (
            "Answer the question using only the information in the memories below. "
            "Be concise and direct -- answer with just the fact asked for, no extra explanation.\n\n"
            f"### Memories\n{passages}\n\n"
            f"### Question\n{question}\n\n"
            "### Answer"
        )
        resp = complete(prompt, model=self.model, max_tokens=self.max_tokens)
        return MethodOutput(
            answer=resp.text.strip(),
            retrieved=retrieved,
            tokens=resp.tokens_in + resp.tokens_out,
            latency=resp.latency,
            cost=resp.cost,
        )

    def reset(self) -> None:
        self._manager = None
