import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from dialogue_benchmarks._paths import PLUGINS_ROOT
from dialogue_benchmarks.contracts import Method

PLUGINS_DIR = PLUGINS_ROOT / "methods"


@dataclass
class MethodInfo:
    name: str
    path: Path
    meta: dict
    implemented: bool


def list_methods() -> list[MethodInfo]:
    infos = []
    if not PLUGINS_DIR.exists():
        return infos
    for d in sorted(PLUGINS_DIR.iterdir()):
        meta_path = d / "meta.json"
        if not d.is_dir() or not meta_path.exists():
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        infos.append(MethodInfo(name=d.name, path=d, meta=meta, implemented=(d / "method.py").exists()))
    return infos


def get_method_info(name: str) -> MethodInfo:
    path = PLUGINS_DIR / name
    meta_path = path / "meta.json"
    if not meta_path.exists():
        raise KeyError(f"'{name}' is not a registered method (no plugins/methods/{name}/meta.json)")
    with open(meta_path) as f:
        meta = json.load(f)
    return MethodInfo(name=name, path=path, meta=meta, implemented=(path / "method.py").exists())


def _import_method_module(name: str):
    info = get_method_info(name)
    if not info.implemented:
        raise NotImplementedError(
            f"method '{name}' has no method.py yet (plugins/methods/{name}/ is a placeholder). "
            f"See README.md for how to implement one."
        )
    module_name = f"dialogue_benchmarks._methods.{name}"
    if module_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(module_name, info.path / "method.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    return sys.modules[module_name]


class _LegacyFunctionMethod(Method):
    """Wraps a plugin that only exposes answer(conversation, question) -> str
    (the original, simplest form of the contract) so it satisfies the
    prepare/answer/reset shape the runner drives everything through.

    Ingestion cost is paid once per question, not once per conversation --
    fine for a stateless baseline like smoke_test, wasteful for anything that
    builds an index or calls a memory framework. Real baselines should define
    a `Method` subclass instead so prepare() runs once per conversation.
    """

    def __init__(self, answer_fn, **config):
        super().__init__(**config)
        self._answer_fn = answer_fn
        self._conversation = None

    def prepare(self, conversation) -> None:
        self._conversation = conversation

    def answer(self, item):
        return self._answer_fn(self._conversation, item.question)


def load_method(name: str, **config) -> Method:
    """Instantiate a method's Method-shaped entry point.

    Supports two plugin styles, detected by what method.py exposes:
    - `class Method(dialogue_benchmarks.contracts.Method): ...`  (preferred for
      anything with real ingestion cost -- RAG index, memory-agent state, ...)
    - a bare `def answer(conversation, question) -> str` function (legacy/simple
      style -- auto-wrapped, kept for trivial baselines like smoke_test)
    """
    module = _import_method_module(name)
    if hasattr(module, "Method"):
        return module.Method(**config)
    if hasattr(module, "answer"):
        return _LegacyFunctionMethod(module.answer, **config)
    raise NotImplementedError(
        f"method '{name}' (plugins/methods/{name}/method.py) exposes neither a `Method` class "
        f"nor an `answer(conversation, question)` function"
    )


def load_answer_fn(name: str):
    """Back-compat helper: returns a plain answer(conversation, question) -> str
    callable for methods that expose one directly. Prefer load_method() for new
    code -- it gives you prepare/reset control for methods with ingestion cost."""
    module = _import_method_module(name)
    if not hasattr(module, "answer"):
        raise NotImplementedError(
            f"method '{name}' only exposes a `Method` class, not a bare answer() function; "
            f"use load_method() instead"
        )
    return module.answer
