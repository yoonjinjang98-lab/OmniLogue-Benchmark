"""no_persona -- persona_prompt with use_persona=False (and nothing else changed).

Loads persona_prompt's implementation from its sibling plugin directory
rather than copying it, so the two can never drift apart.
"""
import importlib.util
from pathlib import Path

_SIBLING = Path(__file__).resolve().parent.parent / "persona_prompt" / "method.py"
_spec = importlib.util.spec_from_file_location("_persona_prompt_for_no_persona", _SIBLING)
_persona_prompt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_persona_prompt)


class Method(_persona_prompt.Method):
    def __init__(self, **config):
        config["use_persona"] = False
        super().__init__(**config)
