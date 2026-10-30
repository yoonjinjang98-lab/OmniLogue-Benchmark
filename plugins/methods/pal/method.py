"""pal -- PAL: Persona-Aware Alignment Framework for Personalized Dialogue
Generation (arXiv 2511.10215, github.com/kylokano/PAL), Llama-3.1-8B setting:

1. persona-aware learning: LoRA SFT on a mixture of two sub-tasks over
   PersonaChat -- "select" (name the one persona sentence relevant to the
   next response, or "No personal description.") and "generate" (the
   response, given that highlighted persona)
2. persona alignment: DPO from the stage-1 model, chosen = the gold
   response, rejected = the stage-1 model's own response generated *without*
   the persona (lr 1e-6, beta 0.1)
3. inference, "select then generate": predict the relevant persona, then
   generate with it highlighted (beam search 3, <= 40 new tokens)

Prompts are PAL's own, vendored verbatim (vendor/pal_prompts.py). Training:
train.py (see its docstring for the one data deviation -- persona relevance
labels). This method loads base + stage-1 adapter (merged) + DPO adapter
from DATA_ROOT/checkpoints/pal/<base>/. For personachat (any dialogue
dataset whose items carry metadata["persona"] and ["history"]).
"""
import importlib.util
import re
from pathlib import Path

from dialogue_benchmarks._paths import DATA_ROOT
from dialogue_benchmarks.contracts import Method as _MethodBase
from dialogue_benchmarks.contracts import MethodOutput

_spec = importlib.util.spec_from_file_location("_pal_prompts", Path(__file__).resolve().parent / "vendor" / "pal_prompts.py")
P = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P)

CHECKPOINTS = DATA_ROOT / "checkpoints" / "pal"


def run_dir(base_model: str) -> Path:
    return CHECKPOINTS / base_model.replace("/", "__")


def strip_history(history: list) -> list:
    """Item histories may carry "Speaker: " prefixes (other adapters); PAL
    formats turns itself as person1/person2."""
    return [h.split(": ", 1)[1] if re.match(r"^[^:]{1,40}: ", h) and not h.startswith("http") else h for h in history]


def load_model(base_model: str, stage1: Path, dpo: Path = None):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(base_model, dtype=torch.bfloat16, device_map={"": 0})
    model = PeftModel.from_pretrained(model, str(stage1)).merge_and_unload()
    if dpo is not None:
        model = PeftModel.from_pretrained(model, str(dpo))
    return model.eval(), tok


def generate(model, tok, prompt: str, max_new_tokens: int = 40, num_beams: int = 3) -> str:
    import torch

    enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **enc, max_new_tokens=max_new_tokens, num_beams=num_beams, do_sample=False,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
        )
    return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()


class Method(_MethodBase):
    def __init__(self, base_model: str = "meta-llama/Llama-3.1-8B-Instruct", checkpoint_dir: str = None, **config):
        config.pop("model", None)  # --model names the prompting LLM of other methods
        super().__init__(base_model=base_model, checkpoint_dir=checkpoint_dir, **config)
        self.base_model = base_model
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else run_dir(base_model)
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        stage1, dpo = self.checkpoint_dir / "stage1_sft", self.checkpoint_dir / "stage2_dpo"
        if not (stage1 / "adapter_config.json").exists():
            raise FileNotFoundError(
                f"pal: no trained checkpoint under {self.checkpoint_dir} -- run "
                f"`.venv/bin/python plugins/methods/pal/train.py --base {self.base_model}` first"
            )
        self._model, self._tok = load_model(self.base_model, stage1, dpo if (dpo / "adapter_config.json").exists() else None)

    def prepare(self, conversation) -> None:
        self._load()

    def answer(self, item) -> MethodOutput:
        meta = item.metadata or {}
        persona = list(meta.get("persona") or [])
        history = strip_history(list(meta.get("history") or []))
        selected = generate(self._model, self._tok, P.build("select", persona, history)).split("\n")[0].strip()
        if not re.match(r"^\d+\.", selected):
            selected = P.NO_PERSONA
        response = generate(self._model, self._tok, P.build("predict_select_then_generate", persona, history, selected))
        return MethodOutput(answer=response.split("\n")[0].strip())

    def reset(self) -> None:
        pass
