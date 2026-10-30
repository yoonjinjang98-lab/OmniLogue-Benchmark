"""llm_sft -- "LLM + SFT": a LoRA-fine-tuned open LLM generating the next turn.

The 2025-26 generation papers' standard baseline across all three axes this
harness covers -- multi-party (DRCR, ACL 2026: Llama-3.2-3B / Qwen3-4B /
Qwen3-8B + SFT on Ubuntu IRC), long-term (LD-Agent, NAACL 2025: LoRA-tuned
ChatGLM3-6B on MSC / Conversation Chronicles) and persona (PAL 2025:
fine-tuned Llama-3.1-8B on PersonaChat). One implementation serves every
task_type="dialogue" dataset:

    prompt = [earlier sessions, if the conversation has `.sessions`]
           + [persona sentences, if item.metadata has them]
           + item.metadata["history"]
           + "write <speaker>'s next message [replying to <addressee>]"

`build_messages()` is the single prompt definition: train.py builds its
training pairs with the same function, so what the model is fine-tuned on
is exactly what it sees at test time.

Train first (train.py, one LoRA adapter per dataset x base model), then run:

    .venv/bin/python plugins/methods/llm_sft/train.py --dataset ubuntu_irc_gen --base Qwen/Qwen3-8B
    .venv/bin/db run --dataset ubuntu_irc_gen --methods llm_sft --config '{"base_model": "Qwen/Qwen3-8B"}'

The adapter is found at DATA_ROOT/checkpoints/llm_sft/<dataset>/<base>/
(dataset = config "train_dataset", else the dataset being evaluated), or
given explicitly as config "adapter". Runs in-process on the visible GPU --
set CUDA_VISIBLE_DEVICES to pick the GPU.
"""
from pathlib import Path

from dialogue_benchmarks import usage
from dialogue_benchmarks._paths import DATA_ROOT
from dialogue_benchmarks.contracts import Method as _MethodBase
from dialogue_benchmarks.contracts import MethodOutput

CHECKPOINTS = DATA_ROOT / "checkpoints" / "llm_sft"
MAX_MEMORY_CHARS = 6000
MAX_HISTORY_TURNS = 15


def adapter_dir(dataset: str, base_model: str) -> Path:
    return CHECKPOINTS / dataset / base_model.replace("/", "__")


def _memory_text(conversation) -> str:
    sessions = getattr(conversation, "sessions", None)
    if not sessions:
        return ""
    datetimes = getattr(conversation, "session_datetimes", {}) or {}
    lines = []
    for idx in sorted(sessions):
        dt = datetimes.get(idx)
        lines.append(f"[Session {idx}" + (f" -- {dt}]" if dt else "]"))
        lines.extend(f"{t.speaker}: {t.text}" for t in sessions[idx])
    text = "\n".join(lines)
    # keep the most recent part when it doesn't fit
    return text[-MAX_MEMORY_CHARS:]


def build_messages(conversation, metadata: dict) -> list:
    """Chat messages (system + user) for one generation item -- shared by
    training and inference."""
    speaker = metadata.get("speaker") or "the next speaker"
    parts = []
    memory = _memory_text(conversation)
    if memory:
        parts.append("Earlier sessions:\n" + memory)
    persona = metadata.get("persona") or []
    if persona:
        parts.append(f"Persona of {speaker}:\n" + "\n".join(f"- {p}" for p in persona))
    history = (metadata.get("history") or [])[-MAX_HISTORY_TURNS:]
    parts.append("Conversation:\n" + ("\n".join(history) if history else "(start of conversation)"))
    task = f"Write {speaker}'s next message"
    if metadata.get("addressee"):
        task += f", replying to {metadata['addressee']}"
    parts.append(task + ".")
    return [
        {"role": "system", "content": "You continue conversations, writing one natural message as the given speaker."},
        {"role": "user", "content": "\n\n".join(parts)},
    ]


def chat_prompt(tokenizer, messages: list) -> str:
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    try:  # Qwen3: answer directly, no <think> block
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


class Method(_MethodBase):
    def __init__(
        self,
        base_model: str = "Qwen/Qwen3-8B",
        adapter: str = None,
        train_dataset: str = None,
        max_new_tokens: int = 50,
        **config,
    ):
        # `db run --model` names the *prompting* LLM for the other methods in
        # the same run (e.g. gemma) -- never this method's base, which only
        # comes from config "base_model"
        config.pop("model", None)
        super().__init__(
            base_model=base_model, adapter=adapter, train_dataset=train_dataset, max_new_tokens=max_new_tokens, **config
        )
        self.base_model = base_model
        self.adapter = adapter
        self.train_dataset = train_dataset
        self.max_new_tokens = max_new_tokens
        self._model = None
        self._tok = None
        self._conversation = None

    def _load(self):
        if self._model is not None:
            return
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dataset = self.train_dataset or usage.current().get("dataset")
        path = Path(self.adapter) if self.adapter else adapter_dir(dataset or "?", self.base_model)
        if not (path / "adapter_config.json").exists():
            raise FileNotFoundError(
                f"llm_sft: no trained LoRA adapter at {path} -- train one first: "
                f".venv/bin/python plugins/methods/llm_sft/train.py --dataset {dataset} --base {self.base_model}"
            )
        self._tok = AutoTokenizer.from_pretrained(self.base_model)
        base = AutoModelForCausalLM.from_pretrained(self.base_model, dtype=torch.bfloat16, device_map={"": 0})
        self._model = PeftModel.from_pretrained(base, str(path)).eval()
        self._torch = torch

    def prepare(self, conversation) -> None:
        self._load()
        self._conversation = conversation

    def answer(self, item) -> MethodOutput:
        prompt = chat_prompt(self._tok, build_messages(self._conversation, item.metadata or {}))
        enc = self._tok(prompt, return_tensors="pt", add_special_tokens=False).to(self._model.device)
        with self._torch.no_grad():
            out = self._model.generate(
                **enc,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self._tok.pad_token_id or self._tok.eos_token_id,
            )
        text = self._tok.decode(out[0, enc["input_ids"].shape[1] :], skip_special_tokens=True)
        return MethodOutput(answer=text.strip().split("\n")[0].strip())

    def reset(self) -> None:
        self._conversation = None
