"""Ubuntu IRC multi-party response *generation* -- same raw files as
`ubuntu_irc` (data/raw/ubuntu_irc/, reused via raw_dir's sibling), the task
those files were originally released for: the GSN format (Hu et al. 2019)
that HeterMPC / MADNet and later multi-party generation papers report
BLEU-1~4 / METEOR / ROUGE-L on.

task_type="dialogue": one item per test instance -- generate the next message
given the context, who is speaking and whom it is addressed to (the speaker
and addressee are inputs in that setting, not something to predict).
metadata carries "history" ("Speaker k: text" lines), "speaker" and
"addressee"; the reference is the gold message. Text is the corpus's own
lowercased, tokenized form ("does n't"); bleu_n tokenizes both sides the same
way so a model's untokenized output isn't penalized for it.
"""
import importlib.util
from pathlib import Path

from dialogue_benchmarks.contracts import EvalItem

_SIBLING = Path(__file__).resolve().parent.parent / "ubuntu_irc" / "adapter.py"
_spec = importlib.util.spec_from_file_location("_ubuntu_irc_adapter_for_gen", _SIBLING)
_ubuntu = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ubuntu)


def load(raw_dir: Path) -> list:
    return load_split(raw_dir, "test")


def load_split(raw_dir: Path, split: str) -> list:
    """train / valid / test (311,725 / 5,000 / 5,000)."""
    return _ubuntu.load_split(raw_dir.parent / "ubuntu_irc", split)


def eval_items(conversation) -> list:
    return [
        EvalItem(
            conversation_id=conversation.conv_id,
            question="produce the next turn",
            reference=conversation.answer,
            metadata={
                "history": [f"{t.speaker}: {t.text}" for t in conversation.turns],
                "speaker": conversation.answer_speaker,
                "addressee": conversation.answer_addressee,
                "category": f"{len({t.speaker for t in conversation.turns})}_speakers",
            },
        )
    ]
