"""Ubuntu IRC addressee recognition -- same raw files as `ubuntu_irc`
(data/raw/ubuntu_irc/, reused via raw_dir's sibling rather than duplicated),
different task: given the context and the next message (and who sends it),
pick which earlier speaker it is addressed to.

item.candidates = the distinct speakers in the context other than the
responder (in order of first appearance), item.reference = the index of the
gold addressee. Instances whose gold addressee isn't among them (ans_adr = -1,
or addressed to someone who never spoke in the window) are skipped -- there's
no correct candidate to pick. Shares ubuntu_irc's parsing (load_split).
"""
import importlib.util
from pathlib import Path

from dialogue_benchmarks.contracts import EvalItem

_SIBLING = Path(__file__).resolve().parent.parent / "ubuntu_irc" / "adapter.py"
_spec = importlib.util.spec_from_file_location("_ubuntu_irc_adapter_for_addressee", _SIBLING)
_ubuntu = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ubuntu)


def load(raw_dir: Path) -> list:
    return _ubuntu.load_split(raw_dir.parent / "ubuntu_irc", "test")


def eval_items(conversation) -> list:
    candidates = []
    for t in conversation.turns:
        if t.speaker and t.speaker != conversation.answer_speaker and t.speaker not in candidates:
            candidates.append(t.speaker)
    if conversation.answer_addressee not in candidates:
        return []
    return [
        EvalItem(
            conversation_id=conversation.conv_id,
            question=(
                f"{conversation.answer_speaker} sends the next message: \"{conversation.answer}\". "
                "Which earlier speaker is it addressed to?"
            ),
            reference=candidates.index(conversation.answer_addressee),
            candidates=candidates,
            metadata={
                "history": [f"{t.speaker}: {t.text}" for t in conversation.turns],
                "speaker": conversation.answer_speaker,
                "response": conversation.answer,
                "category": f"{len(candidates)}_candidates",
            },
        )
    ]
