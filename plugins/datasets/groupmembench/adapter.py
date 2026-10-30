"""Adapter for GroupMemBench --
data/raw/groupmembench/{data/final,code/questions}/<Domain>/...

Source shape:
- `data/final/<Domain>/synthetic_domain_channels_rolevariants_<Domain>.json`:
  dict of channel/project name -> list of message dicts (`msg_node`, `content`,
  `author`, `role`, `timestamp`, `reply_to`, `phase_name`, `topic`, ...). One
  file is the *whole domain's* multi-channel conversation log.
- `code/questions/<Domain>/{abstention,knowledge_update,multi_hop,temporal,
  term_ambiguity,user_implicit}.jsonl`: one QA set per category, each line
  `{id, question, answer, asking_user_id}` -- `asking_user_id` matters because
  some answers depend on who's asking (see meta.json notes).

One `Conversation` = one domain: upstream's own baselines (bm25/dense) pair a
single `--conversation-json` with a single `--questions-jsonl` per run
(baselines/bm25/eval_benchmark.py), and all 6 category files under a domain's
`code/questions/<Domain>/` are meant to run against that same domain's full
channel set -- not a single channel or a single category in isolation.
"""
import json
from dataclasses import dataclass, field
from pathlib import Path

from dialogue_benchmarks.contracts import EvalItem

DOMAINS = ("Finance", "Technology", "Healthcare", "Manufacturing")
QUESTION_CATEGORIES = (
    "abstention",
    "knowledge_update",
    "multi_hop",
    "temporal",
    "term_ambiguity",
    "user_implicit",
)


@dataclass
class GroupMemBenchQuestion:
    id: str
    question: str
    answer: str
    asking_user_id: str
    category: str


@dataclass
class GroupMemBenchDomain:
    conv_id: str
    domain: str
    channels: dict  # channel/project name -> [raw message dict, ...]
    qa: list = field(default_factory=list)  # [GroupMemBenchQuestion, ...]


def _load_channels(raw_dir: Path, domain: str) -> dict:
    path = raw_dir / "data" / "final" / domain / f"synthetic_domain_channels_rolevariants_{domain}.json"
    with open(path) as f:
        return json.load(f)


def _load_questions(raw_dir: Path, domain: str) -> list:
    questions = []
    for category in QUESTION_CATEGORIES:
        path = raw_dir / "code" / "questions" / domain / f"{category}.jsonl"
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                questions.append(
                    GroupMemBenchQuestion(
                        id=d["id"],
                        question=d["question"],
                        answer=d["answer"],
                        asking_user_id=d.get("asking_user_id"),
                        category=category,
                    )
                )
    return questions


def load(raw_dir: Path) -> list:
    return [
        GroupMemBenchDomain(
            conv_id=f"groupmembench_{domain}",
            domain=domain,
            channels=_load_channels(raw_dir, domain),
            qa=_load_questions(raw_dir, domain),
        )
        for domain in DOMAINS
    ]


def eval_items(conversation: GroupMemBenchDomain) -> list:
    """Doesn't rely on the generic `.qa` fallback: `asking_user_id` has to reach
    a method (groupmembench_bm25 reads it from `item.metadata`) and the fallback's
    fixed attr allowlist (category/evidence/question_type/qtype) doesn't include
    it."""
    return [
        EvalItem(
            conversation_id=conversation.conv_id,
            question=q.question,
            reference=q.answer,
            metadata={"asking_user_id": q.asking_user_id, "category": q.category, "id": q.id},
        )
        for q in conversation.qa
    ]
