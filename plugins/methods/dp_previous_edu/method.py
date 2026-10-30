"""dp_previous_edu -- chain baseline: EDU i responds to EDU i-1, relation "Comment"."""
import json

from dialogue_benchmarks.contracts import Method as _MethodBase
from dialogue_benchmarks.contracts import MethodOutput


class Method(_MethodBase):
    def __init__(self, relation: str = "Comment", **config):
        super().__init__(relation=relation, **config)
        self.relation = relation

    def answer(self, item) -> MethodOutput:
        n = int((item.metadata or {}).get("n_edus", 0))
        return MethodOutput(answer=json.dumps([[i - 1, i, self.relation] for i in range(1, n)]))
