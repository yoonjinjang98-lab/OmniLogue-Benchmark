"""addressee_last_speaker -- recency heuristic for addressee recognition.

Ranks item.candidates (speaker labels) by how recently each spoke in
item.metadata["history"] ("Speaker: text" lines), most recent first.
"""
import json

from dialogue_benchmarks.contracts import Method as _MethodBase
from dialogue_benchmarks.contracts import MethodOutput


class Method(_MethodBase):
    def answer(self, item) -> MethodOutput:
        candidates = list(item.candidates or [])
        last_seen = {}
        for pos, line in enumerate((item.metadata or {}).get("history", [])):
            last_seen[line.split(": ", 1)[0]] = pos
        ranking = sorted(range(len(candidates)), key=lambda i: last_seen.get(candidates[i], -1), reverse=True)
        return MethodOutput(answer=json.dumps(ranking))
