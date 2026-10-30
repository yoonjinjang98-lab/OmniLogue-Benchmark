"""emmdg -- EMMDG (Li & Zhao, "EM Pre-training for Multi-party Dialogue
Response Generation", ACL 2023): BART-base pre-trained with an EM loop that
infers missing addressee labels, fine-tuned on Ubuntu IRC with addressee
indicator embeddings. Its released checkpoint and training code are used
as-is (vendor/mpdrg/, patches noted in place): train.py fine-tunes,
predict.py generates the whole test split once with the code's own
inference path, and this method serves those predictions per item -- the
vendored modules can't be imported into the harness process (see
predict.py). For ubuntu_irc_gen (test split = the vendored data's test.json,
byte for byte).
"""
import json
import subprocess
import sys
from pathlib import Path

from dialogue_benchmarks._paths import DATA_ROOT
from dialogue_benchmarks.contracts import Method as _MethodBase
from dialogue_benchmarks.contracts import MethodOutput

HOME = DATA_ROOT / "checkpoints" / "emmdg"


class Method(_MethodBase):
    def __init__(self, work: str = None, **config):
        config.pop("model", None)  # --model names the prompting LLM of other methods
        super().__init__(work=work, **config)
        self.work = Path(work) if work else HOME / "work"
        self._preds = None

    def _predictions(self) -> dict:
        if self._preds is not None:
            return self._preds
        best = sorted(self.work.rglob("best_model.pth"), key=lambda p: p.stat().st_mtime)
        if not best:
            raise FileNotFoundError(
                f"emmdg: no fine-tuned best_model.pth under {self.work} -- run "
                "`.venv/bin/python plugins/methods/emmdg/train.py` first"
            )
        out = best[-1].parent / "harness_test_predictions.full.json"
        if not out.exists() or out.stat().st_mtime < best[-1].stat().st_mtime:
            subprocess.run(
                [sys.executable, "-W", "ignore", str(Path(__file__).with_name("predict.py")),
                 "--work", str(self.work), "--best", str(best[-1]), "--out", str(out)],
                check=True,
            )
        self._preds = json.loads(out.read_text())
        return self._preds

    def prepare(self, conversation) -> None:
        self._qid = "test-" + str(conversation.conv_id).rsplit("_", 1)[-1]

    def answer(self, item) -> MethodOutput:
        preds = self._predictions()
        if self._qid not in preds:
            raise KeyError(f"emmdg: no prediction for {self._qid} (only the Ubuntu IRC test split is generated)")
        return MethodOutput(answer=preds[self._qid])

    def reset(self) -> None:
        pass
