"""CLI: stage-2 "Learning to Avoid Misidentification" (fine) training
(paper Eq. 4-6). Continues training a stage-1 checkpoint with the
contrastive loss over MLM-regenerated persona-inconsistent negatives.

    .venv/bin/python plugins/methods/ltkm/train_stage2_fine.py \\
        --examples /path/to/dialogue_examples.json \\
        --stage1-checkpoint /path/to/checkpoints/stage1_gpt2 \\
        --entailment-scorer /path/to/checkpoints/entailment_scorer \\
        --output-dir /path/to/checkpoints/stage2_gpt2
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ltkm_lib.entailment_scorer import EntailmentScorer
from ltkm_lib.generator import PersonaDialogueGenerator
from ltkm_lib.negative_sampler import NegativeSampleGenerator
from ltkm_lib.training import Stage2Config, load_dialogue_examples, train_stage2_fine


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", required=True, help="Path to a dialogue-examples JSON file")
    parser.add_argument("--stage1-checkpoint", required=True, help="Path to the stage-1 fine-tuned generator")
    parser.add_argument("--entailment-scorer", required=True, help="Path to a trained EntailmentScorer checkpoint")
    parser.add_argument("--mlm", default="bert-base-uncased", help="Pretrained (not fine-tuned) MLM for negatives")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-negatives", type=int, default=3, help="Paper's K")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-6)
    args = parser.parse_args()

    examples = load_dialogue_examples(args.examples)
    generator = PersonaDialogueGenerator(args.stage1_checkpoint)
    entailment_scorer = EntailmentScorer(args.entailment_scorer)
    negative_sampler = NegativeSampleGenerator(args.mlm)

    train_stage2_fine(
        generator,
        examples,
        entailment_scorer,
        negative_sampler,
        config=Stage2Config(num_negatives=args.num_negatives),
        epochs=args.epochs,
        lr=args.lr,
    )
    generator.save(args.output_dir)
    print(f"Saved stage-2 (K+A) model to {args.output_dir}")


if __name__ == "__main__":
    main()
