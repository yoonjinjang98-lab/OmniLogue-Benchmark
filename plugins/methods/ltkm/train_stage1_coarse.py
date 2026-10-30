"""CLI: stage-1 "Learning to Know Myself" (coarse) training (paper Eq. 1).
Augments a dialogue-example set with persona-aware Q/A pairs (via a trained
relation extractor) and fine-tunes a base generator with plain MLE loss.

    .venv/bin/python plugins/methods/ltkm/train_stage1_coarse.py \\
        --examples /path/to/dialogue_examples.json \\
        --relation-extractor /path/to/checkpoints/relation_extractor \\
        --output-dir /path/to/checkpoints/stage1_gpt2 \\
        --base-model gpt2
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ltkm_lib.generator import PersonaDialogueGenerator
from ltkm_lib.question_construction import PersonaAwareQuestionConstructor
from ltkm_lib.relation_extractor import RelationExtractor
from ltkm_lib.training import build_stage1_examples, load_dialogue_examples, train_stage1_coarse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", required=True, help="Path to a dialogue-examples JSON file")
    parser.add_argument("--relation-extractor", required=True, help="Path to a trained RelationExtractor checkpoint")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-model", default="microsoft/DialoGPT-medium")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    args = parser.parse_args()

    examples = load_dialogue_examples(args.examples)
    relation_extractor = RelationExtractor(args.relation_extractor)
    question_constructor = PersonaAwareQuestionConstructor(relation_extractor)
    augmented = build_stage1_examples(examples, question_constructor)

    generator = PersonaDialogueGenerator(args.base_model)
    train_stage1_coarse(generator, augmented, epochs=args.epochs, lr=args.lr)
    generator.save(args.output_dir)
    print(f"Saved stage-1 model to {args.output_dir} ({len(augmented)} examples, {len(examples)} original)")


if __name__ == "__main__":
    main()
