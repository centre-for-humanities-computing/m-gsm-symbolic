import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path

from m_gsm_symbolic.gsm_variation_parser import AnnotatedQuestion
from m_gsm_symbolic.replacements_list import replacements

logger = logging.getLogger(__name__)

repo_root = Path("./scripts/dan_gsm_variations.py").parent.parent


def parser():
    """The user can specify how many variations should be generated of one JSON input"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        "-i",
        type=str,
        required=False,
        default="../data/templates/dan/symbolic",
        help="Provide input directory containing JSON data",
    )
    parser.add_argument(
        "--nvariations",
        "-n",
        type=int,
        required=False,
        default=2,
        help="Number of variations to generate per input (return int)",
    )
    parser.add_argument(
        "--attempts",
        "-a",
        type=int,
        required=False,
        default=5,
        help="Number of attempts to ensure unique variations (return int)",
    )
    args = parser.parse_args()
    return args


def main():
    args = parser()

    input_dir = Path(args.input_dir)

    if not input_dir.exists() and input_dir.is_dir():
        raise ValueError(f"Input dir: '{input_dir}' does not exist")

    output_dir = input_dir.parent / f"{input_dir.name}-variants"
    output_dir.mkdir(parents=True, exist_ok=True)

    for filename in input_dir.glob("*.json"):
        with filename.open("r", encoding="utf-8") as f:
            json_data = f.read()

        # create_variants(n_variants, attempts, input_dir)

        for i in range(args.nvariations):
            question = AnnotatedQuestion.from_json(json_data)

            input_id = question.id_shuffled
            logger.critical(
                f"Generating variation number {i + 1} for question {input_id}"
            )

            generated_question = question.generate_question(replacements)
            generated_question_dict = asdict(generated_question)

            with open(
                output_dir / f"{input_id}_{i + 1}.json", "w", encoding="utf-8"
            ) as output_file:
                json.dump(generated_question_dict, output_file, ensure_ascii=False)
                logger.critical(
                    f"Saved variation {i + 1} for question {input_id} to {output_file.name}"
                )


if __name__ == "__main__":
    main()
