import argparse
import logging
from pathlib import Path

from m_gsm_symbolic.gsm_variation_parser import AnnotatedQuestion
from m_gsm_symbolic.replacements_list import replacements

logger = logging.getLogger(__name__)

def parser():
    """The user can specify how many variations should be generated of one JSON input"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir",
                        "-i",
                        type = str,
                        required = False,
                        default = "../data/templates/dan/test", # "../data/templates/dan/symbolic",
                        help = "Provide input directory containing JSON data")
    parser.add_argument("--nvariations",
                        "-n",
                        type = int,
                        required = False,
                        default = 2,
                        help = "Number of variations to generate per input (return int)")
    args = parser.parse_args()
    return args


def create_variants(args):
    # take args parser argument and make sure it will stay as it was defined
    return


def main():
    args = parser()
    #create_variants(input_dir = args.input_dir, n_variants = args.n_variants) #???

    input_dir = Path(args.input_dir)

    if not input_dir.exists() and input_dir.is_dir():
        raise ValueError(f"Input dir: '{input_dir}' does not exist")

    output_dir = input_dir.parent / f"{input_dir.name}-variants"
    output_dir.mkdir(parents=True, exist_ok=True)

    for filename in input_dir.glob("*.json"):
        for i in range(args.nvariations):
            question = AnnotatedQuestion.from_json(filename)

            input_id = question.id_shuffled
            logger.critical(
                f"Generating variation number {i + 1} for question {input_id}"
            )

            generated_question = question.generate_question(replacements)

            generated_question.to_json(output_dir / f"{input_id}_{i+1}_TESTER.json")
            logger.critical(f"Saved variation {i+1} for question {input_id} to {output_dir.name}")


if __name__ == "__main__":
    main()