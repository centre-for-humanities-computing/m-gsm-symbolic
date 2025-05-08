import os
import json
from dataclasses import asdict
import argparse
import logging

os.chdir("..")
from m_gsm_symbolic.sofie_test_parser import AnnotatedQuestion
from m_gsm_symbolic.replacements_list import default_replacements

logger = logging.getLogger(__name__)

def parser():
    """ The user can specify how many variations should be generated of one JSON input """
    parser = argparse.ArgumentParser()
    parser.add_argument("--nvariations",
                        "-n",
                        type = int,
                        required = False,
                        default = 3,
                        help = "Number of variations to generate per input (return int)")
    parser.add_argument("--attempts",
                        "-a",
                        type = int,
                        required = False,
                        default = 5,
                        help = "Number of attempts to ensure unique variations (return int)")
    args = parser.parse_args()
    return args

def main():

    args = parser()

    input_dir = "../data/templates/dan/symbolic"
    output_dir = "../data/templates/dan/symbolic-variations"

    for filename in os.listdir(input_dir):
        if filename.endswith(".json"):
            file_path = os.path.join(input_dir, filename)

            with open(file_path, "r") as f:
                json_data = f.read()

            for i in range(args.nvariations):
                
                question = AnnotatedQuestion.from_json(json_data)

                input_id = question.id_shuffled
                logger.critical(f"Generating variation number {i+1} for question {input_id}")

                generated_question = question.generate_question(default_replacements)
                generated_question_dict = asdict(generated_question)

                with open(f"{output_dir}/00_{input_id}_v{i+1}_HEYY.json", "w", encoding = 'utf-8') as output_file:
                    json.dump(generated_question_dict, output_file, ensure_ascii = False)

if __name__ == "__main__":
    main()
