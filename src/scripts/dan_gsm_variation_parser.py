import os
import json
from dataclasses import asdict
import argparse
import logging

os.chdir("..")
from m_gsm_symbolic.sofie_test_parser import AnnotatedQuestion

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
                json_data = f.read() # read file as str
                question = AnnotatedQuestion.from_json(json_data)

                input_id = question.id_shuffled # id of specific json input
                logger.critical(f"Processing question ID: {input_id}")

                # Generate N variations
                for i in range(args.nvariations):

                    logger.critical(f"Generating variation: {i+1}")

                    # load as txt instead
                    default_replacements = {
                        "names": ["Sofie", "Andrea", "Freja", "Ida", "Clara", "Anna"],
                        "names_male": ["Christian", "Ole", "Erik", "Niels", "Kasper"],
                        "multiple_ice": [2, 3],
                        "multi_times": [2, 3],
                    }

                    generated_question = question.generate_question(default_replacements)
                    generated_question_dict = asdict(generated_question) # Convert dataclass to dict

                    # Save the dict as JSON file to specified folder 
                    with open(f"{output_dir}/{input_id}_var_{i+1}.json", "w", encoding = 'utf-8') as output_file:
                        json.dump(generated_question_dict, output_file, ensure_ascii = False)

if __name__ == "__main__":
    main()
