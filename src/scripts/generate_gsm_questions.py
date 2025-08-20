import argparse
import logging
from pathlib import Path
import random

from m_gsm_symbolic.gsm_parser import AnnotatedQuestion, Question
from m_gsm_symbolic.load_data import load_replacements

logger = logging.getLogger(__name__)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate samples from annotated JSON templates."
    )
    parser.add_argument("template_path", help="Path to the JSON template file(s).")
    parser.add_argument(
        "num_samples", type=int, help="Number of samples to generate for each template."
    )
    parser.add_argument(
        "language",
        choices=["dan", "eng"],
        help="Language code for the template (e.g., 'eng', 'dan').",
    )
    parser.add_argument("-o", "--output", help="Output directory.")
    parser.add_argument("-nv", "--num_versions", type=int, help="Number of versions")
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress all output except errors and warnings.",
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Enable debug output (debug level logging)",
    )
    return parser


def main(
    language: str,
    template_path: Path,
    num_samples: int,
    num_versions: int,
    output: Path | None,
):
    replacements = load_replacements(language)

    logger.info("Starting sample generation...")
    if template_path.is_dir():
        template_files = list(template_path.glob("*.json"))
        if not template_files:
            raise FileNotFoundError(
                f"No JSON files found in directory: {template_path}"
            )
    elif template_path.is_file():
        template_files = [template_path]
    else:
        raise ValueError(
            f"Invalid path: {template_path} - must be a file or directory."
        )

    for template_file in template_files:
        logger.info(f"Processing template file: {template_file}")
        question_template = AnnotatedQuestion.from_json(template_file)

        # First test (saves the versions correctly but still samples multiple times per input)
        # questions = question_template.generate_questions(num_samples * num_versions, language=language, replacements=replacements)
        # for version, question in enumerate(questions, start=1):
        #    version_output = output / f"v{version}"
        #    version_output.mkdir(parents=True, exist_ok=True)
        #    output_file = version_output / f"{template_file.stem}_{version}.json"
        #    question.to_json(output_file)
        #    logger.info(f"Sample saved to: {output_file}")

        # compute all combinations given the possible assignments + filter out invalid combinations
        valid_combinations = question_template._evaluate_constrained_init_lines(
            question_template.constrained_lines,
            question_template.conditions,
            replacements,
        )

        for version in range(1, num_versions + 1):
            version_output = output / f"v{version}"
            version_output.mkdir(parents=True, exist_ok=True)

            for sample in range(num_samples):
                # for each n sample in n versions, choose a random constrained assignment
                assignment = random.choice(valid_combinations)

                # evaluate unconstrined lines (copied directly from _generate_question func)
                unconstrained_assignments = [
                    question_template._evaluate_unconstrained_init_line(
                        line, replacements
                    )
                    for line in question_template.unconstrained_lines
                ]

                # update the selected constrained assignment with unconstrained ones
                for unconstrained_assignment in unconstrained_assignments:
                    assignment.update(unconstrained_assignment)

                # format q+a and create the question (again copy of _generate_question)
                formatted_question = question_template.format_question(
                    assignment, language
                )
                formatted_answer = question_template.format_answer(assignment, language)

                question = Question(
                    formatted_question,
                    formatted_answer,
                    question_template.id_orig,
                    question_template.id_shuffled,
                )

                output_file = version_output / f"{template_file.stem}_{version}.json"
                question.to_json(output_file)
                logger.info(f"Sample saved to: {output_file}")


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    if args.quiet:
        log_level = logging.WARNING
    elif args.debug:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO

    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    main(
        language=args.language,
        template_path=Path(args.template_path),
        num_samples=args.num_samples,
        num_versions=args.num_versions,
        output=Path(args.output) if args.output else None,
    )
