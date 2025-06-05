import argparse
import logging
from pathlib import Path

from m_gsm_symbolic.gsm_parser import AnnotatedQuestion
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


def main(language: str, template_path: Path, num_samples: int, output: Path | None):
    replacements = load_replacements(language)

    logger.info("Starting sample generation...")
    # Check if the template path is a directory or a file
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
        questions = question_template.generate_questions(
            num_samples, language=language, replacements=replacements
        )
        if output:
            for i, question in enumerate(questions):
                output_dir = output
                output_dir.mkdir(parents=True, exist_ok=True)
                output_file = output_dir / f"{template_file.stem}_{i + 1}.json"
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
        output=Path(args.output) if args.output else None,
    )
