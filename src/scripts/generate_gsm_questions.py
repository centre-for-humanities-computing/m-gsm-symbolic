import argparse
import logging
from pathlib import Path

from m_gsm_symbolic.gsm_parser import AnnotatedQuestion
from m_gsm_symbolic.load_data import load_replacements

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate samples from annotated JSON templates.")
    parser.add_argument("template_path", help="Path to the JSON template file(s).")
    parser.add_argument("num_samples", type=int, help="Number of samples to generate for each template.")
    parser.add_argument("language", choices=["dan", "eng"], help="Language code for the template (e.g., 'eng', 'dan').")
    parser.add_argument("-o", "--output", help="Output directory.")
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress all output except errors and warnings.")
    parser.add_argument("-d", "--debug", action="store_true", help="Enable debug output (debug level logging)")
    
    args = parser.parse_args()

    if args.quiet:
        log_level = logging.WARNING
    elif args.debug:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO

    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    replacements = load_replacements(args.language)
    
    logger.info("Starting sample generation...")
    # Check if the template path is a directory or a file
    template_path = Path(args.template_path)
    if template_path.is_dir():
        template_files = list(template_path.glob("*.json"))
        if not template_files:
            logger.error(f"No JSON files found in directory: {template_path}")
            exit(1)
    elif template_path.is_file():
        template_files = [template_path]
    else:
        logger.error(f"Invalid path: {template_path}")
        exit(1)

    for template_file in template_files:
        logger.info(f"Processing template file: {template_file}")
        question_template = AnnotatedQuestion.from_json(template_file)
        questions = question_template.generate_questions(args.num_samples, replacements)
        if args.output:
            for i, question in enumerate(questions):
                output_dir = Path(args.output)
                output_dir.mkdir(parents=True, exist_ok=True)
                output_file = output_dir / f"{template_file.stem}_{i + 1}.json"
                question.to_json(output_file)
                logger.info(f"Sample saved to: {output_file}")