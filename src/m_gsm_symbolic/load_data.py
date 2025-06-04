import json
import logging
from pathlib import Path

from m_gsm_symbolic.gsm_parser import AnnotatedQuestion

logger = logging.getLogger(__name__)

def load_replacements(language):
    root = Path(__file__).parents[3]
    replacement_path = root / "data"/"templates"/f"{language}"/"replacements.json"
     with replacement_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    else:
        raise ValueError(f"No replacements file found for language: {language}")

def load_data(language):
    if language == "dan":
        template_files = list(Path("data/templates/dan/symbolic").glob("*.json"))
        return [AnnotatedQuestion.from_json(template_file) for template_file in template_files]
    elif language == "eng":
        template_files = list(Path("data/templates/eng/symbolic").glob("*.json"))
        return [AnnotatedQuestion.from_json(template_file) for template_file in template_files]
    else:
        raise ValueError(f"No template files found for language: {language}")