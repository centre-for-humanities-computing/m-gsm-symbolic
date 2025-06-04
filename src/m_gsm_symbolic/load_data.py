import json
import logging
from pathlib import Path

from m_gsm_symbolic.gsm_parser import AnnotatedQuestion

logger = logging.getLogger(__name__)

def load_replacements(language):
    root = Path(__file__).parents[2]
    replacement_path = root /"data"/"templates"/f"{language}"/"replacements.json"
    with replacement_path.open("r", encoding="utf-8") as f:
        return json.load(f)

def load_data(language):
    root = Path(__file__).parents[2]
    template_path = root /"data"/"templates"/f"{language}"/"symbolic"
    template_files = list(template_path.glob("*.json"))
    return [AnnotatedQuestion.from_json(template_file) for template_file in template_files]