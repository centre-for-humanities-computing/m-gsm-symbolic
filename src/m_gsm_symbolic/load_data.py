import json
import logging
from pathlib import Path

from pydantic import BaseModel
from pydantic_evals import Case

from m_gsm_symbolic.gsm_parser import AnnotatedQuestion

default_gsm_dan_path = (
    Path(__file__).parents[2] / "data" / "templates" / "dan" / "symbolic"
)

logger = logging.getLogger(__name__)


def load_replacements(language):
    root = Path(__file__).parents[2]
    replacement_path = root / "data" / "templates" / f"{language}" / "replacements.json"
    with replacement_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_data(language):
    root = Path(__file__).parents[2]
    template_path = root / "data" / "templates" / f"{language}" / "symbolic"
    template_files = list(template_path.glob("*.json"))
    return [
        AnnotatedQuestion.from_json(template_file) for template_file in template_files
    ]


class GSMProblem(BaseModel):
    question: str
    answer: str
    id_orig: int
    filepath: Path

    def to_case(self) -> Case:
        return Case(
            name=str(self.id_orig),
            inputs=self.question,
            expected_output=self.answer,
            metadata={
                "filepath": self.filepath.as_posix(),
            },
        )


def _parse_json_file(filepath: str | Path) -> GSMProblem:
    filepath = Path(filepath)
    with open(filepath, "r", encoding="utf-8") as file:
        content = json.load(file)

    question = content["question"]
    answer = content["answer"]
    id_orig = content["id_orig"]
    filepath = filepath

    problem_data = GSMProblem(
        question=question, answer=answer, id_orig=id_orig, filepath=filepath
    )
    return problem_data


def load_gsm_dan(
    directory_path: str | Path = default_gsm_dan_path,
) -> list[GSMProblem]:
    dir_path = Path(directory_path)
    json_files = list(dir_path.glob("*.json"))

    return [_parse_json_file(f) for f in json_files]
