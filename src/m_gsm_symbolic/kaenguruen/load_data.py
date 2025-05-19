from __future__ import annotations

from pathlib import Path
from typing import TypedDict

default_kaenguruen_path = Path(__file__).parent.parent / "data" / "kaenguruen"


class KaenguruenProblem(TypedDict):
    """
    Class representing a math problem from the Kaenguruen dataset.

    Attributes:
        question: The question text.
        options: The answer options.
        answer: The correct answer.
        solution: The solution to the problem.
        percentage_correct: The percentage of correct answers.
    """

    question: str
    options: str
    answer: str
    solution: str
    percentage_correct: float | None
    filepath: str


def _parse_text_file(filepath: str | Path) -> KaenguruenProblem:
    """
    Parse a single text file containing a math problem.

    Args:
        filepath: Absolute path to the text file

    Returns:
        A dictionary with the parsed sections
    """
    filepath = Path(filepath)

    with open(filepath, "r", encoding="utf-8") as file:
        content = file.read()

    # Split the content by "---" separator
    question, options, answer, solution, percent_correct = content.split("---")
    question = question.strip()
    options = options.replace("Valgmuligheder:", "").strip()
    answer = answer.replace("Svar:", "").strip()
    solution = solution.replace("Løsning:", "").strip()
    percentage_correct = (
        percent_correct.replace("Procent rigtige:", "").replace("%", "").strip()
    )

    if percentage_correct:
        percent_correct = float(percentage_correct) / 100
    else:
        percent_correct = None

    # Create a dictionary to hold the parsed data
    problem_data: KaenguruenProblem = {
        "question": question,
        "options": options,
        "answer": answer,
        "solution": solution,
        "percentage_correct": percent_correct,
        "filepath": filepath.as_posix(),
    }
    return problem_data


def load_kaenguruen(
    directory_path: str | Path = default_kaenguruen_path,
) -> list[KaenguruenProblem]:
    """
    Load all problem data from text files in the specified directory.

    Args:
        directory_path: Path to the directory containing the text files

    Returns:
        A list of dictionaries, each containing parsed problem data
    """
    dir_path = Path(directory_path)

    return [_parse_text_file(file_path) for file_path in dir_path.glob("**/*.txt")]
