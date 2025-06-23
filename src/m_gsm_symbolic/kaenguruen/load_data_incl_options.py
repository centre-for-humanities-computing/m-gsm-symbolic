from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel
from pydantic_evals import Case

default_kaenguruen_path = Path(__file__).parents[3] / "data" / "kaenguruen"


class KaenguruenProblem(BaseModel):
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
    solution: str | None
    percentage_correct: float | None
    filepath: Path

    @property
    def problem_id(self) -> str:
        """
        Generate a unique identifier for the problem based on its filepath.

        Returns:
            A string representing the problem ID.
        """
        return f"{self.filepath.parent.name}/{self.filepath.name}"

    def to_case(self) -> Case:
        """
        Convert the KaenguruenProblem instance to a pydantic_evals Case.

        Returns:
            A Case object with the problem data.
        """
        
        return Case(
            name=self.problem_id,
            inputs=f"{self.question}\n\n{self.options}",
            expected_output=self.answer,
            metadata={
                "solution": self.solution,
                "percentage_correct": self.percentage_correct,
                "filepath": self.filepath.as_posix(),
            },
        )


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
    segments = []
    segment = []
    for line in content.splitlines():
        if line.strip() == "---":
            if segment:
                segments.append("\n".join(segment))
                segment = []
        else:
            segment.append(line)
    if segment:
        segments.append("\n".join(segment))

    question, options, answer, solution, percent_correct = segments
    question = question.strip()
    options = options.replace("Valgmuligheder:", "").strip()
    answer = answer.replace("Svar:", "").strip()
    solution = solution.replace("Løsning:", "").strip()
    percentage_correct = (
        percent_correct.replace("Procent rigtige:", "").replace("%", "").strip()
    )

    percent_correct = float(percentage_correct) / 100 if percentage_correct else None

    # Create a dictionary to hold the parsed data
    problem_data = KaenguruenProblem(
        question=question,
        options=options,
        answer=answer,
        solution=solution if solution else None,
        percentage_correct=percent_correct,
        filepath=filepath,
    )
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
