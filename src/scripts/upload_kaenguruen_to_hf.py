"""
Upload the Kaenguruen dataset to HuggingFace in a format compatible with inspect-ai.

Usage:
    python src/scripts/upload_kaenguruen_to_hf.py --repo <hf-username>/<repo-name>

The resulting dataset can be loaded with inspect-ai like:
    from inspect_ai.dataset import hf_dataset, FieldSpec

    dataset = hf_dataset(
        "<hf-username>/<repo-name>",
        split="train",
        sample_fields=FieldSpec(
            input="input",
            target="target",
            choices="choices",
            id="id",
            metadata=["solution", "percentage_correct", "year", "grade"],
        ),
    )
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from datasets import Dataset, DatasetDict
from huggingface_hub import DatasetCard


DATA_PATH = Path(__file__).parents[2] / "data" / "kaenguruen"
LETTER_START_PATTERN = re.compile(r"^([A-E])\)\s*")


def _parse_options(options_text: str) -> dict[str, str]:
    """Parse options block into {letter: text}, supporting multi-line option content."""
    letter_to_lines: dict[str, list[str]] = {}
    current_letter: str | None = None

    for line in options_text.splitlines():
        m = LETTER_START_PATTERN.match(line.strip())
        if m:
            current_letter = m.group(1)
            rest = LETTER_START_PATTERN.sub("", line.strip()).strip()
            letter_to_lines[current_letter] = [rest] if rest else []
        elif current_letter is not None:
            letter_to_lines[current_letter].append(line.strip())

    return {letter: "\n".join(lines).strip() for letter, lines in letter_to_lines.items()}


def parse_txt_file(filepath: Path) -> dict | None:
    content = filepath.read_text(encoding="utf-8")

    segments: list[str] = []
    segment: list[str] = []
    for line in content.splitlines():
        if line.strip() == "---":
            if segment:
                segments.append("\n".join(segment))
                segment = []
        else:
            segment.append(line)
    if segment:
        segments.append("\n".join(segment))

    if len(segments) != 5:
        print(f"Skipping {filepath}: unexpected number of segments ({len(segments)})")
        return None

    raw_question, raw_options, raw_answer, raw_solution, raw_percent = segments

    question = raw_question.strip()
    options_text = raw_options.replace("Valgmuligheder:", "").strip()
    answer_text = raw_answer.replace("Svar:", "").strip()
    solution = raw_solution.replace("Løsning:", "").strip() or None
    percent_str = raw_percent.replace("Procent rigtige:", "").replace("%", "").strip()
    percentage_correct = float(percent_str) / 100 if percent_str else None

    letter_to_text = _parse_options(options_text)
    if not letter_to_text:
        print(f"Skipping {filepath}: could not parse choices")
        return None

    sorted_letters = sorted(letter_to_text)
    choices = [letter_to_text[k] for k in sorted_letters]

    # Find the target by matching answer text to a choice
    target: str | None = None
    for letter in sorted_letters:
        if letter_to_text[letter].strip() == answer_text.strip():
            target = letter_to_text[letter]
            break

    if target is None:
        print(
            f"Warning {filepath}: answer '{answer_text}' not matched to any choice; "
            f"choices were {letter_to_text}"
        )
        return None

    # Extract year and grade from folder name, e.g. "2024_8-9-klasse"
    folder = filepath.parent.name  # e.g. "2024_8-9-klasse"
    parts = folder.split("_", 1)
    year = int(parts[0]) if parts[0].isdigit() else None
    grade = parts[1].replace("_klasse", "-klasse") if len(parts) > 1 else folder

    problem_id = f"{folder}/{filepath.stem}"

    return {
        "id": problem_id,
        "input": question,
        "choices": choices,
        "target": target,
        "solution": solution,
        "percentage_correct": percentage_correct,
        "year": year,
        "grade": grade,
    }


def build_dataset(data_path: Path = DATA_PATH) -> Dataset:
    records: list[dict] = []
    for txt_file in sorted(data_path.glob("**/*.txt")):
        record = parse_txt_file(txt_file)
        if record is not None:
            records.append(record)

    print(f"Parsed {len(records)} problems")
    return Dataset.from_list(records)


DATASET_CARD = """\
---
language:
- da
license: other
task_categories:
- multiple-choice
pretty_name: Kænguruen Danish Math Competition
size_categories:
- n<1K
extra_gated_prompt: >-
  To access this dataset, please fill in the form below. By submitting, you confirm
  that you will not use this dataset to train AI models.
extra_gated_fields:
  Email: text
  Organization: text
  I agree not to use this dataset for training AI models: checkbox
---

# Kænguruen Danish Math Competition

A dataset of multiple-choice math problems from the Danish Kangaroo math competition
(*Matematikkens Kænguru*), a popular international mathematics contest held annually in
Denmark for students in grades 4–9.

## Dataset description

Kænguruen originates from France (1991) and is now held in 100+ countries with around
6 million participants per year. In Denmark it is organized by
[Danmarks Matematiklærerforening](https://kaenguruen.dk/) and takes place on the third
Thursday of March. Students have 60 minutes to answer 18–24 multiple-choice questions
designed to be "small, different and challenging."

This dataset contains 106 problems drawn from the 2020–2024 competitions across three
grade levels. All problems are in Danish.

## Dataset structure

| Field | Type | Description |
|---|---|---|
| `id` | string | Unique identifier: `{{year}}_{{grade}}/{{problem_number}}` |
| `input` | string | The problem statement |
| `choices` | list[string] | The five answer options (A–E) |
| `target` | string | The correct answer (full text, matching one entry in `choices`) |
| `solution` | string or null | Worked solution, if available |
| `percentage_correct` | float or null | Share of students who answered correctly, if available |
| `year` | int | Competition year (2020–2024) |
| `grade` | string | Grade level: `4-5-klasse`, `6-7-klasse`, or `8-9-klasse` |

## Example

```json
{{
  "id": "2020_6-7-klasse/21",
  "input": "Et 3-cifret tal kaldes spektakulært, hvis dets midterste ciffer er større end summen\\naf første og sidste ciffer. Fx er 360, 361 og 362 spektakulære tal og de kommer også\\nlige efter hinanden i rækkefølge.\\n\\nHvad er det største antal spektakulære 3-cifrede tal, der kommer lige efter hinanden\\ni rækkefølge?",
  "choices": ["5", "6", "7", "8", "9"],
  "target": "8",
  "solution": "9 er det største mulige midterste tal og da tallet skal være 3 cifferet må det første tal være 1, dermed har vi:\\n190, 191, 192, 193, 194, 195, 196, 197\\n198 er ikke spektakulært da 1+8=9.\\nSå svaret er 8.",
  "percentage_correct": null,
  "year": 2020,
  "grade": "6-7-klasse"
}}
```

## Loading the dataset

### With 🤗 datasets

```python
from datasets import load_dataset

dataset = load_dataset("{repo}", split="test")
```

### With inspect-ai

```python
from inspect_ai.dataset import hf_dataset, FieldSpec

dataset = hf_dataset(
    "{repo}",
    split="test",
    sample_fields=FieldSpec(
        input="input",
        target="target",
        choices="choices",
        id="id",
        metadata=["solution", "percentage_correct", "year", "grade"],
    ),
)
```

## Curation

This dataset was curated by Kenneth Enevoldsen, Sofie Mosegaard, Nicolas Legrand, and
Simon Enni. The source data files are available in the original repository:
[centre-for-humanities-computing/m-gsm-symbolic](https://github.com/centre-for-humanities-computing/m-gsm-symbolic/tree/0783195d51f9ee405f61bee7a3e7af1734761d05/data/kaenguruen).
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload Kaenguruen dataset to HuggingFace")
    parser.add_argument(
        "--repo",
        required=True,
        help="HuggingFace repository in the form <username>/<repo-name>",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        default=False,
        help="Make the dataset repository private",
    )
    args = parser.parse_args()

    dataset = build_dataset()

    dataset_dict = DatasetDict({"test": dataset})

    print(f"Pushing to HuggingFace: {args.repo}")
    dataset_dict.push_to_hub(args.repo, private=args.private)

    card = DatasetCard(DATASET_CARD.format(repo=args.repo))
    card.push_to_hub(args.repo)
    print("Done!")
    print()
    print("Load with inspect-ai:")
    print(
        f"""
from inspect_ai.dataset import hf_dataset, FieldSpec

dataset = hf_dataset(
    "{args.repo}",
    split="test",
    sample_fields=FieldSpec(
        input="input",
        target="target",
        choices="choices",
        id="id",
        metadata=["solution", "percentage_correct", "year", "grade"],
    ),
)
"""
    )


if __name__ == "__main__":
    main()
