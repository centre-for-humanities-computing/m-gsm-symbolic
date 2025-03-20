# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "litellm>=1.61.16",
# ]
# ///

import logging
from pathlib import Path
from typing import cast

from litellm import completion
from litellm.types.utils import ModelResponse

logger = logging.getLogger(__name__)


def translate_sample(content: str, target_language: str, model: str):
    messages = [
        {
            "content": f"Translate the following content into {target_language}. Do not translate the json format and do not wrap the package in a code block.",
            "role": "system",
        },
        {"content": content, "role": "user"},
    ]
    response = completion(model=model, messages=messages)
    response = cast(ModelResponse, response)

    return response["choices"][0]["message"]["content"]


def translate_all(read_dir: Path, save_dir: Path, target_langauge: str, model: str):
    files = read_dir.glob("*.json")
    files = list(files)

    logger.info(f"Translating {len(files)} samples")

    for sample_path in files:
        write_path = save_dir / sample_path.name
        write_path.parent.mkdir(parents=True, exist_ok=True)

        if write_path.exists():
            logger.info(f"Skipping {sample_path.name}")
            continue

        with sample_path.open("r") as f:
            content = f.read()

        trl_content = translate_sample(content, target_language, model)

        with write_path.open("w") as f:
            f.write(trl_content)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    repo_path = Path(__file__).parent.parent
    read_path = repo_path / "data" / "templates" / "eng" / "symbolic"
    write_path = repo_path / "data" / "templates" / "dan" / "symbolic-translated"

    model = "openai/gpt-4o"
    target_language = "Danish"

    translate_all(
        read_dir=read_path, save_dir=write_path, target_langauge="Danish", model=model
    )
