from pathlib import Path

from litellm import completion


def translate_sample(content: str, target_language: str, model: str):
    messages = [
        {
            "content": f"Translate the following content into {target_language}. Do not translate the json format.",
            "role": "system",
        },
        {"content": content, "role": "user"},
    ]
    response = completion(model=model, messages=messages)
    return response["choices"][0]["message"]["content"]


def translate_all(read_dir: Path, save_dir: Path, target_langauge: str, model: str):
    files = read_dir.glob("*.json")

    for sample_path in files:
        write_path = save_dir / sample_path.name
        write_path.mkdir(parents=True, exist_ok=True)

        with sample_path.open("r") as f:
            content = f.read()

        trl_content = translate_sample(content, target_language, model)

        with write_path.open("w") as f:
            f.write(trl_content)

        break


if __name__ == "__main__":
    read_path = Path(__file__) / "data" / "templates" / "eng" / "symbolic"
    write_path = Path(__file__) / "data" / "templates" / "dan" / "symbolic"

    model = "openai/gpt-4o"
    target_language = "Danish"

    translate_all(
        read_dir=read_path, save_dir=write_path, target_langauge="Danish", model=model
    )
