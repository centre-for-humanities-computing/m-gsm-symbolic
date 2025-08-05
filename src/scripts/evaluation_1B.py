from pathlib import Path

import argparse
import pandas as pd
import torch
import random

from pydantic_evals import Dataset
from pydantic_evals.evaluators import LLMJudge

from transformers import AutoTokenizer, AutoModelForCausalLM

from m_gsm_symbolic.kaenguruen.load_data import load_kaenguruen
from m_gsm_symbolic.load_data import load_gsm_dan, load_gsm_eng


if torch.cuda.is_available():
    dev = "cuda"
else:
    dev = "cpu"
device = torch.device(dev)


class HuggingFaceAgent:
    def __init__(self, model: str, examples: list = []):
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(model, torch_dtype="auto").to(
            device
        )
        self.examples = examples

    def _build_prompt(self, prompt):
        examples = []
        for case in self.examples:
            example = f"Problem: {case.inputs}\n\nSolution: {case.expected_output}"
            examples.append(example)

        prompt = f"Problem: {prompt}\n\nSolution:"
        examples.append(prompt)
        prompt = "\n\n".join(examples)
        return prompt

    def run(self, prompt: str):
        prompt = self._build_prompt(prompt)
        model_input = self.tokenizer(prompt, return_tensors="pt").to(device)

        model_input.pop("token_type_ids", None)

        input_len = model_input["input_ids"].shape[-1]

        model_output = self.model.generate(**model_input, max_new_tokens=500)
        del model_input
        model_output = model_output[0][input_len:]

        output_decoded = self.tokenizer.decode(model_output, skip_special_tokens=True)
        del model_output

        return output_decoded


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        "-m",
        required=True,
        choices=[
            "google/gemma-3-1b-pt",
            "google/gemma-3-1b-it",
            "meta-llama/Llama-3.2-1B",
            "meta-llama/Llama-3.2-1B-Instruct",
            "allenai/OLMo-1B-hf",
            "PleIAs/Pleias-1.2b-Preview",
        ],
        help="Specify which model to evaluate",
    )
    parser.add_argument(
        "--dataset",
        "-d",
        required=True,
        choices=["kaenguruen", "gsm_dan", "gsm_eng"],
        help="Specify which dataset to evaluate",
    )
    args = parser.parse_args()
    return args


def main():
    args = parser()

    if args.dataset == "kaenguruen":
        cases = [p.to_case() for p in load_kaenguruen()]
    elif args.dataset == "gsm_dan":
        cases = [p.to_case() for p in load_gsm_dan()]
    elif args.dataset == "gsm_eng":
        cases = [p.to_case() for p in load_gsm_eng()]

    random.seed(42)
    random.shuffle(cases)

    judge_llm_name = "openai:gpt-4o-2024-08-06"
    response_model_name = args.model
    agent_evaluated = HuggingFaceAgent(response_model_name, examples=cases[-3:])

    evaluator = LLMJudge(
        rubric="The output should be consist expected output, which is the correct answer to the question.",
        model=judge_llm_name,
        include_input=True,
    )

    ds = Dataset(
        cases=cases[0:97],
        evaluators=[evaluator],
    )

    async def answer_question(question: str) -> str:
        r = agent_evaluated.run(question)
        return r

    report = ds.evaluate_sync(answer_question)

    if "/" in response_model_name:
        response_model_name = response_model_name.split("/")[-1]

    root = Path(__file__).parent.parent.parent
    save_path = (
        root
        / "data"
        / "evaluation_results"
        / args.dataset
        / f"{response_model_name.replace(':', '__')}.json"
    )

    rows = []

    for case in report.cases:
        rows.append(
            {
                "name": case.name,
                "inputs": case.inputs,
                "expected_output": case.expected_output,
                "output": case.output,
                "task_duration": case.task_duration,
                "assertion_value": case.assertions["LLMJudge"].value,
                "assertion_reason": case.assertions["LLMJudge"].reason,
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(save_path.with_suffix(".csv"), index=False)


if __name__ == "__main__":
    main()
