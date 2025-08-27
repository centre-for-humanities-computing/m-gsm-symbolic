from pathlib import Path

import os
import argparse
import re
from dataclasses import dataclass
import pandas as pd
import torch
import random

from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams

from pydantic_evals import Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext

from transformers import AutoTokenizer

from m_gsm_symbolic.kaenguruen.load_data import load_kaenguruen
from m_gsm_symbolic.load_data import load_gsm_dan, load_gsm_eng


if torch.cuda.is_available():
    dev = "cuda"
else:
    dev = "cpu"
device = torch.device(dev)


answer_pattern = r"[^#]{0,300}####\s\d+"


class HuggingFaceAgent:
    def __init__(
        self, model: str, gpu: int, examples: list, answer_pattern: str = answer_pattern
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = LLM(
            model=model, tensor_parallel_size=gpu, max_model_len=8192, max_num_seqs=1
        )
        self.cases = examples
        self.answer_pattern = answer_pattern

    def _build_prompt(self, prompt):
        examples = []
        for case in self.cases:
            example = f"Problem: {case.inputs}\n\nSolution: {case.expected_output}"
            examples.append(example)

        prompt = f"Problem: {prompt}\n\nSolution:"
        examples.append(prompt)
        prompt = "\n\n".join(examples)
        return prompt

    def run(self, prompt: str):
        prompt = self._build_prompt(prompt)
        guided_params = GuidedDecodingParams(regex=self.answer_pattern)
        sampling_params = SamplingParams(guided_decoding=guided_params, max_tokens=500)
        model_output = self.model.generate(prompt, sampling_params=sampling_params)

        return model_output[0].outputs[0].text


@dataclass
class AnswerOnlyMatch(Evaluator):
    def evaluate(self, ctx: EvaluatorContext) -> bool:
        m_pred = re.search(r"####\s*((\d|,)+)", ctx.output)
        m_true = re.search(r"####\s*((\d|,)+)", ctx.expected_output)
        if not m_pred or not m_true:
            return False

        return float(m_pred.group(1).replace(",", "")) == float(
            m_true.group(1).replace(",", "")
        )


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        "-p",
        required=True,
        help="Provide a path to the data folder",
    )
    parser.add_argument(
        "--model",
        "-m",
        required=True,
        choices=[
            "google/gemma-3-1b-pt",
            "google/gemma-3-1b-it",
            "meta-llama/Llama-3.2-1B",
            "meta-llama/Llama-3.2-1B-Instruct",
            "meta-llama/Llama-3.1-8B",
            "meta-llama/Llama-3.1-8B-Instruct",
            "meta-llama/Llama-3.1-70B",
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
    parser.add_argument(
        "--gpu",
        "-g",
        type=int,
        required=True,
        help="Specify number of GPU",
    )
    args = parser.parse_args()
    return args


def main():
    args = parser()

    path = Path(args.path)

    for folder in sorted(path.iterdir()):
        print(f"Processing folder {folder.name}")

        if args.dataset == "kaenguruen":
            cases = [p.to_case() for p in load_kaenguruen(folder)]
        elif args.dataset == "gsm_dan":
            cases = [p.to_case() for p in load_gsm_dan(folder)]
        elif args.dataset == "gsm_eng":
            cases = [p.to_case() for p in load_gsm_eng(folder)]

        random.seed(42)
        random.shuffle(cases)

        response_model_name = args.model
        num_gpu = args.gpu
        agent_evaluated = HuggingFaceAgent(
            response_model_name,
            gpu=num_gpu,
            examples=cases[-3:],
            answer_pattern=answer_pattern,
        )

        ds = Dataset(
            cases=cases[0:97],
            evaluators=[AnswerOnlyMatch()],
        )

        async def answer_question(question: str) -> str:
            r = agent_evaluated.run(question) # noqa: F821
            return r

        report = ds.evaluate_sync(answer_question)

        if "/" in response_model_name:
            response_model_name = response_model_name.split("/")[-1]

        root = Path(__file__).parent.parent.parent

        if not os.path.isdir(
            root
            / "data"
            / "evaluation_results"
            / f"{args.dataset}"
            / f"{response_model_name}"
        ):
            os.makedirs(
                root
                / "data"
                / "evaluation_results"
                / f"{args.dataset}"
                / f"{response_model_name}"
            )

        save_path = (
            root
            / "data"
            / "evaluation_results"
            / f"{args.dataset}"
            / f"{response_model_name}"
            / f"{response_model_name.replace(':', '__')}_{folder.name}.json"
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
                    "correct": case.assertions["AnswerOnlyMatch"].value,
                }
            )

        df = pd.DataFrame(rows)
        df.to_csv(save_path.with_suffix(".csv"), index=False)

        del cases
        del agent_evaluated
        del ds

        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
