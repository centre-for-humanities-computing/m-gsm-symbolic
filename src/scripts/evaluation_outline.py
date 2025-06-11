"""
based on this tutorial:
https://ai.pydantic.dev/evals/#installation

"""

from pathlib import Path

import pandas as pd
from pydantic_ai import Agent
from pydantic_evals import Dataset
from pydantic_evals.evaluators import LLMJudge

from m_gsm_symbolic.kaenguruen.load_data import load_kaenguruen

cases = [p.to_case() for p in load_kaenguruen()]


judge_llm_name = "openai:gpt-4o-2024-08-06"
response_model_name = "openai:gpt-4.1-nano-2025-04-14"

agent_evaluated = Agent(response_model_name)


evaluator = LLMJudge(
    rubric="The output should be consist expected output, which is the correct answer to the question.",
    model=judge_llm_name,
    include_input=True,
    include_expected_output=True,
)
ds = Dataset(
    cases=cases,
    evaluators=[evaluator],
)


async def answer_question(question: str) -> str:
    r = await agent_evaluated.run(question)
    return r.output


root = Path(__file__).parent.parent
save_path = (
    root
    / "data"
    / "evaluation_results"
    / "kaenguruen"
    / f"{response_model_name.replace(':', '__')}.json"
)

report = ds.evaluate_sync(answer_question)
save_path.parent.mkdir(parents=True, exist_ok=True)
with save_path.open("w") as f:
    f.write(report.model_dump_json(indent=2))


# tried reloading the report, but that doesn't work. Added an issue here: https://github.com/pydantic/pydantic-ai/issues/1953
# from pydantic_evals.reporting import EvaluationReport

# with save_path.open("r") as f:
#     report = EvaluationReport.model_validate_json(f.read())
# report = EvaluationReport.model_validate_json(report.model_dump_json(indent=2))


# create table from report
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

# save to csv
df.to_csv(save_path.with_suffix(".csv"), index=False)
