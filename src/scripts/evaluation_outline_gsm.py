from pathlib import Path

import pandas as pd
from pydantic_ai import Agent
from pydantic_evals import Dataset
from pydantic_evals.evaluators import LLMJudge

from m_gsm_symbolic.load_data import load_gsm_dan

cases = [p.to_case() for p in load_gsm_dan()]

judge_llm_name = "openai:gpt-4o-2024-08-06"
response_model_name = "openai:gpt-4.1-nano-2025-04-14"

agent_evaluated = Agent(response_model_name)

evaluator = LLMJudge(
    rubric="The output should be consist expected output, which is the correct answer to the question.",
    model=judge_llm_name,
    include_input=True,
)

ds = Dataset(
    cases=cases,
    evaluators=[evaluator],
)


async def answer_question(question: str) -> str:
    r = await agent_evaluated.run(question)
    return r.output


report = ds.evaluate_sync(answer_question)

root = Path(__file__).parent.parent.parent
save_path = (
    root
    / "data"
    / "evaluation_results"
    / "gsm_dan"
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
