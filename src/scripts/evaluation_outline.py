"""
based on this tutorial:
https://ai.pydantic.dev/evals/#installation

"""

from importlib.metadata import version

from pydantic_ai import Agent
from pydantic_evals import Dataset
from pydantic_evals.evaluators import LLMJudge

from m_gsm_symbolic.kaenguruen.load_data import load_kaenguruen

print(version("pydantic_evals"))

cases = [p.to_case() for p in load_kaenguruen()]


agent_evaluated = Agent("openai:gpt-4.1-nano-2025-04-14")
# response = agent.run_sync("What is 2+2")
# print(reponse.output)
# # '2 + 2 equals 4.'

evaluator = LLMJudge(
    rubric="The output should be consist expected output, which is the correct answer to the question.",
    model="openai:gpt-4o",
    include_input=True,
    include_expected_output=True,
)
ds = Dataset(
    cases=cases[:1],  # simple example with just one case # TODO: remove
    evaluators=[evaluator],
)


# async def answer_question(question: str) -> str:
#     r = await agent_evaluated.run(question)
#     r = await answer_extraction.run(question)
#     return r.output


# dummy
async def answer_question(question: str) -> str:
    return "I don't know maybe 42?"


report = ds.evaluate_sync(answer_question)
report.print(include_input=True, include_output=True, include_durations=True)

case_rep = report.cases[0]
case_rep.output
case_rep.expected_output
case_rep.assertions

# Problem:
# Using the gpt4o model, to evaluate the answer of the gpt-4.1-nano model leads to the answer being correct (even if it is not).
# I suspect that this is a two way problem:
# the gpt4o is biased towards the answer of the gpt-4.1-nano model, and
# the llm judge is badly prompted - we could likely to better with final answer extraction and then comparing if the two outputs are equal using an LLM judge
