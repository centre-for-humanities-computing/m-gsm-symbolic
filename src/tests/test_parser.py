import json
import pytest
from m_gsm_symbolic.parser import Question

json_data = r"""
{
"question": "Benny saw a 10-foot shark with 2 6-inch remoras attached to it. What percentage of the shark's body length is the combined length of the remoras?",
"answer": "First, find the combined length of the remoras in inches: 6 inches/remora * 2 remoras = <<6*2=12>>12 inches\nThen divide that number by 12 to convert it to feet: 12 inches / 12 inches/foot = <<1=1>>1 foot\nThen divide the combined remora length in feet by the shark's length and multiply by 100% to express the answer as a percentage: 1 foot / 10 feet * 100% = 10%\n#### 10",
"id_orig": 473,
"id_shuffled": 0,
"question_annotated": "{n, Benny} saw a {x,10}-foot {big_fish,shark} with {k,2} {y,6}-inch remoras attached to it. What percentage of the {big_fish,shark}'s body length is the combined length of the remoras?\n\n#init:\n- n = sample(names)\n- big_fish = sample([\"dolphin\", \"whale\", \"shark\"])\n- $x = range(10, 500, 10)\n- $k = range(2, 10)\n- $y = range(2, 100)\n\n#conditions:\n- k * y < x * 12\n- divides(k*y,12)\n- divides(x * 12, k*y)\n- divides(100, x * 12 / (k*y))\n\n#answer: int((k * y)/ (x * 12) * 100)",
"answer_annotated": "First, find the combined length of the remoras in inches: {y} inches/remora * {k} remoras = <<{y}*{k}={k*y}>>{k*y} inches.\nThen divide that number by 12 to convert it to feet: {k*y} inches / 12 inches/foot = <<{k*y}/12>>{k*y//12} foot.\nThen divide the combined remora length in feet by the {big_fish}'s length and multiply by 100% to express the answer as a percentage: {k*y//12} foot / {x} feet * 100% = {int(k*y/(12*x)*100)}%\n\n#### {int(k*y/(12*x)*100)}"
}
"""

@pytest.fixture
def question():
    question = Question.from_json(json.loads(json_data))
    return question

def test_load_from_json():
    data = json.loads(json_data)
    question = Question.from_json(json_data)

    assert question.question == data["question"]
    assert question.answer == data["answer"]

def test_parse_with_default_values(question: Question):
    assert question.default_values == {"n": "Benny", "x": "10", "k": "2", "y": "6"}

    assert question.question == question.generate_question(default=True)
    assert question.answer == question.generate_answer(default=True)

def test_parse_with_custom_values(question: Question):
    values = {"n": "Kenneth", "x": "12", "k": "3", "y": "9"}

    correct_answer_string = "First, find the combined length of the remoras in inches: 9 inches/remora * 3 remoras = <<9*3=27>>27 inches.\nThen divide that number by 12 to convert it to feet: 27 inches / 12 inches/foot = <<2=2>>2 foot.\nThen divide the combined remora length in feet by the shark's length and multiply by 100% to express the answer as a percentage: 2 foot / 12 feet * 100% = 16.666666666666664%\n\n#### 16.666666666666664"
    new_answer = question.generate_answer(values=values)
    assert new_answer == correct_answer_string


