import json
import random
import re
from dataclasses import dataclass
from typing import Self

constants = {
    "names": [
        "Alice",
        "Bob",
        "Charlie",
        "David",
        "Eve",
        "Frank",
        "Grace",
        "Helen",
        "Ivy",
        "Jack",
        "Kate",
        "Liam",
        "Mia",
        "Noah",
        "Olivia",
        "Peter",
        "Quinn",
        "Rose",
        "Sam",
        "Tina",
        "Uma",
        "Vince",
        "Wendy",
        "Xander",
        "Yara",
        "Zane",
    ],
}


functions = {
    "range": lambda start, stop, step: range(start, stop, step),
    "sample": lambda items: random.choice(items),
    "divides": lambda a, b: a % b == 0,
}


@dataclass
class Question:
    question: str
    answer: str
    id_orig: int
    id_shuffled: int
    question_annotated: str
    answer_annotated: str

    @classmethod
    def from_json(cls, json_data: str) -> Self:
        data = json.loads(json_data)
        return cls(**data)

    @property
    def init(self) -> str:
        """extract init section from question_annotated"""
        return self.question_annotated.split("#init:")[1].split("#conditions:")[0]

    @property
    def default_values(self) -> dict[str, str]:
        """extract default values from the annotated question"""
        default_values = {}

        # find {var_name, value} pairs
        regex = r"\{(\w+),(\w+)\}"
        matches = re.findall(regex, self.question_annotated)
        for match in matches:
            var_name, value = match
            default_values[var_name] = value

        return default_values

    @staticmethod
    def remove_calculations(text: str) -> str:
        """remove calculations from the text"""
        return re.sub(r"<<.*?>>", "", text)


if __name__ == "__main__":
    # Test JSON string
    json_data = """
    {
    "question": "Benny saw a 10-foot shark with 2 6-inch remoras attached to it. What percentage of the shark's body length is the combined length of the remoras?",
    "answer": "First, find the combined length of the remoras in inches: 6 inches/remora * 2 remoras = <<6*2=12>>12 inches\nThen divide that number by 12 to convert it to feet: 12 inches / 12 inches/foot = <<1=1>>1 foot\nThen divide the combined remora length in feet by the shark's length and multiply by 100% to express the answer as a percentage: 1 foot / 10 feet * 100% = 10%\n#### 10",
    "id_orig": 473,
    "id_shuffled": 0,
    "question_annotated": "{n, Benny} saw a {x,10}-foot {big_fish,shark} with {k,2} {y,6}-inch remoras attached to it. What percentage of the {big_fish,shark}'s body length is the combined length of the remoras?\n\n#init:\n- n = sample(names)\n- big_fish = sample([\"dolphin\", \"whale\", \"shark\"])\n- $x = range(10, 500, 10)\n- $k = range(2, 10)\n- $y = range(2, 100)\n\n#conditions:\n- k * y < x * 12\n- divides(k*y,12)\n- divides(x * 12, k*y)\n- divides(100, x * 12 / (k*y))\n\n#answer: int((k * y)/ (x * 12) * 100)",
    "answer_annotated": "First, find the combined length of the remoras in inches: {y} inches/remora * {k} remoras = <<{y}*{k}={k*y}>>{k*y} inches.\nThen divide that number by 12 to convert it to feet: {k*y} inches / 12 inches/foot = <<{k*y}/12>>{k*y//12} foot.\nThen divide the combined remora length in feet by the {big_fish}'s length and multiply by 100% to express the answer as a percentage: {k*y//12} foot / {x} feet * 100% = {int(k*y/(12*x)*100)}%\n\n#### {int(k*y/(12*x)*100)}"
    }
    """

    data = json.loads(json_data)

    question = Question.from_json(data)
    assert question.default_values == {"n": "Benny", "x": "10", "k": "2", "y": "6"}
    assert (
        question.remove_calculations(question.question)
        == "Benny saw a 10-foot shark with 2 6-inch remoras attached to it. What percentage of the shark's body length is the combined length of the remoras?"
    )
    assert question.question == question.generate_question(default=True)
    assert question.answer == question.generate_answer(default=True)

