import json
import re
import random
import pytest
from typing import Self, Any, Callable
import logging

from m_gsm_symbolic.sofie_test_parser import AnnotatedQuestion, Question, eval_expressions, main

logger = logging.getLogger(__name__)

functions = {"range": lambda start, stop, step = 1: range(start, stop, step),
             "sample": lambda items, n = 2: random.sample(items, n),
             "divides": lambda a, b: a % b == 0,}

json_data = r"""
{
"question": "Der er i øjeblikket 16 år mellem Mia og Emma. Hvis Mia, der er yngre end Emma, er 40 år gammel, hvad er gennemsnittet af deres alder?",
"answer": "Hvis Mia er 40 år, er Emma 40+16 = <<40+16=56>>56 år.\nSummen af deres alder er 56+40 = <<56+40=96>>96 år\nGennemsnitsalderen for de to er 96/2 = <<96/2=48>>48 år\n#### 48",
"id_orig": 1277,
"id_shuffled": 31,
"question_annotated": "Der er i øjeblikket {age_diff,16} år mellem {name1,Mia} og {name2,Emma}. Hvis {name1,Mia}, wder er yngre end {name2,Emma}, er {age1,40} år gammel, hvad er gennemsnittet af deres alder?\n\n#init:\n- name1, name2 = sample(names, 2)\n- $age_diff = range(5, 30)\n- $age1 = range(15, 75)\n\n#conditions:\n- is_int((2*age1 + age_diff) / 2)\n\n#answer: (2*age1 + age_diff) // 2",
"answer_annotated": "Hvis {name1} er {age1} år gammel, er {name2} {age1}+{age_diff} = <<{age1}+{age_diff}={age1+age_diff}>>{age1+age_diff} år.\nSummen af deres alder er {age1+age_diff}+{age1} = <<{age1+age_diff}+{age1}={2*age1+age_diff}>>{2*age1+age_diff} år\nGennemsnitsalderen for de to er {2*age1+age_diff}/2 = <<{2*age1+age_diff}/2={(2*age1+age_diff)//2}>>{(2*age1+age_diff)//2} år\n#### {(2*age1+age_diff)//2}"
}"""


@pytest.fixture
def annotated_question():
    question = AnnotatedQuestion.from_json(json_data)
    return question

def test_load_from_json():
    question = AnnotatedQuestion.from_json(json_data)
    data = json.loads(json_data)
    assert question.question == data["question"]
    assert question.answer == data["answer"]


def test_default_values_extraction(annotated_question):
    """test the extraction of default values"""
    assert annotated_question.default_values == {'age_diff': '16', 'name1': 'Mia', 'name2': 'Emma', 'age1': '40'}
    assert "age_diff" in annotated_question.default_values
    assert "age1" in annotated_question.default_values


def test_apply_init_rules_and_replace(annotated_question):
    """test apply_init_rules and replace_values"""
    var = {"names": ["Sofie", "Kenneth"]}
    new_values = annotated_question.apply_init_rules(var)
    q_text, a_text = annotated_question.replace_values(new_values)
    assert "{name1}" not in q_text
    assert "Sofie" in q_text
    assert "Kenneth" in q_text


def test_eval_expressions():
    """test calculations (eval_expressions func)"""
    fixed_var = {"age1": 40, "age_diff": 16}
    expr = ("Hvis {name1} er {age1} år gammel, er {name2} {age1}+{age_diff} = <<{age1}+{age_diff}={age1+age_diff}>>")
    result = eval_expressions(expr, fixed_var)
    assert "40" in result
    assert "16" in result
    assert "56" in result


def test_generate_question(annotated_question):
    """test the generate_question func"""
    replacements = {"names": ["Sofie", "Kenneth"]}
    generated_question = annotated_question.generate_question(replacements)

    assert "Sofie" in generated_question.question
    assert "Kenneth" in generated_question.question

    assert isinstance(generated_question.question, str) and generated_question.question  != ""  
    assert isinstance(generated_question.answer, str) and generated_question.answer != ""  
    assert generated_question.id_orig == annotated_question.id_orig


def test_main():
    json_data = r"""
    {
    "question": "Der er i øjeblikket 16 år mellem Mia og Emma. Hvis Mia, der er yngre end Emma, er 40 år gammel, hvad er gennemsnittet af deres alder?",
    "answer": "Hvis Mia er 40 år, er Emma 40+16 = <<40+16=56>>56 år.\nSummen af deres alder er 56+40 = <<56+40=96>>96 år\nGennemsnitsalderen for de to er 96/2 = <<96/2=48>>48 år\n",
    "id_orig": 1277,
    "id_shuffled": 31,
    "question_annotated": "Der er i øjeblikket {age_diff,16} år mellem {name1,Mia} og {name2,Emma}. Hvis {name1,Mia}, wder er yngre end {name2,Emma}, er {age1,40} år gammel, hvad er gennemsnittet af deres alder?\n\n",
    "answer_annotated": "Hvis {name1} er {age1} år gammel, er {name2} {age1}+{age_diff} = <<{age1}+{age_diff}={age1+age_diff}>>{age1+age_diff} år.\nSummen af deres alder er {age1+age_diff}+{age1} = <<{age1+age_diff}+{age1}={2*age1+age_diff}>>{2*age1+age_diff} år\nGennemsnitsalderen for de to er {2*age1+age_diff}/2 = <<{2*age1+age_diff}/2={(2*age1+age_diff)//2}>>{(2*age1+age_diff)//2} år\n"
    }"""

    main(json_data)

