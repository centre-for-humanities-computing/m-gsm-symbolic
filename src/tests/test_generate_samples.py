import pytest
import json
import m_gsm_symbolic.generate_samples
from m_gsm_symbolic.generate_samples import (
    parse_template,
    parse_init_rules,
    generate_valid_combinations,
    format_question,
    format_answer,
    main_logic
)

# Helper to create a temporary JSON file for testing templates
@pytest.fixture
def temp_json_file(tmp_path):
    def _create_temp_file(content_dict, filename="temp_template.json"):
        file_path = tmp_path / filename
        with open(file_path, 'w') as f:
            json.dump(content_dict, f)
        return str(file_path)
    return _create_temp_file

# --- Test parse_template --- 
@pytest.mark.parametrize("template_content, expected_question_text, expected_init_rules, expected_conditions, expected_answer_formula", [
    (
        {
            "question_annotated": "Q: {var1, val1}? \n\n#init:\n- var1 = range(1,3)\n#conditions:\n- var1 > 0\n#answer: var1 * 2",
            "answer_annotated": "A: {var1*2}\n#### {var1*2}"
        },
        "Q: {var1, val1}?",
        ["var1 = range(1,3)"],
        ["var1 > 0"],
        "var1 * 2"
    ),
    (
        {
            "question_annotated": "Another Q? \n\n#init:\n- x = [1,2]\n- y = sample(names)\n#conditions:\n- x > 0\n#answer: x+y",
            "answer_annotated": "Ans: {x+y}\n#### {x+y}"
        },
        "Another Q?",
        ["x = [1,2]", "y = sample(names)"],
        ["x > 0"],
        "x+y"
    ),
    # Test with missing sections
    (
        {
            "question_annotated": "Only question text.",
            "answer_annotated": "Only answer text."
        },
        "Only question text.",
        [],
        [],
        None
    ),
    (
        {
            "question_annotated": "Q with init only.\n\n#init:\n- v = 10",
            "answer_annotated": "A"
        },
        "Q with init only.",
        ["v = 10"],
        [],
        None
    ),
    (
        {
            "question_annotated": "Q with conditions only.\n\n#conditions:\n- c < 1",
            "answer_annotated": "A"
        },
        "Q with conditions only.",
        [],
        ["c < 1"],
        None
    ),
     (
        {
            "question_annotated": "Q with answer only.\n\n#answer: ans",
            "answer_annotated": "A"
        },
        "Q with answer only.",
        [],
        [],
        "ans"
    ),
    (
        {
            "question_annotated": "Q {a,b} and {c,d}.\n\n#init:\n- a = range(1,2)\n- c = range(3,4)\n#conditions:\n- a > 0\n- c > 2\n#answer: a+c",
            "answer_annotated": "Final: {a+c}\n#### {a+c}"
        },
        "Q {a,b} and {c,d}.",
        ["a = range(1,2)", "c = range(3,4)"],
        ["a > 0", "c > 2"],
        "a+c"
    ),
    # Test with single newline before #init
    (
        {
            "question_annotated": "Q single newline.\n#init:\n- var = 1",
            "answer_annotated": "Ans"
        },
        "Q single newline.",
        ["var = 1"],
        [],
        None
    ),
    # Test with #init at the very beginning
    (
        {
            "question_annotated": "#init:\n- var = 1\n#conditions:\n- var > 0",
            "answer_annotated": "Ans"
        },
        "",
        ["var = 1"],
        ["var > 0"],
        None
    )
])
def test_parse_template(temp_json_file, template_content, expected_question_text, expected_init_rules, expected_conditions, expected_answer_formula):
    file_path = temp_json_file(template_content)
    result = parse_template(file_path)
    assert result["question_template_text"].strip() == expected_question_text.strip()
    assert result["init_rules_str"] == expected_init_rules
    assert result["conditions_str"] == expected_conditions
    assert result["answer_formula_str"] == expected_answer_formula

# --- Test parse_init_rules ---
@pytest.fixture(autouse=True)
def setup_replacements_for_tests(monkeypatch):
    # Provide a minimal, controlled replacements for tests
    mock_replacements_data = {
        "names": ["Alice", "Bob"],
        "numbers_str": ["one", "two"],
        "numbers_val": [1,2]
    }
    # Patch the 'replacements' object in the 'scripts.generate_samples' module
    monkeypatch.setattr(m_gsm_symbolic.generate_samples, "replacements", mock_replacements_data)

@pytest.mark.parametrize("init_rules, expected_vars_info", [
    (
        ["$x = range(1, 4)", "name = sample(names)"],
        {
            "x": {'values': [1, 2, 3], 'display_map': {}, 'is_numeric': True},
            "name": {'values': ["Alice", "Bob"], 'display_map': {}, 'is_numeric': False}
        }
    ),
    (
        ["val = 100", "options = sample([(\"Opt1\", 1), (\"Opt2\", 2)])"],
        {
            "val": {'values': [100], 'display_map': {}, 'is_numeric': False},
            "options": {'values': [1, 2], 'display_map': {1: "Opt1", 2: "Opt2"}, 'is_numeric': False}
        }
    ),
    (
        ["fixed_list = [10, 20, 30]"],
        {
            "fixed_list": {'values': [[10, 20, 30]], 'display_map': {}, 'is_numeric': False}
        }
    ),
    (
        ["str_val = \"hello\""],
        {
            "str_val": {'values': ["hello"], 'display_map': {}, 'is_numeric': False}
        }
    )
])
def test_parse_init_rules(init_rules, expected_vars_info):
    result = parse_init_rules(init_rules)
    assert result == expected_vars_info

# --- Test generate_valid_combinations ---
@pytest.mark.parametrize("parsed_vars, conditions, expected_combinations_count, expected_sample_valid_combo", [
    (
        {
            "x": {'values': [1, 2, 3], 'display_map': {}, 'is_numeric': True},
            "y": {'values': [10, 20], 'display_map': {}, 'is_numeric': True}
        },
        ["x < 3", "y == 10"],
        2, # (1,10), (2,10)
        {"x": 1, "y": 10} # One example of a valid combo
    ),
    (
        {
            "a": {'values': [0, 1], 'display_map': {}, 'is_numeric': True},
            "b": {'values': [0, 1, 2, 3], 'display_map': {}, 'is_numeric': False}
        },
        ["a == 1", "b > 1"],
        2, # (1, 2), (1, 3)
        {"a": 1, "b": 2}
    ),
    (
        {
            "z": {'values': [100, 200, 300], 'display_map': {}, 'is_numeric': True}
        },
        ["z > 500"], # No combination will be valid
        0,
        None
    ),
    (
        {
            "n1": {'values': [1,2,3], 'display_map': {}, 'is_numeric': True},
            "n2": {'values': [4,5,6], 'display_map': {}, 'is_numeric': True}
        },
        ["divides(n2, n1)", "n1 < n2"], # e.g. (1,4), (1,5), (1,6), (2,4), (2,6), (3,6)
        6,
        {"n1": 2, "n2": 4}
    )
])
def test_generate_valid_combinations(parsed_vars, conditions, expected_combinations_count, expected_sample_valid_combo):
    unconstrained_vars, valid_combinations = generate_valid_combinations(parsed_vars, conditions)
    
    assert len(valid_combinations) == expected_combinations_count
    
    if expected_sample_valid_combo:
        # Check if the sample valid combo is present in the generated combinations
        is_sample_present = False
        for combo in valid_combinations:
            match = True
            for key, val in expected_sample_valid_combo.items():
                if key not in combo or combo[key] != val:
                    match = False
                    break
            if match:
                is_sample_present = True
                break
        assert is_sample_present
    elif expected_combinations_count == 0:
        assert not valid_combinations  # Should be an empty list

# --- Test format_question ---
@pytest.mark.parametrize("template_text, combination, parsed_vars_info, expected_question", [
    (
        "Q: {val, 10}, Name: {name, Alice}",
        {"val": 20, "name": "Bob"},
        {
            "val": {'values': [20], 'display_map': {}, 'is_numeric': True},
            "name": {'values': ["Bob"], 'display_map': {}, 'is_numeric': False}
        },
        "Q: 20, Name: Bob"
    ),
    (
        "Item: {item_id, XYZ}, Display: {item_display, Item X}",
        {"item_id": 123, "item_display": "Actual Item 123"},
        {
            "item_id": {'values': [123], 'display_map': {}, 'is_numeric': True},
            "item_display": {'values': ["Actual Item 123"], 'display_map': {123: "Item 123 Display"}, 'is_numeric': False}
        },
        "Item: 123, Display: Actual Item 123"
    ),
    (
        "This is {var_with_display, placeholder}. Value is {num_val, 0}.",
        {"var_with_display": "actual_key", "num_val": 5},
        {
            "var_with_display": {'values': ["actual_key"], 'display_map': {"actual_key": "Displayed Text"}, 'is_numeric': False},
            "num_val": {'values': [5], 'display_map': {}, 'is_numeric': True}
        },
        "This is Displayed Text. Value is 5."
    )
])
def test_format_question(template_text, combination, parsed_vars_info, expected_question):
    assert format_question(template_text, combination, parsed_vars_info) == expected_question

# --- Test format_answer ---
@pytest.mark.parametrize("answer_template, combination, answer_formula, expected_answer", [
    (
        "Result: {x+y}. Calculation: <<{x}+{y}={x+y}>>\n#### {x+y}",
        {"x": 5, "y": 3},
        "x+y",
        "Result: 8. Calculation: <<5+3=8>>\n#### 8"
    ),
    (
        "Value is {val/2}. Int val is {int(val/2)}.\n#### {int(val/2)}",
        {"val": 10},
        "int(val/2)",
        "Value is 5. Int val is 5.\n#### 5"
    ),
    (
        "Text: {name.upper()}\n#### {name.upper()[:3]}",
        {"name": "Alice"},
        "name.upper()[:3]",
        "Text: ALICE\n#### ALI"
    ),
    (
        "Final result is {z*10}\n#### {z*10}",
        {"z": 7},
        None,
        "Final result is 70\n#### 70"
    ),
    (
        "Template says: {a-b}\n#### {a-b} but formula is different",
        {"a": 10, "b": 3},
        "a*b",
        "Template says: 7\n#### 30"
    )
])
def test_format_answer(answer_template, combination, answer_formula, expected_answer):
    assert format_answer(answer_template, combination, answer_formula) == expected_answer


# --- Test main_logic (Integration tests) ---
@pytest.fixture
def sample_template_file(temp_json_file):
    content = {
        "question_annotated": "Q: {name, X} has {count, 0} items. Cost: {price, 0}.\n\n#init:\n- name = sample(names)\n- $count = range(1, 3) \n- $price = range(10,12)\n#conditions:\n- count * price < 25 \n#answer: count * price",
        "answer_annotated": "A: {name} has {count} items. Total: {count*price}.\n#### {count*price}"
    }
    return temp_json_file(content, "main_logic_template.json")

def test_main_logic_generates_output(sample_template_file, tmp_path):
    output_file = tmp_path / "output.json"
    num_samples = 1 

    main_logic(sample_template_file, num_samples, str(output_file))

    assert output_file.exists()
    with open(output_file, 'r') as f:
        output_data = json.load(f)
    
    assert len(output_data) == num_samples
    assert "question" in output_data[0]
    assert "answer" in output_data[0]
    assert "variables" in output_data[0]
    assert output_data[0]["variables"]["name"] in ["Alice", "Bob"]
    assert 1 <= output_data[0]["variables"]["count"] <= 2
    assert 10 <= output_data[0]["variables"]["price"] <= 11
    product = output_data[0]["variables"]["count"] * output_data[0]["variables"]["price"]
    assert product < 25
    assert f"#### {product}" in output_data[0]["answer"]


def test_main_logic_handles_no_valid_combinations(temp_json_file, tmp_path, capsys):
    template_content = {
        "question_annotated": "Q.\n\n#init:\n- $v = range(1, 5)\n#conditions:\n- v > 10\n#answer: v",
        "answer_annotated": "A.\n#### {v}"
    }
    template_file = temp_json_file(template_content, "no_combo_template.json")
    output_file = tmp_path / "no_combo_output.json"

    main_logic(template_file, 1, str(output_file))

    captured = capsys.readouterr()
    assert "No valid combinations found" in captured.out
    assert not output_file.exists()


def test_main_logic_warns_for_too_many_samples(sample_template_file, tmp_path, capsys):
    output_file = tmp_path / "warn_output.json"
    requested_samples = 10
    
    main_logic(sample_template_file, requested_samples, str(output_file))
    
    captured = capsys.readouterr()
    assert "Warning: Requested 10 samples, but only" in captured.out
    assert "unique valid combinations exist" in captured.out
    assert output_file.exists()
    with open(output_file, 'r') as f:
        output_data = json.load(f)
    # Instead of hardcoding the count, assert it's greater than 0 and less than requested
    assert 0 < len(output_data) < requested_samples

