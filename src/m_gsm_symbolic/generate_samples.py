from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
import itertools
import random
import ast
import argparse
import os
from functools import reduce
from typing import Self
from m_gsm_symbolic.replacements_list import replacements

@dataclass
class Question:
    question: str
    answer: str
    id_orig: int
    id_shuffled: int

    def to_json(self, filepath: Path) -> None:
        with filepath.open("w", encoding = "utf-8") as f:
            json.dump(asdict(self), f, ensure_ascii = False)

@dataclass
class AnnotatedQuestion(Question):
    question_annotated: str
    answer_annotated: str

    @classmethod
    def from_json(cls, filepath: Path) -> Self:
        with filepath.open("r", encoding = "utf-8") as f:
            data = json.load(f)
        return cls(**data)
    
    @property
    def question_template(self) -> str:
        """extract question template from question_annotated"""
        return self.question_annotated.splitlines()[0].strip()

    @property
    def vars(self) -> list[str]:
        """extract variable names from question_annotated"""
        init_block = self.question_annotated.split("#init:")[1].split("#conditions:")[0].strip().splitlines()
        linevars = [extract_vars_from_init_line(line) for line in init_block if "=" in line]
        varsets = [set(var) for var in linevars]
        return reduce(set.union, varsets)

    @property
    def init(self) -> list[str]:
        """extract variable definitions from question_annotated"""
        init_block = self.question_annotated.split("#init:")[1].split("#conditions:")[0].strip().splitlines()
        return [line.strip("- ") for line in init_block]

    @property
    def conditions(self) -> list[str]:
        """extract conditions from question_annotated"""
        condition_block = self.question_annotated.split("#conditions:")[1].split("#answer:")[0].strip().splitlines()
        return [line.strip("- ") for line in condition_block]
    
    @property
    def constrained_vars(self) -> list[str]:
        """extract variable names from conditions"""
        return [var for var in self.vars if is_variable_mentioned(var, self.conditions)]
    
def extract_vars_from_init_line(line: str) -> list[str]: 
    """extract variable names from a line"""
    vars = line.split("=")[0].strip("- ").strip("$").split(",")
    return [var.strip() for var in vars]
    
def is_init_line_constrained(line: str, constrained_vars: list[str]) -> bool:
    """check if a line is constrained"""
    return any(var in extract_vars_from_init_line(line) for var in constrained_vars)

def is_int(val):
    return isinstance(val, int) or (isinstance(val, float) and val.is_integer())

def divides(a, b):
    if not (isinstance(a, (int, float)) and isinstance(b, (int, float))):
        raise TypeError("Both arguments must be int or float.")
    if b == 0: 
        return False
    return a % b == 0

def sample(items, n=1):
    """Sample n items from the list"""
    return random.sample(items, n)

def sample_sequential(items, n):
    """Sample n sequential items from the list"""
    start_idx = random.randint(0, len(items) - 1)
    return [items[(start_idx + i) % len(items)] for i in range(n)]

def is_variable_mentioned(variable_name, text_list):
    """
    Check if a variable is mentioned in any text from a list.
    """
    # Create a regular expression that matches the variable name
    # surrounded by word boundaries to ensure it's a standalone reference
    # \b is a word boundary that matches positions where a word character
    # is not followed or preceded by another word character
    variable_pattern = re.compile(r'\b%s\b' % re.escape(variable_name), re.I)

    for text in text_list:
        if variable_pattern.search(text):
            return True
    
    return False

def range_possibilities(start, end, step=1):
    """Generate a list of numbers from start to end with a given step."""
    if start > end:
        return []
    return list(range(start, end + 1, step))

def sample_possibilities(items, n=1):
    """Sample a list of items."""
    return items

def strip_elements(lst):
    """Strip whitespace from each element in a list"""
    return [elem.strip() for elem in lst]

EVAL_CONTEXT_HELPERS = {
    "is_int": is_int,
    "divides": divides,
    "int": int,
    "float": float,
    "round": round,
    "str": str,
    "len": len,
    "sample": sample,
    "sample_sequential": sample_sequential,
    "list": list,
    "range": range,
}

COMBINATION_HELPERS = {
    "range": range_possibilities,
    "sample": sample_possibilities,
    "list": list,
}

def evaluate_unconstrained_init_line(init_line):
    """ Evaluate a single unconstrained init line and return the assignments."""
    #  If the line is unconstrained, we evaluate it directly since no other variables depend on it.
    print(f"Evaluating unconstrained init line: {init_line}")
    assignments = {}
    var_part, definition_part = init_line.split("=", 1)
    vars = strip_elements(var_part.strip("$").split(","))
    definition_part = definition_part.strip()

    if definition_part.startswith("range(") or definition_part.startswith("list(range("):
        definition_part = "sample(" + definition_part + ")"
    try:
        vals = list(eval(definition_part, {"__builtins__": {}}, EVAL_CONTEXT_HELPERS | replacements))
        print(f"Variables: {vars}, Definition part: {definition_part}, Evaluated values: {vals}")
    except Exception as e:
        print(f"Error evaluating assignment for {var_part}: {definition_part} -> {e}")
        raise e
    if isinstance(vals, list) and len(vals) == len(vars):
        for var, val in zip(vars, vals):
            assignments[var] = val
    else:
        print(f"Warning: {vars} and {vals} are incompatible for line {init_line}.")
    
    return assignments

def evaluate_constrained_init_lines(init_lines, conditions):
    """ Returns a list of valid combinations of values for the constrained init lines."""
    
    possible_assignments = get_all_possible_assignments(init_lines)

    all_combinations = get_all_combinations(possible_assignments)
    print(f"Number of combinations: {len(all_combinations)}")

    return filter_invalid_combinations(all_combinations, conditions)

def get_all_possible_assignments(init_lines):
    possible_assignments = {}
    for line in init_lines:
        var_part, definition_part = line.split("=", 1)
        vars = strip_elements(var_part.strip("$").split(","))
        definition_part = definition_part.strip()
        print(f"Variables: {vars}, Definition part: {definition_part}")
        if len(vars) == 1:
            var_name = vars[0].strip()
            try:
                possible_values = eval(definition_part, {"__builtins__": {}}, COMBINATION_HELPERS | replacements)
                # Save as a list of tuples to make it easier to generate combinations
                possible_assignments[var_name] = list(zip([var_name] * len(possible_values), possible_values))
            except Exception as e:
                print(f"Error evaluating line '{line}': {e}")
                raise e
        else:
            print(f"Warning: Constrained init line '{line}' has more than 1 variable.")

    return possible_assignments

def get_all_combinations(possibilities):
    all_combinations = list(itertools.product(*possibilities.values()))
    combination_dicts = [{k:v for k,v in combo} for combo in all_combinations]
    return combination_dicts

def filter_invalid_combinations(combinations, conditions):
    valid_combinations = []
    for combo in combinations:
        valid = True
        for cond in conditions:
            temp_combo = combo | {k: v[1] for k, v in combo.items() if isinstance(v, tuple)}
            if not eval(cond, {"__builtins__": {}}, EVAL_CONTEXT_HELPERS | temp_combo):
                valid = False
                break

        if valid:
            valid_combinations.append(combo)

    print(f"Number of valid combinations: {len(valid_combinations)}")
    return valid_combinations

def format_question(question_template_text, combination):
    def replace_placeholder(match):
        var_name = match.group(1)
        if var_name in combination:
            value = combination[var_name]
            return str(value[0]) if isinstance(value, tuple) else str(value)
        return match.group(0)
    return re.sub(r"\{(\w+),\s*([^}]+)\}", replace_placeholder, question_template_text)

def format_answer(answer_annotated_template, combination, ):
    eval_env = {**EVAL_CONTEXT_HELPERS, **combination}

    def eval_curly_expr(match):
        expr_str = match.group(1)
        try:
            val = eval(expr_str, {"__builtins__": {}}, eval_env)
            # Convert integer-like floats to integers for display
            if isinstance(val, float) and val.is_integer():
                val = int(val)
            return str(val)
        except Exception:
            return match.group(0) 

    processed_text = re.sub(r"\{([^}]+)\}", eval_curly_expr, answer_annotated_template)
            
    return processed_text

def generate_question(template_filepath):

    question = AnnotatedQuestion.from_json(Path(template_filepath))
    print(f"Vars: {question.vars}")
    print(f"Init lines: {question.init}")
    print(f"Conditions: {question.conditions}")
    print(f"Constrained vars: {question.constrained_vars}")
    unconstrained_lines = [line for line in question.init if not is_init_line_constrained(line, question.constrained_vars)]
    constrained_lines = [line for line in question.init if is_init_line_constrained(line, question.constrained_vars)]
    unconstrained_assignments = [evaluate_unconstrained_init_line(line) for line in unconstrained_lines]
    print(f"Unconstrained assignments: {unconstrained_assignments}")
    constrained_assignments = random.choice(evaluate_constrained_init_lines(constrained_lines, question.conditions))
    print(f"Constrained assignments: {constrained_assignments}")
    all_assignments = constrained_assignments | reduce(lambda x, y: x | y, unconstrained_assignments)
    print(f"All assignments: {all_assignments}")
    formatted_question = format_question(question.question_template, all_assignments)
    print(f"Formatted question: {formatted_question}")
    formatted_answer = format_answer(question.answer_annotated, all_assignments)
    print(f"Formatted answer: {formatted_answer}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate samples from annotated JSON templates.")
    parser.add_argument("template_filepath", help="Path to the JSON template file.")
    parser.add_argument("num_samples", type=int, help="Number of samples to generate.")
    parser.add_argument("-o", "--output", help="Path to the output JSON file (optional). Prints to stdout if not provided.")
    
    args = parser.parse_args()

    generate_question(args.template_filepath)
