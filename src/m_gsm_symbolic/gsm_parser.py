from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
import itertools
import random
import logging
from functools import reduce
from typing import Self
import numpy as np
from fractions import Fraction

logger = logging.getLogger(__name__)

def is_int(value):
    return isinstance(value, int) or (isinstance(value, float) and value.is_integer())

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

def np_arange_sample(start, end, step=1):
    """Sample an item from a numpy arange"""
    if start > end:
        return []
    return random.choice(np.arange(start, end, step).tolist())

def frac_format(value):
    """Format a value as a fraction if it is a float, otherwise return as is."""
    if isinstance(value, float):
        # Convert float to Fraction
        frac = Fraction(value).limit_denominator()
        return f"{frac.numerator}/{frac.denominator}" if frac.denominator != 1 else str(frac.numerator)
    return str(value)

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
    """Return possibilities for given range statement."""
    if start > end:
        return []
    return list(range(start, end + 1, step))

def range_possibilities_str(start, end, step, numbers):
    """Return possibilities for given range statement."""
    possible_numbers = range_possibilities(start, end, step)
    possible_number_names = [numbers[i-1] for i in possible_numbers]
    if not possible_numbers:
        return []
    return list(zip(possible_numbers, possible_number_names))

def np_arange_possibilities(start, end, step=1):
    """Return possibilities for given numpy arange statement."""
    if start > end:
        return []
    return np.arange(start, end, step).tolist()

def sample_possibilities(items, n=1):
    """Return possibilities for given sample statement."""
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
    "np.arange": np_arange_sample,
    "Fraction": frac_format,
}

COMBINATION_HELPERS = {
    "range": range_possibilities,
    "range_str": range_possibilities_str,
    "np.arange": np_arange_possibilities, 
    "sample": sample_possibilities,
    "list": list,
}

def try_parse_float(value):
    """Try to parse a value as float, return None if it fails."""
    if not isinstance(value, str):
        return value
    try:
        return float(value)
    except ValueError:
        return value
    
def try_parse_fraction(value):
    """Try to parse a value as a fraction, return None if it fails."""
    if not isinstance(value, str):
        return value
    if '/' in value:
        try:
            num, denom = value.split('/')
            return Fraction(int(num), int(denom))
        except ValueError:
            return value
    return value

def capitalize_sentences(text):
    """Capitalize the first letter of each sentence using regex."""
    import re
    
    # Capitalize first letter of text
    text = text[0].upper() + text[1:] if text else text
    
    # Capitalize letters after sentence-ending punctuation
    text = re.sub(r'([.!?]+\s*)([a-z])', 
                  lambda m: m.group(1) + m.group(2).upper(), 
                  text)
    
    return text

def format_numbers_by_language(text, language):
    import re
    def format_number(match):
        number_str = match.group(0)
        
        if '.' in number_str:
            integer_part, decimal_part = number_str.split('.')
            number = int(integer_part)
            formatted_int = f"{number:,}" if number >= 10000 else str(number)
            
            if language == "dan":
                return formatted_int.replace(",", ".") + "," + decimal_part
            else:
                return formatted_int + "." + decimal_part
        else:
            number = int(number_str)
            if number >= 10000:
                formatted = f"{number:,}"
                return formatted.replace(",", ".") if language == "dan" else formatted
            else:
                return number_str
    
    return re.sub(r'\b\d+(?:\.\d+)\b|\b\d{5,}\b', format_number, text)

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
class AnnotatedQuestion:
    question: str
    answer: str
    id_orig: int
    id_shuffled: int
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
    def variables(self) -> list[str]:
        """extract variable names from question_annotated"""
        init_block = self.question_annotated.split("#init:")[1].split("#conditions:")[0].strip().splitlines()
        variables_per_line = [self._extract_variables_from_init_line(line) for line in init_block if "=" in line]
        variable_sets = [set(v) for v in variables_per_line]
        return reduce(set.union, variable_sets)

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
    def constrained_variables(self) -> list[str]:
        """extract variable names from conditions"""
        return [v for v in self.variables if is_variable_mentioned(v, self.conditions)]
    
    @property
    def unconstrained_lines(self) -> list[str]:
        """extract unconstrained lines from question_annotated"""
        return [line for line in self.init if not self._is_init_line_constrained(line, self.constrained_variables)]
    
    @property
    def constrained_lines(self) -> list[str]:
        """extract constrained lines from question_annotated"""
        return [line for line in self.init if self._is_init_line_constrained(line, self.constrained_variables)]
    
    @property
    def example_assignments(self) -> dict:
        """extract example assignments from question_annotated"""
        assignment_tuples = re.findall(r"\{(\w+),\s*([^}]+)\}", self.question_template)
        def parse_value(val):
            if val.isnumeric():
                return int(val)
            elif val.startswith("(") and val.endswith(")"):
                v1, v2 = strip_elements(list(val[1:-1].split(",")))
                return v1, v2
            return val
       
        assignments = {var: parse_value(val) for var, val in assignment_tuples}
        answer_vars = re.findall(r"\{(\w+)\}", self.answer_annotated)
        # Ensure that all variables in the answer are also in the question template
        for var in answer_vars:
            if var not in assignments:
                print(f"Warning: Variable {var} found in answer but not in question template. Attempting to derive value from other variables. In question {self.id_shuffled}.")
                assignment_line = next((line for line in self.init if var in self._extract_variables_from_init_line(line)), None)
                vars = self._extract_variables_from_init_line(assignment_line)
                other_var = next((v for v in vars if v != var), None)
                if other_var and other_var in assignments:
                    known_value = re.escape(assignments[other_var])
    
                    # Try to find the unknown value in either position
                    # Pattern 1: known value is first, unknown is second
                    pattern1 = r'\("' + known_value + r'",\s*"([^"]+)"\)'
                    # Pattern 2: known value is second, unknown is first  
                    pattern2 = r'\("([^"]+)",\s*"' + known_value + r'"\)'
                    
                    match = re.search(pattern1, assignment_line)
                    if not match:
                        match = re.search(pattern2, assignment_line)
                    
                    print(f"Pattern 1: {pattern1}")
                    print(f"Pattern 2: {pattern2}")
                    print(f"Assignment line: {assignment_line}")
                    print(f"Other variable: {other_var}")
                    print(f"Match: {match}")
                    try:
                        assignments[var] = match.group(1)
                    except AttributeError:
                        raise AttributeError(f"Could not find a match for variable {var} in assignment line: {assignment_line}. Please check the question template.")
                else:
                    raise ValueError(f"Variable {var} not found in assignments, and no other variable found to derive value from. Please check the question template for question {self.id_shuffled}.")
            
        return assignments

    def _extract_variables_from_init_line(self, line: str) -> list[str]: 
        """extract variable names from a line"""
        variables = line.split("=")[0].strip("- ").strip("$").split(",")
        return [v.strip() for v in variables]
    
    def _is_init_line_constrained(self, line: str, constrained_variables: list[str]) -> bool:
        """check if a line is constrained"""
        return any(v in self._extract_variables_from_init_line(line) for v in constrained_variables)

    def _evaluate_unconstrained_init_line(self, init_line, replacements):
        """ Evaluate a single unconstrained init line and return the assignments."""
        #  If the line is unconstrained, we evaluate it directly since no other variables depend on it.
        logger.debug(f"Evaluating unconstrained init line: {init_line}")
        assignments = {}
        variable_part, definition_part = init_line.split("=", 1)
        variables = strip_elements(variable_part.strip("$").split(","))
        definition_part = definition_part.strip()

        if definition_part.startswith("range(") or definition_part.startswith("list(range("):
            definition_part = "sample(" + definition_part + ")"
        try:
            values = list(eval(definition_part, {"__builtins__": {}}, EVAL_CONTEXT_HELPERS | replacements))
            logger.debug(f"Variables: {variables}, Definition part: {definition_part}, Evaluated values: {values}")
        except Exception as e:
            logger.error(f"Error evaluating assignment for {variable_part}: {definition_part} -> {e}")
            raise e
        if (isinstance(values, list) or isinstance(values, tuple)) and len(values) == len(variables):
            for var, val in zip(variables, values):
                assignments[var] = val
        else:
            logger.warning(f"Warning: {variables} and {values} are incompatible for line {init_line}.")
        
        return assignments

    def _evaluate_constrained_init_lines(self, init_lines, conditions, replacements):
        """ Returns a list of valid combinations of values for the constrained init lines."""
        
        possible_assignments = self._get_all_possible_assignments(init_lines, replacements)
        all_combinations = self._get_all_combinations(possible_assignments)
        return self._filter_invalid_combinations(all_combinations, conditions)

    def _get_all_possible_assignments(self, init_lines, replacements):
        possible_assignments = {}
        for line in init_lines:
            variable_part, definition_part = line.split("=", 1)
            variables = strip_elements(variable_part.strip("$").split(","))
            definition_part = definition_part.strip()
            logger.debug(f"Variables: {variables}, Definition part: {definition_part}")
            if len(variables) == 1:
                variable_name = variables[0].strip()
                try:
                    possible_values = eval(definition_part, {"__builtins__": {}}, COMBINATION_HELPERS | replacements)
                    # Save as a list of tuples to make it easier to generate combinations
                    possible_assignments[variable_name] = list(zip([variable_name] * len(possible_values), possible_values))
                except Exception as e:
                    logger.error(f"Error evaluating line '{line}': {e}")
                    raise e
            else:
                logger.warning(f"Constrained init line '{line}' has more than 1 variable. Skipping...")

        return possible_assignments

    def _get_all_combinations(self, possibilities):
        all_combinations = list(itertools.product(*possibilities.values()))
        combination_dicts = [{k:v for k,v in combination} for combination in all_combinations]
        return combination_dicts

    def _filter_invalid_combinations(self, combinations, conditions):
        valid_combinations = []
        for combination in combinations:
            is_valid = True
            for cond in conditions:
                temp_combination = combination | {k: v[1] for k, v in combination.items() if isinstance(v, tuple)}
                if not eval(cond, {"__builtins__": {}}, EVAL_CONTEXT_HELPERS | temp_combination):
                    is_valid = False
                    break

            if is_valid:
                valid_combinations.append(combination)

        logger.debug(f"Number of valid combinations: {len(valid_combinations)}")
        return valid_combinations

    def format_question(self, assignments, language: str = "eng"):
        def replace_placeholder(match):
            variable_name = match.group(1)
            if variable_name in assignments:
                value = assignments[variable_name]
                return str(value[0]) if isinstance(value, tuple) else str(value)
                
            return match.group(0)
        
        processed_text = re.sub(r"\{(\w+),\s*([^}]+)\}", replace_placeholder, self.question_template)
        processed_text = format_numbers_by_language(processed_text, language)
        return capitalize_sentences(processed_text)

    def format_answer(self, assignments, language: str = "eng"):
        # Handle tuples in the combination
        def contains_arithmetic_operators(expr):
            # Check if the expression is a simple arithmetic expression
            return any(op in expr for op in ['+', '-', '=', '*', '/', '%', '**', '//'])

        def eval_curly_expr(match):
            full_match = match.group(0)      # Full match including prefix/suffix
            expr_str = match.group(1)        # Expression inside curly braces
            
            # Get prefix and suffix by examining the position in the text
            start_pos = match.start()
            end_pos = match.end()
            
            prefix = ""
            suffix = ""
            
            # Check for prefix character
            if start_pos > 0:
                prefix_char = processed_text[start_pos - 1]
                if not prefix_char.isspace():
                    prefix = prefix_char
            
            # Check for suffix character  
            if end_pos < len(processed_text):
                suffix_char = processed_text[end_pos]
                if not suffix_char.isspace():
                    suffix = suffix_char
            
            logger.debug(f"Evaluating expression: {expr_str}")
            if contains_arithmetic_operators(expr_str) or contains_arithmetic_operators(prefix) or contains_arithmetic_operators(suffix):
                eval_env = EVAL_CONTEXT_HELPERS | assignments | {k: int(v[1]) if v[1].isnumeric() else v[1] for k, v in assignments.items() if isinstance(v, tuple)}
            else:
                eval_env = EVAL_CONTEXT_HELPERS | assignments | {k: v[0] for k, v in assignments.items() if isinstance(v, tuple)}
            # Parse the occational float...
            eval_env = eval_env | {k: try_parse_float(v) for k, v in eval_env.items()}
            # Parse the occational fraction...
            eval_env = eval_env | {k: try_parse_fraction(v) for k, v in eval_env.items()}
            try:
                value = eval(expr_str, {"__builtins__": {}}, eval_env)
                logger.debug(f"Evaluated value: {value}")
                # Convert integer-like floats to integers for display
                if isinstance(value, float) and value.is_integer():
                    value = int(value)
                if language == "dan" and isinstance(value, (int, float)):
                    value = str(value).replace(".", ",")
                else:
                    value = str(value)
                return str(value)
            except Exception as e:
                logger.error(f"Error evaluating expression '{expr_str}': {e} with environment {eval_env} for answer {self.answer_annotated} with assignments{assignments} in file {self.id_shuffled}")
                raise e

        # Use a simpler pattern that doesn't capture prefix/suffix
        processed_text = self.answer_annotated
        processed_text = re.sub(r"\{([^}]+)\}", lambda m: eval_curly_expr(m), processed_text)
        processed_text = format_numbers_by_language(processed_text, language)
        return capitalize_sentences(processed_text)
    
    def _generate_question(self, language, replacements: dict[str, list]) -> Question:
        unconstrained_assignments = [self._evaluate_unconstrained_init_line(line, replacements) for line in self.unconstrained_lines]
        logger.debug(f"Unconstrained assignments: {unconstrained_assignments}")
        constrained_assignments = random.choice(self._evaluate_constrained_init_lines(self.constrained_lines, self.conditions, replacements))
        logger.debug(f"Constrained assignments: {constrained_assignments}")
        collected_assignments = constrained_assignments | reduce(lambda x, y: x | y, unconstrained_assignments)
        logger.debug(f"All assignments: {collected_assignments}")
        formatted_question = self.format_question(collected_assignments, language)
        logger.info(f"Formatted question: {formatted_question}")
        formatted_answer = self.format_answer(collected_assignments, language)
        logger.info(f"Formatted answer: {formatted_answer}")
        
        return Question(formatted_question, formatted_answer, self.id_orig, self.id_shuffled)

    def generate_questions(self, n, language: str, replacements: dict[str, list]) -> list[Question]:

        questions = []
        for i in range(n):
            try:
                question = self._generate_question(language, replacements)
                questions.append(question)
            except Exception as e:
                logger.error(f"Error generating question {i + 1}: {e}")
                continue
        return questions


