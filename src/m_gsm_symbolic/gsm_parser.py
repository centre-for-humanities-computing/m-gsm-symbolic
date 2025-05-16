from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
import itertools
import random
import argparse
import logging
from functools import reduce
from typing import Self

# Set up logger
logger = logging.getLogger(__name__)

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
    def variables(self) -> list[str]:
        """extract variable names from question_annotated"""
        init_block = self.question_annotated.split("#init:")[1].split("#conditions:")[0].strip().splitlines()
        variables_per_line = [extract_variables_from_init_line(line) for line in init_block if "=" in line]
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
        return [line for line in self.init if not is_init_line_constrained(line, self.constrained_variables)]
    
    @property
    def constrained_lines(self) -> list[str]:
        """extract constrained lines from question_annotated"""
        return [line for line in self.init if is_init_line_constrained(line, self.constrained_variables)]
    
def extract_variables_from_init_line(line: str) -> list[str]: 
    """extract variable names from a line"""
    variables = line.split("=")[0].strip("- ").strip("$").split(",")
    return [v.strip() for v in variables]
    
def is_init_line_constrained(line: str, constrained_variables: list[str]) -> bool:
    """check if a line is constrained"""
    return any(v in extract_variables_from_init_line(line) for v in constrained_variables)

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
}

COMBINATION_HELPERS = {
    "range": range_possibilities,
    "sample": sample_possibilities,
    "list": list,
}

def evaluate_unconstrained_init_line(init_line):
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
    if isinstance(values, list) and len(values) == len(variables):
        for var, val in zip(variables, values):
            assignments[var] = val
    else:
        logger.warning(f"Warning: {variables} and {values} are incompatible for line {init_line}.")
    
    return assignments

def evaluate_constrained_init_lines(init_lines, conditions):
    """ Returns a list of valid combinations of values for the constrained init lines."""
    
    possible_assignments = get_all_possible_assignments(init_lines)
    all_combinations = get_all_combinations(possible_assignments)
    return filter_invalid_combinations(all_combinations, conditions)

def get_all_possible_assignments(init_lines):
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

def get_all_combinations(possibilities):
    all_combinations = list(itertools.product(*possibilities.values()))
    combination_dicts = [{k:v for k,v in combination} for combination in all_combinations]
    return combination_dicts

def filter_invalid_combinations(combinations, conditions):
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

def format_question(question_template_text, combination):
    def replace_placeholder(match):
        variable_name = match.group(1)
        if variable_name in combination:
            value = combination[variable_name]
            return str(value[0]) if isinstance(value, tuple) else str(value)
        return match.group(0)
    return re.sub(r"\{(\w+),\s*([^}]+)\}", replace_placeholder, question_template_text)

def format_answer(answer_annotated_template, combination, ):
    # Handle tuples in the combination
    eval_env = EVAL_CONTEXT_HELPERS | combination | {k: v[1] for k, v in combination.items() if isinstance(v, tuple)}

    def eval_curly_expr(match):
        expr_str = match.group(1)
        logger.debug(f"Evaluating expression: {expr_str}")
        try:
            value = eval(expr_str, {"__builtins__": {}}, eval_env)
            logger.debug(f"Evaluated value: {value}")
            # Convert integer-like floats to integers for display
            if isinstance(value, float) and value.is_integer():
                value = int(value)
            return str(value)
        except Exception as e:
            logger.error(f"Error evaluating expression '{expr_str}': {e}")
            raise e

    processed_text = re.sub(r"\{([^}]+)\}", eval_curly_expr, answer_annotated_template)
            
    return processed_text

def generate_question(template_path: Path) -> Question:
    question = AnnotatedQuestion.from_json(template_path)
    logger.debug(f"Annotated question: {question}")
    unconstrained_assignments = [evaluate_unconstrained_init_line(line) for line in question.unconstrained_lines]
    logger.debug(f"Unconstrained assignments: {unconstrained_assignments}")
    constrained_assignments = random.choice(evaluate_constrained_init_lines(question.constrained_lines, question.conditions))
    logger.debug(f"Constrained assignments: {constrained_assignments}")
    all_assignments = constrained_assignments | reduce(lambda x, y: x | y, unconstrained_assignments)
    logger.debug(f"All assignments: {all_assignments}")
    formatted_question = format_question(question.question_template, all_assignments)
    logger.info(f"Formatted question: {formatted_question}")
    formatted_answer = format_answer(question.answer_annotated, all_assignments)
    logger.info(f"Formatted answer: {formatted_answer}")
    
    # TODO: Set id_shuffled to something meaningful
    return Question(formatted_question, formatted_answer, template_path.stem, 0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate samples from annotated JSON templates.")
    parser.add_argument("template_path", help="Path to the JSON template file(s).")
    parser.add_argument("num_samples", type=int, help="Number of samples to generate for each template.")
    parser.add_argument("language", help="Language code for the template (e.g., 'en', 'da').")
    parser.add_argument("-o", "--output", help="Output directory.")
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress all output except errors and warnings.")
    parser.add_argument("-d", "--debug", action="store_true", help="Enable debug output (debug level logging)")
    
    args = parser.parse_args()
    
    if args.language == "da":
        from m_gsm_symbolic.replacements_list_da import replacements
    else:
        from m_gsm_symbolic.replacements_list_en import replacements

    # Configure logging
    if args.quiet:
        log_level = logging.WARNING
    elif args.debug:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO

    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger.info("Starting sample generation...")
    # Check if the template path is a directory or a file
    template_path = Path(args.template_path)
    if template_path.is_dir():
        template_files = list(template_path.glob("*.json"))
        if not template_files:
            logger.error(f"No JSON files found in directory: {template_path}")
            exit(1)
    elif template_path.is_file():
        template_files = [template_path]
    else:
        logger.error(f"Invalid path: {template_path}")
        exit(1)
    for template_file in template_files:
        logger.info(f"Processing template file: {template_file}")
        # Generate samples for each template file
        for i in range(args.num_samples):
            # Generate a sample
            logger.info(f"Generating question {i + 1}/{args.num_samples}")
            question = generate_question(template_file)
            if args.output:
                output_dir = Path(args.output)
                output_dir.mkdir(parents=True, exist_ok=True)
                output_file = output_dir / f"{template_file.stem}_{i + 1}.json"
                question.to_json(output_file)
                logger.info(f"Sample saved to: {output_file}")


