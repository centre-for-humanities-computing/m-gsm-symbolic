import json
import random
import re
from dataclasses import dataclass
from typing import Self, Any, Callable
import logging

logger = logging.getLogger(__name__)

functions = {"range": lambda start, stop, step = 1: range(start, stop, step),
             "sample": lambda items, n = 2: random.sample(items, min(len(items), n)),
             "divides": lambda a, b: a % b == 0,}

def eval_braces(match, num_values):
    """ Helper function to evalulate the expressions inside the {}"""
    expr = match.group(1)  # Get the expression inside the {}
    try:
        result = eval(expr, {}, num_values)  # Evaluate the math
        return str(result)  # Replace with the result
    except Exception as e:
        return match.group(0)

def eval_expressions(text: str, new_values: dict[str, str]) -> str:
    '''
    Evaluate all expressions like {age1+age_diff} and <<expression=result>> in the text.
    Only keep numeric values. Uses the eval_braces helper function to replace expressions inside brace
    '''
    num_values = {}
    for i, v in new_values.items():
        try:
            num_values[i] = int(v) # Try converting the value to an integer
        except (ValueError, TypeError):
            pass  # Skip non-numeric values like names    
    text = re.sub(r"\{([^,{}][^{}]*)\}", lambda match: eval_braces(match, num_values), text)
    return text

@dataclass  
class Question:  
    question: str  
    answer: str  
    id_orig: int  
    id_shuffled: int 

@dataclass
class AnnotatedQuestion(Question):
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
        """extract default values from both the annotated question and answer"""
        default_values = {}
        regex = r"\{(\w+),(\w+)\}" # find {var_name, value} pairs. Any alphanumeric chr
        annotated_question_answer = self.question_annotated + self.answer_annotated
        matches = re.findall(regex, annotated_question_answer) # returns a list containing all matches
        for var_name, value in matches:
            default_values[var_name] = value
        return default_values

    @staticmethod
    def remove_calculations(text: str) -> str:
        """remove calculations from the text. re.sub() replaces the calculations with noting"""
        return re.sub(r"<<.*?>>", "", text)  
    
    def apply_init_rules(self,
                         replacements: dict[str, Any] = {},
                         functions: dict[str, Callable] = functions) -> dict[str, str]:
        """apply init rules to generate new variable values"""

        new_values = {}
        combined = {**replacements, "functions": functions}

        for line in self.init.strip().splitlines(): # split str into seperate lines at line break "\n"
            line = line.lstrip("-").strip() # remove the leading "-" character + space
            if "=" not in line:
                continue
            var, rule = line.split("=", 1) # split var (left side of =) and "rule" (right side of =) by "="
            var = var.strip().lstrip("$") # remove space at the beginning + chr $ in front of the variables
            rule = rule.strip() # again, remove space at the beginning of the string

            if "range" in rule:
                range_arguments = list(map(int, re.findall(r"\d+", rule))) # find range of values, digit chr
                if len(range_arguments) == 2:
                    start, stop = range_arguments
                    new_values[var] = str(random.choice(combined["functions"]["range"](start, stop)))
                elif len(range_arguments) == 3:
                    start, stop, step = range_arguments
                    new_values[var] = str(random.choice(combined["functions"]["range"](start, stop, step)))

            elif "sample" in rule:
                match_list = re.search(r"sample\((.*)\)", rule)
                if not match_list:
                    continue
                sample_expression = match_list.group(1)
                sampled_arguments = eval(f"functions['sample']({sample_expression})", combined)
                var_arguments = [v.strip() for v in var.split(",")]

                for var_sample, val_sample in zip(var_arguments, sampled_arguments):
                    logger.info(f"Assigning {var_sample} = {val_sample}") 
                    new_values[var_sample] = val_sample

        return new_values


    def replace_values(self, new_values: dict[str, str]) -> None:
        """replace default values in the annotated strings with new sampled values from new_values dict"""
        for var, default_value in self.default_values.items():
            new_value = new_values.get(var, default_value)

            self.question_annotated = re.sub(f"{{{var},{default_value}}}", str(new_value), self.question_annotated)
            self.answer_annotated = re.sub(f"{{{var},{default_value}}}", str(new_value), self.answer_annotated)

        for var, new_value in new_values.items():
            self.question_annotated = re.sub(rf"\{{{var}\}}", str(new_value), self.question_annotated)
            self.answer_annotated = re.sub(rf"\{{{var}\}}", str(new_value), self.answer_annotated)

        return self.question_annotated, self.answer_annotated    


    def generate_question(self,
                          replacements: dict[str, Any] = {},
                          functions: dict[str, Callable] = functions) -> Question:
        """
        The function should take the "original question", then sample new values using the apply_init_rules,
        replace the default values using the replace_values function, evaluate any embedded math expressions,
        and return everything as Question dataclass
        """
        new_values = self.apply_init_rules(replacements, functions)
        q_annotated, a_annotated = self.replace_values(new_values)
        a_final = eval_expressions(a_annotated, new_values)
        return Question(question = self.remove_calculations(q_annotated),
                        answer = self.remove_calculations(a_final),
                        id_orig = self.id_orig,
                        id_shuffled = self.id_shuffled)
