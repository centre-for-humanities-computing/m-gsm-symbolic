import json
import random
import re
from dataclasses import dataclass
from typing import Self


functions = {
    "range": lambda start, stop, step=1: range(start, stop, step), # step default 1, so it can handle 2 and 3 arguments
    "sample": lambda items, n = 2: random.sample(items, n), # trying to make the dict handle multiple items 
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
        except ValueError:
            pass  # Skip non-numeric values like names    
    text = re.sub(r"\{([^,{}][^{}]*)\}", lambda match: eval_braces(match, num_values), text)
    return text


@dataclass
class GeneratedQuestion:
    question: str
    answer: str
    id_orig: int
    id_shuffled: int


@dataclass
class AnnotatedQuestion:
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
        """extract default values from both the annotated question and answer"""
        default_values = {}

        # find {var_name, value} pairs
        regex = r"\{(\w+),(\w+)\}" # any alphanumeric chr

        # hmmm might not be the smartest solution? But I couldnt replace variable values for the 
        # answer annoted, if I didnt extract the default dict from it? 
        annotated_question_answer = self.question_annotated + self.answer_annotated

        matches = re.findall(regex, annotated_question_answer) # returns a list containing all matches
        for var_name, value in matches:
            default_values[var_name] = value
        return default_values

    @staticmethod
    def remove_calculations(text: str) -> str:
        """remove calculations from the text. re.sub() "replaces" the calculations with noting"""
        return re.sub(r"<<.*?>>", "", text)  
    
    def apply_init_rules(self,
                        names: list[str],
                        multiple_ice: list[int],
                        multi_times: list[int],
                        functions: dict[str, callable]) -> dict[str, str]:

        """apply init rules to generate new variable values"""
        new_values = {}

        for line in self.init.strip().splitlines(): # split str into seperate lines at line break "\n"
            line = line.lstrip("-").strip() # remove the leading "-" character + space
            var, rule = line.split("=") # split var (left side of =) and "rule" (right side of =) by "="
            var = var.strip().lstrip("$") # remove space at the beginning + chr $ in front of the variables
            rule = rule.strip() # again, remove space at the beginning of the string

            if "range" in rule:
                range_arguments = list(map(int, re.findall(r"\d+", rule))) # find range of values, digit chr
                if len(range_arguments) == 2: #hmm might not be the most "stable" solution
                    start, stop = range_arguments
                    new_values[var] = str(random.choice(functions["range"](start, stop)))
                elif len(range_arguments) == 3:
                    start, stop, step = range_arguments
                    new_values[var] = str(random.choice(functions["range"](start, stop, step)))
                
            elif "sample" in rule:
                match_list = re.search(r"sample\((.+?)\)", rule)
                sample_expression = match_list.group(1)

                sampled_arguments = eval(f"functions['sample']({sample_expression})", {"names": names,
                                                                                       "multiple_ice": multiple_ice,
                                                                                       "multi_times": multi_times,
                                                                                       "functions": functions})
                var_arguments = [v.strip() for v in var.split(",")]

                for var_sample, val_sample in zip(var_arguments, sampled_arguments):
                    print(f"Assigning {var_sample} = {val_sample}") 
                    new_values[var_sample] = val_sample

            #divides????
            # I dont think it nessecary to define "divides" -cant it just be calulcated automatically in my eval?

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
    

    # TEST
    def generate_question(self,
                          names: list[str],
                          multiple_ice: list[int],
                          multi_times: list[int],
                          functions: dict[str, callable]) -> GeneratedQuestion:
        """
        The function should take the "original question", then sample new values using the apply_init_rules,
        replace the default values using the replace_values function, evaluate any embedded math expressions,
        and return everything as a GeneratedQuestion dataclass
        """
        new_values = self.apply_init_rules(names, multiple_ice, multi_times, functions)
        q_annotated, a_annotated = self.replace_values(new_values)
        a_final = eval_expressions(a_annotated, new_values)
        return GeneratedQuestion(question = self.remove_calculations(q_annotated),
                                 answer = self.remove_calculations(a_final),
                                 id_orig = self.id_orig,
                                 id_shuffled = self.id_shuffled)
    

if __name__ == "__main__":
    
    names = ["Sofie", "Andrea", "Freja", "Ida", "Clara", "Anna"]
    multiple_ice = [2, 3]
    multi_times = [2, 3]

    # sample, range, divides
    json_data_1 = r"""
    {
    "question": "Et brød har 24 skiver. Hvis Andrea kan spise 2 skiver om dagen, mens Johan kan spise dobbelt så meget, hvor mange dage holder brødet så?",
    "answer": "Johan kan spise 2 x 2 = <<2*2=4>>4 skiver om dagen.\nSammen kan Andrea og Johan spise 2 + 4 = <<2+4=6>>6 skiver om dagen.\nSå et brød vil holde i 24/6 = <<24/6=4>>4 dage.\n#### 4",
    "id_orig": 921,
    "id_shuffled": 37,
    "question_annotated": "Et {item,brød} har {n,24} {unit,skiver}. Hvis {name1,Andrea} kan spise {x,2} {unit,skiver} om dagen, mens {name2,Johan} kan spise {mult,dobbelt} så meget, hvor mange dage holder {item,brød}et så?\n\n#init:\n- item = sample([\"pizza\", \"kage\", \"tærte\", \"lasagne\"])\n- unit = sample([\"stykker\", \"portioner\", \"kuverter\"])\n- name1, name2 = sample(names, 2)\n- $n = range(12, 49, 3)\n- $x = range(2, 6)\n- $mult = sample(multiple_ice+multi_times)\n\n#conditions:\n- divides(n, x + x*mult)\n\n#answer: n // (x + x*mult)",
    "answer_annotated": "{name2} kan spise {x} x {mult} = <<{x}*{mult}={x*mult}>>{x*mult} {unit} om dagen.\nSammen kan {name1} og {name2} spise {x} + {x*mult} = <<{x}+{x*mult}={x+x*mult}>>{x+x*mult} {unit} om dagen.\nSå et {item} vil holde i {n}/{x+x*mult} = <<{n}/{x+x*mult}={n//(x+x*mult)}>>{n//(x+x*mult)} dage.\n#### {n//(x+x*mult)}"
    }"""

    # sample, range
    json_data_2 = r"""
    {
    "question": "Der er i øjeblikket 16 år mellem Mia og Emma. Hvis Mia, der er yngre end Emma, er 40 år gammel, hvad er gennemsnittet af deres alder?",
    "answer": "Hvis Mia er 40 år, er Emma 40+16 = <<40+16=56>>56 år.\nSummen af deres alder er 56+40 = <<56+40=96>>96 år\nGennemsnitsalderen for de to er 96/2 = <<96/2=48>>48 år\n#### 48",
    "id_orig": 1277,
    "id_shuffled": 31,
    "question_annotated": "Der er i øjeblikket {age_diff,16} år mellem {name1,Mia} og {name2,Emma}. Hvis {name1,Mia}, wder er yngre end {name2,Emma}, er {age1,40} år gammel, hvad er gennemsnittet af deres alder?\n\n#init:\n- name1, name2 = sample(names, 2)\n- $age_diff = range(5, 30)\n- $age1 = range(15, 75)\n\n#conditions:\n- is_int((2*age1 + age_diff) / 2)\n\n#answer: (2*age1 + age_diff) // 2",
    "answer_annotated": "Hvis {name1} er {age1} år gammel, er {name2} {age1}+{age_diff} = <<{age1}+{age_diff}={age1+age_diff}>>{age1+age_diff} år.\nSummen af deres alder er {age1+age_diff}+{age1} = <<{age1+age_diff}+{age1}={2*age1+age_diff}>>{2*age1+age_diff} år\nGennemsnitsalderen for de to er {2*age1+age_diff}/2 = <<{2*age1+age_diff}/2={(2*age1+age_diff)//2}>>{(2*age1+age_diff)//2} år\n#### {(2*age1+age_diff)//2}"
    }"""

    # Create the Question object
    question = AnnotatedQuestion.from_json(json_data_2)

    # Print values using the dataclass
    print(f"Question:\n {question.question}")
    print(f"\nInit section:\n {question.init}")
    print(f"\nDefault values: {question.default_values}")
    print(f"\nAnswer witout calculations:\n {question.remove_calculations(question.answer)}")

    # Sample new values using the init rules and save in a dict
    new_values = question.apply_init_rules(names, multiple_ice, multi_times, functions)
    print(f"New values: {new_values}")

    # replace the default values
    question.replace_values(new_values)

    #print("\nUpdated annotated question:")
    print(question.question_annotated)

    #print("\nUpdated annotated answer:")
    print(question.answer_annotated)

    # calculate result
    question.answer_annotated = eval_expressions(question.answer_annotated, new_values)
    print(question.answer_annotated)

    ## TEST - generate question function
    print("\n\nTEST, generated question func")

    generated_question = question.generate_question(names, multiple_ice, multi_times, functions)
    print(f"Generated question: {generated_question.question}")
    print(f"Generated answer: {generated_question.answer}")