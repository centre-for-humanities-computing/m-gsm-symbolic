\
import json
import re
import itertools
import random
import ast
import argparse
import os
from m_gsm_symbolic.replacements_list import replacements

# --- Helper functions for eval context ---
def is_int(val):
    return isinstance(val, int) or (isinstance(val, float) and val.is_integer())

def divides(a, b):
    if not (isinstance(a, (int, float)) and isinstance(b, (int, float))):
        return False # Ensure inputs are numeric
    if b == 0: return False
    return a % b == 0

EVAL_CONTEXT_HELPERS = {
    "is_int": is_int,
    "divides": divides,
    "int": int,
    "float": float,
    "round": round,
    "str": str,
    "len": len,
    # Add other safe builtins or math functions if needed, e.g. from 'math' module
}

# --- Parsing the template ---
def parse_template(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    question_annotated_full = data["question_annotated"]
    answer_annotated_template = data["answer_annotated"]

    # First, extract the question text before any metadata
    # Find the first occurrence of any metadata section
    metadata_start_idx = -1
    for marker in [ "#init:", "#conditions:", "#answer:"]:
        idx = question_annotated_full.find(marker)
        if idx != -1 and (metadata_start_idx == -1 or idx < metadata_start_idx):
            metadata_start_idx = idx
    
    if metadata_start_idx != -1:
        question_template_text = question_annotated_full[:metadata_start_idx]
        metadata_block = question_annotated_full[metadata_start_idx:]
    else:
        question_template_text = question_annotated_full
        metadata_block = ""

    init_rules_str = []
    conditions_str = []
    answer_formula_str = None

    if metadata_block:
        current_section = None
        for line in metadata_block.splitlines():
            line = line.strip()
            if not line: continue
            if line.startswith("#init:"):
                current_section = "init"
            elif line.startswith("#conditions:"):
                current_section = "conditions"
            elif line.startswith("#answer:"):
                current_section = "answer"
                answer_formula_str = line[len("#answer:"):].strip()
            elif current_section == "init" and line.startswith("- "):
                init_rules_str.append(line[2:])
            elif current_section == "conditions" and line.startswith("- "):
                conditions_str.append(line[2:])
    
    return {
        "question_template_text": question_template_text,
        "answer_annotated_template": answer_annotated_template,
        "init_rules_str": init_rules_str,
        "conditions_str": conditions_str,
        "answer_formula_str": answer_formula_str,
    }

def parse_init_rules(init_rules_str):
    parsed_vars = {}
    for rule_str in init_rules_str:
        var_name, definition = rule_str.split("=", 1)
        var_name = var_name.strip()
        definition = definition.strip()
        
        is_numeric_flag = False
        if var_name.startswith("$"):
            var_name = var_name[1:]
            is_numeric_flag = True

        current_var_info = {'values': [], 'display_map': {}, 'is_numeric': is_numeric_flag}

        if definition.startswith("range("):
            try:
                args_str = definition[len("range("):-1]
                args = [int(a.strip()) for a in args_str.split(',')]
                current_var_info['values'] = list(range(*args))
            except Exception as e:
                print(f"Error parsing range for {var_name}: {definition} -> {e}")
        elif definition.startswith("sample("):
            content_str = definition[len("sample("):-1].strip()
            try:
                potential_list_or_key = ast.literal_eval(content_str)
                if isinstance(potential_list_or_key, list):
                    if potential_list_or_key and isinstance(potential_list_or_key[0], tuple) and len(potential_list_or_key[0]) == 2:
                        values = []
                        display_map = {}
                        for str_val, actual_val in potential_list_or_key:
                            values.append(actual_val)
                            display_map[actual_val] = str_val
                        current_var_info['values'] = values
                        current_var_info['display_map'] = display_map
                    else:
                        current_var_info['values'] = potential_list_or_key
                # If not a list, ast.literal_eval might have returned a string (the key itself)
                elif isinstance(potential_list_or_key, str) and potential_list_or_key in replacements:
                     current_var_info['values'] = replacements[potential_list_or_key]
                else: # Not a list literal, assume it's a key for replacements if it's a simple string
                    print(f"Warning: sample content '{content_str}' for var '{var_name}' not a list or recognized key after ast.literal_eval.")

            except (SyntaxError, ValueError): # Not a valid Python literal, assume it's a key string
                 if content_str in replacements:
                    current_var_info['values'] = replacements[content_str]
                 else:
                    print(f"Warning: sample key '{content_str}' for var '{var_name}' not in replacements or invalid format.")
        else:
            try:
                current_var_info['values'] = [ast.literal_eval(definition)]
            except (SyntaxError, ValueError):
                print(f"Warning: Could not parse direct assignment for {var_name}: {definition}")
        
        parsed_vars[var_name] = current_var_info
    return parsed_vars

# --- Generating and filtering combinations ---
def generate_valid_combinations(parsed_vars, conditions_str):
    var_names = list(parsed_vars.keys())
    if not var_names: return [] # No variables to combine
    # No reason to check for constraints on variables that are not in the conditions
    constrained_vars = {k:v for k,v in parsed_vars.items() if any(k in condition for condition in conditions_str)}
    unconstrained_vars = {k:v for k,v in parsed_vars.items() if k not in constrained_vars}
    constrained_value_lists = [v["values"] for v in constrained_vars.values()]
    all_combinations = itertools.product(*constrained_value_lists)

    valid_constrained_combinations = []

    for combo_values in all_combinations:
        current_combination_dict = dict(zip(list(constrained_vars.keys()), combo_values))
        eval_env = {**EVAL_CONTEXT_HELPERS, **current_combination_dict}
        
        all_conditions_met = True
        for cond_str in conditions_str:
            try:
                # Use a restricted global scope for eval, allowing only our helpers and variables
                if not eval(cond_str, {"__builtins__": {}}, eval_env):
                    all_conditions_met = False
                    break
            except Exception: # Catches NameError for undefined functions too
                all_conditions_met = False
                break
        
        if all_conditions_met:
            valid_constrained_combinations.append(current_combination_dict)
            
    return unconstrained_vars, valid_constrained_combinations

# --- Formatting output ---
def format_question(question_template_text, combination, parsed_vars_info):
    def replace_placeholder(match):
        var_name = match.group(1)
        if var_name in combination:
            actual_value = combination[var_name]
            var_info = parsed_vars_info.get(var_name)
            if var_info and var_info['display_map']:
                return str(var_info['display_map'].get(actual_value, actual_value))
            return str(actual_value)
        return match.group(0)
    return re.sub(r"\{(\w+),\s*([^}]+)\}", replace_placeholder, question_template_text)

def format_answer(answer_annotated_template, combination, answer_formula_str):
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

    if answer_formula_str:
        # The answer_formula_str is an expression, possibly wrapped in {} itself in some conventions,
        # but here it's taken directly from #answer: line.
        # Ensure it's treated as an expression to evaluate.
        final_answer_expr_to_eval = answer_formula_str.strip("{}") # Remove if wrapped
        try:
            # Simulate a match object for eval_curly_expr if formula was like "{...}"
            # Or directly evaluate if it's a bare expression.
            if answer_formula_str.startswith("{") and answer_formula_str.endswith("}"):
                 final_answer_val_str = eval_curly_expr(type('obj', (object,),{'group': lambda i, expr=final_answer_expr_to_eval: expr})())
            else: # Bare expression
                 val = eval(final_answer_expr_to_eval, {"__builtins__": {}}, eval_env)
                 # Make sure integer-like floats are displayed as integers
                 if isinstance(val, float) and val.is_integer():
                     val = int(val)
                 final_answer_val_str = str(val)


            processed_text = re.sub(r"####\s*.*", f"#### {final_answer_val_str}", processed_text, count=1)
        except Exception as e:
            processed_text = re.sub(r"####\s*.*", f"#### [Error evaluating: {final_answer_expr_to_eval}]", processed_text, count=1)
            
    return processed_text

def main_logic(template_filepath, num_samples_to_generate, output_filepath=None):

    template_data = parse_template(template_filepath)
    parsed_vars_info = parse_init_rules(template_data["init_rules_str"])
    if not all(parsed_vars_info.get(var, {}).get('values') for var in parsed_vars_info):
        print("Error: Some variables have no possible values. Check init rules and replacements list.")
        return

    unconstrained_vars, valid_combinations = generate_valid_combinations(parsed_vars_info, template_data["conditions_str"])

    if not valid_combinations:
        print("No valid combinations found that satisfy all conditions.")
        return

    actual_num_to_generate = min(num_samples_to_generate, len(valid_combinations))
    if num_samples_to_generate > len(valid_combinations):
        print(f"Warning: Requested {num_samples_to_generate} samples, but only {len(valid_combinations)} unique valid combinations exist. Generating {len(valid_combinations)} samples.")

    selected_combinations = random.sample(valid_combinations, actual_num_to_generate)
    selected_unconstrained_values = [{k: random.choice(v["values"]) for k, v in unconstrained_vars.items()} for _ in range(actual_num_to_generate)]
    merged_combinations = [dict(combo, **unconstrained) for combo, unconstrained in zip(selected_combinations, selected_unconstrained_values)]
    output_samples = []

    for i, combo in enumerate(merged_combinations):
        generated_question = format_question(
            template_data["question_template_text"], combo, parsed_vars_info
        )
        generated_answer = format_answer(
            template_data["answer_annotated_template"], combo, template_data["answer_formula_str"]
        )
        output_samples.append({
            "id_original_template": os.path.basename(template_filepath),
            "sample_index": i,
            "variables": combo,
            "question": generated_question,
            "answer": generated_answer
        })

    if output_filepath:
        with open(output_filepath, 'w', encoding='utf-8') as f:
            json.dump(output_samples, f, indent=2, ensure_ascii=False)
        print(f"Generated {len(output_samples)} samples and saved to {output_filepath}")
    else:
        print(json.dumps(output_samples, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate samples from annotated JSON templates.")
    parser.add_argument("template_filepath", help="Path to the JSON template file.")
    parser.add_argument("num_samples", type=int, help="Number of samples to generate.")
    parser.add_argument("-o", "--output", help="Path to the output JSON file (optional). Prints to stdout if not provided.")
    
    args = parser.parse_args()

    main_logic(args.template_filepath, args.num_samples, args.output)
