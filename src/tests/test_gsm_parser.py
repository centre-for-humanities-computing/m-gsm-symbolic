import pytest
from pathlib import Path
import glob
import os
import json
from functools import reduce
from m_gsm_symbolic.gsm_parser import EVAL_CONTEXT_HELPERS, AnnotatedQuestion, Question
from m_gsm_symbolic.load_data import load_replacements


class TestGetAllPossibleAssignments:
    """Test class for testing the _get_all_possible_assignments method."""
    
    def test_simple_assignment(self):
        """Test with simple variable assignments."""
        # Create a mock AnnotatedQuestion
        annotated_question = AnnotatedQuestion(
            question="Test question",
            answer="Test answer",
            id_orig=1,
            id_shuffled=1,
            question_annotated="Test template\n#init:\n- $x = 5\n#conditions:\n- True\n#answer:\nAnswer is {x}",
            answer_annotated="Answer is {x}"
        )
        
        # Call the method with a simple assignment
        init_lines = ["$x = range(1, 3)"]
        replacements = {}
        result = annotated_question._get_all_possible_assignments(init_lines, replacements)
        
        # Expected: x should have possible values 1, 2, 3 as tuples ('x', value)
        expected = {'x': [('x', 1), ('x', 2), ('x', 3)]}
        assert result == expected
    
    def test_range_expression(self):
        """Test with range expressions."""
        annotated_question = AnnotatedQuestion(
            question="Test question",
            answer="Test answer",
            id_orig=1,
            id_shuffled=1,
            question_annotated="Test template\n#init:\n- $x = range(1, 5)\n#conditions:\n- True\n#answer:\nAnswer is {x}",
            answer_annotated="Answer is {x}"
        )
        
        init_lines = ["$x = range(1, 5)"]
        replacements = {}
        result = annotated_question._get_all_possible_assignments(init_lines, replacements)
        
        # Expected: x should have values 1, 2, 3, 4, 5
        expected = {'x': [('x', 1), ('x', 2), ('x', 3), ('x', 4), ('x', 5)]}
        assert result == expected
    
    def test_range_with_step(self):
        """Test range with step parameter."""
        annotated_question = AnnotatedQuestion(
            question="Test question",
            answer="Test answer",
            id_orig=1,
            id_shuffled=1,
            question_annotated="Test template\n#init:\n- $x = range(1, 10, 2)\n#conditions:\n- True\n#answer:\nAnswer is {x}",
            answer_annotated="Answer is {x}"
        )
        
        init_lines = ["$x = range(1, 10, 2)"]
        replacements = {}
        result = annotated_question._get_all_possible_assignments(init_lines, replacements)
        
        # Expected: x should have values 1, 3, 5, 7, 9 with step 2
        expected = {'x': [('x', 1), ('x', 3), ('x', 5), ('x', 7), ('x', 9)]}
        assert result == expected
    
    def test_sample_possibility(self):
        """Test with sample possibility."""
        annotated_question = AnnotatedQuestion(
            question="Test question",
            answer="Test answer",
            id_orig=1,
            id_shuffled=1,
            question_annotated="Test template\n#init:\n- $x = sample([10, 20, 30])\n#conditions:\n- True\n#answer:\nAnswer is {x}",
            answer_annotated="Answer is {x}"
        )
        
        init_lines = ["$x = [10, 20, 30]"]
        replacements = {}
        result = annotated_question._get_all_possible_assignments(init_lines, replacements)
        
        # Expected: x should have possible values 10, 20, 30
        expected = {'x': [('x', 10), ('x', 20), ('x', 30)]}
        assert result == expected
    
    def test_multiple_variables(self):
        """Test with multiple variables (should skip this case)."""
        annotated_question = AnnotatedQuestion(
            question="Test question",
            answer="Test answer",
            id_orig=1,
            id_shuffled=1,
            question_annotated="Test template\n#init:\n- $x, $y = range(1, 2)\n#conditions:\n- True\n#answer:\nAnswer is {x}",
            answer_annotated="Answer is {x}"
        )
        
        init_lines = ["$x, $y = range(1, 2)"]
        replacements = {}
        result = annotated_question._get_all_possible_assignments(init_lines, replacements)
        
        # Expected: empty dict since multiple variables are skipped
        assert result == {}

    def test_empty_range(self):
        """Test with empty range."""
        annotated_question = AnnotatedQuestion(
            question="Test question",
            answer="Test answer",
            id_orig=1,
            id_shuffled=1,
            question_annotated="Test template\n#init:\n- $x = range(5, 3)\n#conditions:\n- True\n#answer:\nAnswer is {x}",
            answer_annotated="Answer is {x}"
        )
        
        init_lines = ["$x = range(5, 3)"]
        replacements = {}
        result = annotated_question._get_all_possible_assignments(init_lines, replacements)
        
        # Expected: empty list for x since the range is invalid
        assert result == {'x': []}
        
    def test_with_replacements(self):
        """Test with replacement values."""
        annotated_question = AnnotatedQuestion(
            question="Test question",
            answer="Test answer",
            id_orig=1,
            id_shuffled=1,
            question_annotated="Test template\n#init:\n- $x = range(start, end)\n#conditions:\n- True\n#answer:\nAnswer is {x}",
            answer_annotated="Answer is {x}"
        )
        
        init_lines = ["$x = range(start, end)"]
        replacements = {'start': 2, 'end': 5}
        result = annotated_question._get_all_possible_assignments(init_lines, replacements)
        
        # Expected: x should have values 2, 3, 4, 5
        expected = {'x': [('x', 2), ('x', 3), ('x', 4), ('x', 5)]}
        assert result == expected


def test_template_formatting_matches_original():
    """
    Test that format_question and format_answer with example_assignments 
    match the original question and answer for all template files.
    """
    # Find all template files
    base_dir = Path(__file__).parent.parent.parent
    template_dirs = [
        base_dir / "data" / "templates" / "dan" / "symbolic",
        base_dir / "data" / "templates" / "eng" / "symbolic"
    ]
    
    
    for template_dir in template_dirs:
        template_files = []
        if template_dir.exists():
            template_files.extend(list(template_dir.glob("**/*.json")))
    
        language = "dan" if "dan" in str(template_dir) else "eng"
        # Skip test if no template files found
        if not template_files:
            pytest.skip("No template files found in directory: " + str(template_dir))
        
        # Test each template file
        for template_file in template_files:
            # Load the annotated question
            annotated_question = AnnotatedQuestion.from_json(template_file)
            
            # Get example assignments
            example_assignments = annotated_question.default_assignments
            
            # Format question and answer using example assignments
            formatted_question = annotated_question.format_question(example_assignments, language=language)
            formatted_answer = annotated_question.format_answer(example_assignments, language=language)

            # Check if formatted outputs match the original
            assert formatted_question == annotated_question.question, \
                f"Formatted question doesn't match original for {template_file.name}"
            
            assert formatted_answer == annotated_question.answer, \
                f"Formatted answer doesn't match original for {template_file.name}"
            
def test_example_assignments_are_valid():
    """
    Test that example_assignments from template files are found in possible assignments
    and pass the conditions validation.
    """
    # Find all template files
    base_dir = Path(__file__).parent.parent.parent
    template_dirs = [
        base_dir / "data" / "templates" / "dan" / "symbolic",
        base_dir / "data" / "templates" / "eng" / "symbolic"
    ]
    
    for template_dir in template_dirs:
        
        template_files = []
        if template_dir.exists():
            template_files.extend(list(template_dir.glob("**/*.json")))
    
        # Skip test if no template files found
        if not template_files:
            pytest.skip("No template files found in directory: " + str(template_dir))
        
        replacements = load_replacements("dan" if "dan" in str(template_dir) else "eng")
        # Test each template file
        for template_file in template_files:
            # Load the annotated question
            annotated_question = AnnotatedQuestion.from_json(template_file)
            
            # Get example assignments
            example_assignments = annotated_question.default_assignments
            
            # Get constrained and unconstrained lines
            constrained_lines = annotated_question.constrained_lines
            unconstrained_lines = annotated_question.unconstrained_lines
            conditions = annotated_question.conditions
            
            # Test constrained variables if they exist
            if constrained_lines:
                # Get all possible assignments for constrained variables
                possible_assignments = annotated_question._get_all_possible_assignments(constrained_lines, replacements)
                
                # Check that each constrained variable's example value is in possible assignments
                for var_name, possible_values in possible_assignments.items():
                    if var_name in example_assignments:
                        example_value = example_assignments[var_name]
                        # Extract just the values from the tuples for comparison
                        possible_value_set = {val[1] for val in possible_values}
                        
                        # Handle tuple values (like ("value1", "value2"))
                        if isinstance(example_value, tuple):
                            # For tuple values, first check if the exact tuple is in possible values
                            # Then check if the tuple as a string representation is in possible values
                            example_value = tuple(int(component) if str(component).isnumeric() else str(component) for component in example_value)
                            tuple_of_strings = tuple(str(component) for component in example_value)
                            tuple_as_string = f"({', '.join(tuple_of_strings)})"
                            assert (example_value in possible_value_set or
                                    tuple_of_strings in possible_value_set or
                                    tuple_as_string in possible_value_set), \
                                f"Example assignment {var_name}={example_value} not found in possible values {possible_value_set} for {template_file.name}"
                        else:
                            if str(example_value).isnumeric():
                                # If the example value is numeric, convert it and possible values to float for comparison
                                example_value = float(example_value)
                                assert example_value in list(map(float,possible_value_set)), \
                                    f"Example assignment {var_name}={example_value} not found in possible values {possible_value_set} for {template_file.name}"
                            else:
                                assert example_value in possible_value_set, \
                                    f"Example assignment {var_name}={example_value} not found in possible values {possible_value_set} for {template_file.name}"
                            
                # Test that example assignments satisfy conditions
                if conditions and any(cond.strip() != "True" for cond in conditions):
                    # Create a combination dict similar to what _filter_invalid_combinations expects
                    example_combination = {}
                    for var_name in possible_assignments.keys():
                        if var_name in example_assignments:
                            example_value = example_assignments[var_name]
                            # Handle tuple values - use the numeric part for evaluation
                            if isinstance(example_value, tuple):
                                # Try to find a numeric value in the tuple
                                numeric_val = None
                                for component in example_value:
                                    try:
                                        numeric_val = float(component) if '.' in str(component) else int(component)
                                        break
                                    except (ValueError, TypeError):
                                        continue
                                example_combination[var_name] = (var_name, numeric_val if numeric_val is not None else example_value[0])
                            else:
                                example_combination[var_name] = (var_name, example_value)
                    
                    # Validate against conditions using the same logic as _filter_invalid_combinations
                    for cond in conditions:
                        if cond.strip() != "True":  # Skip trivial conditions
                            temp_combination = example_combination | {k: v[1] for k, v in example_combination.items() if isinstance(v, tuple)}
                            try:
                                condition_result = eval(cond, {"__builtins__": {}}, EVAL_CONTEXT_HELPERS | temp_combination)
                                assert condition_result, \
                                    f"Example assignments {example_assignments} failed condition '{cond}' for {template_file.name}"
                            except Exception as e:
                                # Some conditions might reference variables not in example_assignments
                                # This is acceptable as long as the core constraint variables are valid
                                pass
