from pathlib import Path
from typing import Dict, List, Any

def parse_text_file(filepath: str) -> Dict[str, Any]:
    """
    Parse a single text file containing a math problem.
    
    Args:
        filepath: Absolute path to the text file
        data_dir: Base directory for calculating relative path
        
    Returns:
        A dictionary with the parsed sections
    """
    problem_data = {}
    problem_data["file"] = filepath
    
    with open(filepath, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Split the content by "---" separator
    question, options, answer, solution, percent_correct = content.split("---")
    problem_data["question"] = question.strip()
    problem_data["options"] = options.replace("Valgmuligheder:", "").strip()
    problem_data["answer"] = answer.replace("Svar:", "").strip()
    problem_data["solution"] = solution.replace("Løsning:", "").strip()
    problem_data["percentage_correct"] = percent_correct.replace("Procent rigtige:", "").replace("%", "").strip()
                
    return problem_data


def load_kaenguruen_data(directory_path: str) -> List[Dict[str, Any]]:
    """
    Load all problem data from text files in the specified directory.
    
    Args:
        directory_path: Path to the directory containing the text files
        
    Returns:
        A list of dictionaries, each containing parsed problem data
    """
    # Convert to Path object for easier recursive traversal
    dir_path = Path(directory_path)

    return [parse_text_file(file_path) for file_path in dir_path.glob("**/*.txt")]
