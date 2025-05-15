#!/usr/bin/env python3
"""
Script to parse Kaenguruen math problems from text files into a list of dictionaries.
The dictionaries can be later dumped to JSON files.
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any


def parse_text_file(filepath: str, data_dir: str) -> Dict[str, Any]:
    """
    Parse a single text file containing a math problem.
    
    Args:
        filepath: Absolute path to the text file
        data_dir: Base directory for calculating relative path
        
    Returns:
        A dictionary with the parsed sections
    """
    # Initialize an empty dictionary to store problem data
    problem_data = {}
    
    # Add the relative path to the dictionary
    rel_path = os.path.relpath(filepath, data_dir)
    problem_data["file"] = rel_path
    
    try:
        # Read the file content
        with open(filepath, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Split the content by "---" separator
        sections = content.split("---")
        
        # Clean and assign each section to the appropriate key
        if len(sections) >= 1:
            # Remove the filepath comment if present
            question_text = sections[0]
            if "// filepath:" in question_text:
                question_text = question_text.split("\n", 1)[1]
            problem_data["question"] = question_text.strip()
        
        if len(sections) >= 2:
            # Extract options, removing the header
            options_text = sections[1]
            if "Valgmuligheder:" in options_text:
                options_text = options_text.replace("Valgmuligheder:", "").strip()
            problem_data["options"] = options_text.strip()
        
        if len(sections) >= 3:
            # Extract answer, removing the header
            answer_text = sections[2]
            if "Svar:" in answer_text:
                answer_text = answer_text.replace("Svar:", "").strip()
            problem_data["answer"] = answer_text.strip()
        
        if len(sections) >= 4:
            # Extract solution, removing the header
            solution_text = sections[3]
            if "Løsning:" in solution_text:
                solution_text = solution_text.replace("Løsning:", "").strip()
            problem_data["solution"] = solution_text.strip()
        
        if len(sections) >= 5:
            # Extract percentage correct, removing the header
            percent_text = sections[4]
            if "Procent rigtige:" in percent_text:
                percent_text = percent_text.replace("Procent rigtige:", "").strip()
                # Remove the percentage sign if present and convert to float
                percent_text = percent_text.replace("%", "").strip()
                try:
                    problem_data["percentage_correct"] = float(percent_text)
                except ValueError:
                    problem_data["percentage_correct"] = percent_text
            else:
                problem_data["percentage_correct"] = percent_text.strip()
                
    except Exception as e:
        print(f"Error parsing file {filepath}: {e}")
        return {"error": str(e), "file": rel_path}
    
    return problem_data


def load_data_from_directory(directory_path: str) -> List[Dict[str, Any]]:
    """
    Load all problem data from text files in the specified directory.
    
    Args:
        directory_path: Path to the directory containing the text files
        
    Returns:
        A list of dictionaries, each containing parsed problem data
    """
    problems = []
    data_dir = directory_path
    
    # Convert to Path object for easier recursive traversal
    dir_path = Path(directory_path)
    
    # Find all text files recursively
    for file_path in dir_path.glob("**/*.txt"):
        # Parse the file and add to the list
        problem_data = parse_text_file(str(file_path), data_dir)
        problems.append(problem_data)
    
    return problems


def save_to_json_files(problems: List[Dict[str, Any]], output_dir: str) -> None:
    """
    Save each problem to a separate JSON file with enumerated filenames.
    
    Args:
        problems: List of problem dictionaries
        output_dir: Directory to save the JSON files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save each problem to a separate file
    for i, problem in enumerate(problems):
        output_file = os.path.join(output_dir, f"{i:04d}.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(problem, f, ensure_ascii=False, indent=2)
    
    print(f"Saved {len(problems)} problems to {output_dir} directory")


def main():
    """Main function to parse command line arguments and execute the script."""
    parser = argparse.ArgumentParser(description="Parse Kaenguruen math problems from text files")
    parser.add_argument("--data_dir", type=str, required=True, 
                        help="Directory containing the text files")
    parser.add_argument("--output_dir", type=str, default="kaenguruen_problems",
                        help="Output directory for JSON files (default: kaenguruen_problems)")
    parser.add_argument("--pretty", action="store_true", 
                        help="Print a formatted preview of the first problem")
    
    args = parser.parse_args()
    
    # Load data from the specified directory
    problems = load_data_from_directory(args.data_dir)
    
    # Save each problem to a separate JSON file
    save_to_json_files(problems, args.output_dir)
    
    # Print a preview if requested
    if args.pretty and problems:
        print("\nPreview of the first parsed problem:")
        preview = json.dumps(problems[0], ensure_ascii=False, indent=2)
        print(preview)
    
    print(f"Total problems parsed: {len(problems)}")


if __name__ == "__main__":
    main()
