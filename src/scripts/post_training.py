"""
Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

Adapted from the following resources:
- https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_(8B)-GRPO.ipynb

----------------------------------------------------------------------------------------

This script is used to perform post-training for Reasoning on with GRPO.

Potential models to use :
-------------------------

meta-llama/meta-Llama-3.1-8B-Instruct
meta-llama/Llama-3.2-1B
allenai/OLMo-1B-hf
PleIAs/Pleias-1.2b-Preview
google/gemma-3-1b-pt

"""

import re
import argparse
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
import wandb
from dotenv import load_dotenv
from huggingface_hub import login
from pathlib import Path
import os

# Login --------------------------------------------------------------------------------

load_dotenv(Path.cwd() / ".env")  # Load .env file

# Hugging Face
token = os.getenv("huggingface_api_key")
login(token=token)

# Weights & Biases
token = os.getenv("wandb_api_key")
wandb.login(key=token)


# Argument parser ----------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Post-training for Reasoning with GRPO."
    )
    parser.add_argument(
        "--model_name", type=str, default="meta-llama/Llama-3.2-3B-Instruct"
    )

    return parser.parse_args()


args = parse_args()

model_name = args.model_name
output_dir = f"outputs/models/{model_name}"

max_seq_length = 1024
lora_rank = 32  # Larger rank = smarter, but slower

# Loading model and tokenizer ----------------------------------------------------------
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    load_in_4bit=True,  # False for LoRA 16bit
    fast_inference=True,  # Enable vLLM fast inference
    max_lora_rank=lora_rank,
    gpu_memory_utilization=0.6,  # Reduce if out of memory
)

model = FastLanguageModel.get_peft_model(
    model,
    r=lora_rank,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],  # Remove QKVO if out of memory
    lora_alpha=lora_rank,
    use_gradient_checkpointing="unsloth",  # Enable long context finetuning
    random_state=3407,
)

# Loading Training dataset -------------------------------------------------------------
# no need to filter for translatet questions, as they are all from the test set
dataset = load_dataset("openai/gsm8k", "main", split="train")


def extract_hash_answer(text):
    if "####" not in text:
        return None
    return text.split("####")[1].strip()


reasoning_start = "<start_working_out>"
reasoning_end = "<end_working_out>"
solution_start = "<SOLUTION>"
solution_end = "</SOLUTION>"

system_prompt = f"""You are given a problem.
Think about the problem and provide your working out.
Place it between {reasoning_start} and {reasoning_end}.
Then, provide your solution between {solution_start}{solution_end}"""


# We need to format the dataset to match the system prompt
dataset = dataset.map(
    lambda x: {
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": x["question"]},
        ],
        "answer": extract_hash_answer(x["answer"]),
    }
)


# Regular expressions ------------------------------------------------------------------

# Regular expressions to match the format of the answer
match_format = re.compile(
    rf"^[\s]{{0,}}"
    rf"{reasoning_start}.+?{reasoning_end}.*?"
    rf"{solution_start}(.+?){solution_end}"
    rf"[\s]{{0,}}$",
    flags=re.MULTILINE | re.DOTALL,
)

# Regular expressions to extract the numbers from the solution
match_numbers = re.compile(
    rf"{solution_start}.*?([\d\.]{{1,}})", flags=re.MULTILINE | re.DOTALL
)

# Reward functions ---------------------------------------------------------------------


def match_format_exactly(completions, **kwargs):
    scores = []
    for completion in completions:
        score = 0
        response = completion[0]["content"]
        # Match if format is seen exactly!
        if match_format.search(response) is not None:
            score += 3.0
        scores.append(score)
    return scores


def match_format_approximately(completions, **kwargs):
    scores = []
    for completion in completions:
        score = 0
        response = completion[0]["content"]
        # Count how many keywords are seen - we penalize if too many!
        # If we see 1, then plus some points!
        score += 0.5 if response.count(reasoning_start) == 1 else -1.0
        score += 0.5 if response.count(reasoning_end) == 1 else -1.0
        score += 0.5 if response.count(solution_start) == 1 else -1.0
        score += 0.5 if response.count(solution_end) == 1 else -1.0
        scores.append(score)
    return scores


def check_answer(prompts, completions, answer, **kwargs):
    responses = [completion[0]["content"] for completion in completions]

    extracted_responses = [
        guess.group(1) if (guess := match_format.search(r)) is not None else None
        for r in responses
    ]

    scores = []
    for guess, true_answer in zip(extracted_responses, answer):
        score = 0
        if guess is None:
            scores.append(0)
            continue
        # Correct answer gets 3 points!
        if guess == true_answer:
            score += 3.0
        # Match if spaces are seen, but less reward
        elif guess.strip() == true_answer.strip():
            score += 1.5
        else:
            # We also reward it if the answer is close via ratios!
            # Ie if the answer is within some range, reward it!
            try:
                ratio = float(guess) / float(true_answer)
                if ratio >= 0.9 and ratio <= 1.1:
                    score += 1.0
                elif ratio >= 0.8 and ratio <= 1.2:
                    score += 0.5
                else:
                    score -= 1.5  # Penalize wrong answers
            except (ValueError, TypeError):
                score -= 1.5  # Penalize
        scores.append(score)
    return scores


match_numbers = re.compile(
    solution_start + r".*?([\d\.\,]{1,})", flags=re.MULTILINE | re.DOTALL
)

global PRINTED_TIMES
PRINTED_TIMES = 0
global PRINT_EVERY_STEPS
PRINT_EVERY_STEPS = 5


def check_numbers(prompts, completions, answer, **kwargs):
    question = prompts[0][-1]["content"]
    responses = [completion[0]["content"] for completion in completions]

    extracted_responses = [
        guess.group(1) if (guess := match_numbers.search(r)) is not None else None
        for r in responses
    ]

    scores = []
    # Print only every few steps
    global PRINTED_TIMES
    global PRINT_EVERY_STEPS
    if PRINTED_TIMES % PRINT_EVERY_STEPS == 0:
        print(
            "*" * 20,
            f"Question:\n{question}",
            f"\nAnswer:\n{answer[0]}",
            f"\nResponse:\n{responses[0]}",
            f"\nExtracted:\n{extracted_responses[0]}",
        )
    PRINTED_TIMES += 1

    for guess, true_answer in zip(extracted_responses, answer):
        if guess is None:
            scores.append(0)
            continue
        # Convert to numbers
        try:
            true_answer = float(true_answer.strip())
            # Remove commas like in 123,456
            guess = float(guess.strip().replace(",", ""))
            scores.append(1.5 if guess == true_answer else -0.5)
        except (ValueError, TypeError):
            scores.append(0)
            continue
    return scores


# Get the maximum prompt length so we don't accidentally truncate it!
max_prompt_length = max(
    dataset.map(
        lambda x: {
            "tokens": tokenizer.apply_chat_template(
                x["prompt"], add_generation_prompt=True, tokenize=True
            )
        },
        batched=True,
    ).map(lambda x: {"length": len(x["tokens"])})["length"]
)
max_prompt_length += 1  # + 1 just in case!

# Training the model -------------------------------------------------------------------

training_args = GRPOConfig(
    learning_rate=5e-6,
    weight_decay=0.01,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    optim="adamw_8bit",
    logging_steps=25,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,  # Increase to 4 for smoother training
    num_generations=4,  # Decrease if out of memory
    max_prompt_length=max_prompt_length,
    max_completion_length=max_seq_length - max_prompt_length,
    num_train_epochs=5,
    max_steps=-1,  # Set to -1 for a full training run
    save_steps=250,
    max_grad_norm=1.0,
    report_to="wandb",  # Can use Weights & Biases
    output_dir="outputs",
)


trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        match_format_exactly,
        match_format_approximately,
        check_answer,
        check_numbers,
    ],
    args=training_args,
    train_dataset=dataset,
)
trainer.train()

# Local saving -------------------------------------------------------------------------
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

# Push to Hub --------------------------------------------------------------------------
model.push_to_hub(f"LegrandNico/{model_name.split('/')[1]}-GRPO")
tokenizer.push_to_hub(f"LegrandNico/{model_name.split('/')[1]}-GRPO")
