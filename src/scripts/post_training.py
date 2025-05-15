# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>
# adapted from the following resources:
# - https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_(8B)-GRPO.ipynb#scrollTo=DkIvEkIIkEyB
# - https://github.com/Mohammadjafari80/GSM8K-RLVR
# --------------------------------------------------------------------------------------
# This script is used to perform post-training for Reasoning on with GRPO.
#
# Potential models to use:
# meta-llama/meta-Llama-3.1-8B-Instruct
# meta-llama/Llama-3.2-1B
# allenai/OLMo-1B-hf
# PleIAs/Pleias-1.2b-Preview
# google/gemma-3-1b-pt
#
# Potential datasets to post-train on:
# deepmind/math_dataset

import re
import argparse
from unsloth import FastModel
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Post-training for Reasoning with GRPO."
    )
    parser.add_argument("--model_name", type=str, default="unsloth/gemma-3-1b-it")
    parser.add_argument(
        "--training_dataset", type=str, default="deepmind/math_dataset"
    )

    return parser.parse_args()

args = parse_args()

model_name = args.model_name
training_dataset = args.training_dataset
output_dir = f"outputs/models/{model_name}"

max_prompt_length = 256
max_seq_length = 1024

# Loading model and tokenizer ----------------------------------------------------------
model, tokenizer = FastModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    load_in_4bit=False,
    load_in_8bit=False,
    full_finetuning=False,
)

model = FastModel.get_peft_model(
    model,
    finetune_vision_layers=False,
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=8,
    lora_alpha=8,
    lora_dropout=0,
    bias="none",
    random_state=3407,
)

# Loading Training dataset -------------------------------------------------------------
dataset = load_dataset(training_dataset, "algebra__linear_1d", split="train")


def extract_answer(text):
    return re.sub(r"^b'(.*)\\n'$", r"\1", text) # Remove line breat and bytes symbols

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
        "answer": extract_answer(x["answer"]),
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
    """Check if the completion matches the format exactly."""
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
    """Check if the completion matches the format approximately."""
    scores = []
    for completion in completions:
        score = 0
        response = completion[0]["content"]
        # Count how many keywords are seen - we penalize if too many!
        # If we see 1, then plus some points!
        score += 0.5 if response.count(reasoning_start) == 1 else -0.5
        score += 0.5 if response.count(reasoning_end) == 1 else -0.5
        score += 0.5 if response.count(solution_start) == 1 else -0.5
        score += 0.5 if response.count(solution_end) == 1 else -0.5
        scores.append(score)
    return scores


def check_answer(prompts, completions, answer, **kwargs):
    question = prompts[0][-1]["content"]
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
        # Match if spaces are seen
        elif guess.strip() == true_answer.strip():
            score += 1.5
        else:
            # We also reward it if the answer is close via ratios!
            # Ie if the answer is within some range, reward it!
            try:
                ratio = float(guess) / float(true_answer)
                if ratio >= 0.9 and ratio <= 1.1:
                    score += 0.5
                elif ratio >= 0.8 and ratio <= 1.2:
                    score += 0.25
                else:
                    score -= 1.0  # Penalize wrong answers
            except:
                score -= 0.5  # Penalize
        scores.append(score)
    return scores


def check_numbers(prompts, completions, answer, **kwargs):
    """Reward model for getting the numbers right."""
    question = prompts[0][-1]["content"]
    responses = [completion[0]["content"] for completion in completions]

    extracted_responses = [
        guess.group(1) if (guess := match_numbers.search(r)) is not None else None
        for r in responses
    ]

    scores = []
    print(
        "*" * 20,
        f"Question:\n{question}",
        f"\nAnswer:\n{answer[0]}",
        f"\nResponse:\n{responses[0]}",
        f"\nExtracted:\n{extracted_responses[0]}",
    )
    for guess, true_answer in zip(extracted_responses, answer):
        if guess is None:
            scores.append(0)
            continue
        # Convert to numbers
        try:
            true_answer = float(true_answer.strip())
            guess = float(guess.strip())
            scores.append(1.5 if guess == true_answer else 0.0)
        except:
            scores.append(0)
            continue
    return scores

# Training the model -------------------------------------------------------------------

training_args = GRPOConfig(
    learning_rate=5e-6,
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.1,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    optim="adamw_torch_fused",
    logging_steps=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,  # Increase to 4 for smoother training
    num_generations=4,  # Decrease if out of memory
    max_prompt_length=max_prompt_length,
    max_completion_length=max_seq_length - max_prompt_length,
    # num_train_epochs = 1, # Set to 1 for a full training run
    max_steps=50,
    save_steps=50,
    max_grad_norm=0.1,
    report_to="wandb",
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
