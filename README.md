# Cross Lingual Transfer of Reasoning



This projects contains the following data sources

| Name                | Language | Source                | Description                                                                                                                                                                                                                     |
| ------------------- | -------- | --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| gsm8k (100 samples) | eng      | gsm8k/ml-gsm-symbolic | A 100 samples of [GSM8k](https://huggingface.co/datasets/openai/gsm8k)                                                                                                                                                          |
| gsm-variants        | eng      | ml-gsm-symbolic       | Derived from from the paper "[Understanding the Limitations of Mathematical Reasoning in Large Language Models](https://machinelearning.apple.com/research/gsm-symbolic)". Which makes a symbolic variants of the GSM8k samples |
| gsm-ml-translated   | dan      |                       | A GPT 4o translation of gsm8k-variants                                                                                                                                                                                          |
| gsm-dan             | dan      |                       | A localized version of the machine translation                                                                                                                                                                                  |
| gsm-dan-variants    | dan      |                       | Symbolic variants of the gsm-dan version                                                                                                                                                                                        |
| kænguruen           | dan      |                       | A conversion of [Kænguruen](https://kaenguruen.dk/wp-content/uploads/2024/05/Kaenguruen-2024-8-9-svar-og-losning.pdf) to a machine readable format.                                                                             |


We expect to get a performance table somewhat like:

| Datasets (→)                                   | gsm8k | gsm-variants | gsm-dan | gsm-dan-variants | kænguruen |
| ---------------------------------------------- | ----- | ------------ | ------- | ---------------- | --------- |
| *Different samples*                            |       | ✔︎            |         | ✔︎                | ✔︎         |
| *Cross-lingual transfer*                       |       |              | ✔︎       | ✔︎                | ✔︎         |
| *Different distribution*                       |       |              |         |                  | ✔︎         |
| Degree of Generalization                       | 0     | 1            | 2       | 3                | 4         |
| **Models**                                     |       |              |         |                  |           |
| LLama8B 3.2                                    | 0.10  | 0.10         | 0.08    | 0.08             | 0.05      |
| LLama8B 3.2 +  RL (*gsm8k)                     | 0.60  | 0.58         | 0.40    | 0.38             | 0.20      |
| LLama8B 3.2 +  RL (*gsm8k + gsm-ml-translated) | 0.60  | 0.58         | 0.58    | 0.48             | 0.25      |
| LLama 70B ...                                  |       |              |         |                  |           |

\* We might decide not to train on gsm8k to keep it "held-out" and instead train on another available dataset and its translation.

Where degree of generalization is as follows:

1) Minimal generalization
2) Generalization to unseen samples, but similar distribution
3) Generalization to Danish but just translated, but similar distribution
4) Generalization to Danish as well as unseen samples, but similar distribution
5) Generalization to Danish, unseen samples and under a novel distribution

## Process

1) Initial translation with using chatgpt 4o (completed)
```
uv run scr/scripts/initial_translation.py
```

2) Manual correction and localization, this could for example include (partially completed)
    - fixing spelling errors
    - replacing names with natively plausible names
    - converting units to metrics

3) Generate samples (not completed)

- implement naive parser (for gsm8k, gsm-dan)
- implement symbolic variant parser (for *-variants)

4) Convert Kænguruen (not completed)

5) Build Training setup (partially completed)

Niklas has been working on it

6) Train Models (not/partially completed) 

7) Evaluation Setup (not completed)

