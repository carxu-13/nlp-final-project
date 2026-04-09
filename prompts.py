"""
Prompt templates for the three experimental conditions:
1. Baseline (no CoT)
2. Zero-shot CoT
3. Few-shot CoT
"""

import json

from cryptarithm_utils import format_mapping, parse_ground_truth


FINAL_ANSWER_FORMAT = "Final Answer: A=1, B=2, C=3"


def render_final_answer(answer: str) -> str:
    mapping = parse_ground_truth(answer)
    if not mapping:
        return f"Final Answer: {answer}"
    return f"Final Answer: {format_mapping(mapping)}"


BASELINE_SYSTEM = (
    "You are solving a cryptarithm (verbal arithmetic) puzzle. "
    "Each letter represents a unique digit (0-9). "
    "No leading letter of any word can be zero. "
    f"Return exactly one line in this format: {FINAL_ANSWER_FORMAT}. "
    "Do not include reasoning or any other text."
)

BASELINE_USER = (
    "Solve this cryptarithm. Each letter stands for a unique digit (0-9). "
    "No word can have a leading zero.\n\n"
    "{equation}\n\n"
    f"Return exactly one line in this format: {FINAL_ANSWER_FORMAT}"
)

ZERO_SHOT_COT_SYSTEM = (
    "You are solving a cryptarithm (verbal arithmetic) puzzle. "
    "Each letter represents a unique digit (0-9). "
    "No leading letter of any word can be zero. "
    "Use column-by-column reasoning, track carries, and backtrack if a tentative "
    "assignment leads to a contradiction. "
    f"End with a separate final line in this exact format: {FINAL_ANSWER_FORMAT}"
)

ZERO_SHOT_COT_USER = (
    "Solve this cryptarithm. Each letter stands for a unique digit (0-9). "
    "No word can have a leading zero.\n\n"
    "{equation}\n\n"
    "Think step by step. Use carries, contradictions, and backtracking when needed. "
    "Only put the complete letter-to-digit mapping on the final line.\n"
    f"{FINAL_ANSWER_FORMAT}"
)

FEW_SHOT_COT_SYSTEM = (
    "You are solving cryptarithm (verbal arithmetic) puzzles. "
    "Each letter represents a unique digit (0-9). "
    "No leading letter of any word can be zero. "
    "Solve them by analyzing column constraints from right to left, "
    "tracking carries, eliminating impossible assignments, and explicitly "
    "backtracking or brute-forcing the remaining possibilities when needed. "
    f"The final line must use this exact format: {FINAL_ANSWER_FORMAT}"
)


def build_few_shot_user_prompt(equation: str, examples: list[dict]) -> str:
    """Build the few-shot CoT prompt with worked examples."""
    prompt_parts = [
        "Here are some worked examples of solving cryptarithm puzzles:\n"
    ]

    for i, ex in enumerate(examples, 1):
        prompt_parts.append(f"--- Example {i} ---")
        prompt_parts.append(f"Problem: {ex['question']}")
        prompt_parts.append(f"Reasoning:\n{ex['walkthrough']}")
        prompt_parts.append(f"{render_final_answer(ex['answer'])}\n")

    prompt_parts.append("--- Now solve this problem ---")
    prompt_parts.append(f"Problem: {equation}")
    prompt_parts.append(
        "\nSolve step by step using column-by-column analysis from right to left, "
        "tracking carries, and eliminating contradictions. If a partial assignment "
        "fails, say so briefly and backtrack. You may brute-force the remaining "
        "consistent cases once the constraints are tight.\n"
        "Write your reasoning first. Then end with exactly one separate final line:\n"
        f"{FINAL_ANSWER_FORMAT}"
    )

    return "\n".join(prompt_parts)


def load_few_shot_examples(path: str = "data/few_shot_examples_with_walkthroughs.json") -> list[dict]:
    """Load the manually-constructed few-shot examples with walkthroughs."""
    with open(path) as f:
        return json.load(f)
