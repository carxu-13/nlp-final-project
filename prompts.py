"""
Prompt templates for the three experimental conditions:
1. Baseline (no CoT)
2. Zero-shot CoT
3. Few-shot CoT
"""

import json


BASELINE_SYSTEM = (
    "You are solving a cryptarithm (verbal arithmetic) puzzle. "
    "Each letter represents a unique digit (0-9). "
    "No leading letter of any word can be zero. "
    "Return ONLY the letter-to-digit mapping in the format: A=1, B=2, C=3, ..."
)

BASELINE_USER = (
    "Solve this cryptarithm. Each letter stands for a unique digit (0-9). "
    "No word can have a leading zero.\n\n"
    "{equation}\n\n"
    "Provide the mapping for every letter as: X=d for each letter."
)

ZERO_SHOT_COT_SYSTEM = (
    "You are solving a cryptarithm (verbal arithmetic) puzzle. "
    "Each letter represents a unique digit (0-9). "
    "No leading letter of any word can be zero."
)

ZERO_SHOT_COT_USER = (
    "Solve this cryptarithm. Each letter stands for a unique digit (0-9). "
    "No word can have a leading zero.\n\n"
    "{equation}\n\n"
    "Think step by step. "
    "After your reasoning, provide the final mapping for every letter as: X=d for each letter."
)

FEW_SHOT_COT_SYSTEM = (
    "You are solving cryptarithm (verbal arithmetic) puzzles. "
    "Each letter represents a unique digit (0-9). "
    "No leading letter of any word can be zero. "
    "Solve them by analyzing column constraints from right to left, "
    "tracking carries, and eliminating impossible assignments."
)


def build_few_shot_user_prompt(equation: str, examples: list[dict]) -> str:
    """Build the few-shot CoT prompt with worked examples."""
    prompt_parts = [
        "Here are some worked examples of solving cryptarithm puzzles:\n"
    ]

    for i, ex in enumerate(examples, 1):
        prompt_parts.append(f"--- Example {i} ---")
        prompt_parts.append(f"Problem: {ex['question']}")
        prompt_parts.append(f"Solution walkthrough:\n{ex['walkthrough']}")
        prompt_parts.append(f"Answer: {ex['answer']}\n")

    prompt_parts.append("--- Now solve this problem ---")
    prompt_parts.append(f"Problem: {equation}")
    prompt_parts.append(
        "\nSolve step by step using column-by-column analysis from right to left, "
        "tracking carries, and eliminating contradictions. "
        "After your reasoning, provide the final mapping for every letter as: X=d for each letter."
    )

    return "\n".join(prompt_parts)


def load_few_shot_examples(path: str = "data/few_shot_examples_with_walkthroughs.json") -> list[dict]:
    """Load the manually-constructed few-shot examples with walkthroughs."""
    with open(path) as f:
        return json.load(f)
