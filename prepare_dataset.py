"""
Step 1: Download and filter the cryptarithm dataset from HuggingFace,
stratify by difficulty, and sample 100 problems for evaluation.
"""

import json
import random
from datasets import load_dataset

random.seed(42)


def count_addends(equation: str) -> int:
    """Count addends from the bare puzzle string like 'TWO + TWO = FOUR'."""
    parts = equation.split("=")
    if len(parts) != 2:
        return -1
    left = parts[0]
    return left.count("+") + 1


def deduplicate(problems: list[dict]) -> list[dict]:
    """Remove duplicate puzzles (same equation)."""
    seen = set()
    unique = []
    for p in problems:
        key = p["puzzle"]
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


def main():
    print("Loading dataset from HuggingFace...")
    # Load all splits and combine
    ds_train = load_dataset("theblackcat102/cryptarithm", split="train")
    ds_test = load_dataset("theblackcat102/cryptarithm", split="test")
    ds_val = load_dataset("theblackcat102/cryptarithm", split="validate")

    all_rows = list(ds_train) + list(ds_test) + list(ds_val)
    print(f"Total rows across all splits: {len(all_rows)}")

    # Extract using metadata fields for accurate puzzle/letter info
    filtered = []
    for row in all_rows:
        metadata = row.get("metadata", {})
        puzzle = metadata.get("puzzle", "")
        answer = row.get("answer", "")
        difficulty_info = metadata.get("difficulty", {})
        num_letters = difficulty_info.get("num_letters", 0)

        if not puzzle or not answer:
            continue

        n_addends = count_addends(puzzle)
        if n_addends < 1 or n_addends > 2:
            continue

        if num_letters > 8:
            continue

        filtered.append({
            "puzzle": puzzle.strip(),
            "question": puzzle.strip(),  # use bare equation as the prompt input
            "answer": answer.strip(),
            "num_unique_letters": num_letters,
            "num_addends": n_addends,
        })

    # Deduplicate
    filtered = deduplicate(filtered)
    print(f"After filtering (≤2 addends, ≤8 unique letters) and dedup: {len(filtered)}")

    # Stratify by difficulty tier
    easy = [p for p in filtered if p["num_unique_letters"] <= 5]
    medium = [p for p in filtered if 6 <= p["num_unique_letters"] <= 7]
    hard = [p for p in filtered if p["num_unique_letters"] == 8]

    print(f"Easy (≤5 letters): {len(easy)}")
    print(f"Medium (6-7 letters): {len(medium)}")
    print(f"Hard (8 letters): {len(hard)}")

    # Sample according to paper: 34 easy, 33 medium, 33 hard
    random.shuffle(easy)
    random.shuffle(medium)
    random.shuffle(hard)

    # Reserve 3 problems for few-shot examples (1 from each tier for variety)
    few_shot_examples = [easy.pop(0), medium.pop(0), hard.pop(0)]

    n_easy = min(34, len(easy))
    n_medium = min(33, len(medium))
    n_hard = min(33, len(hard))

    sampled_easy = easy[:n_easy]
    sampled_medium = medium[:n_medium]
    sampled_hard = hard[:n_hard]

    # Add difficulty tier labels
    for p in sampled_easy:
        p["tier"] = "easy"
    for p in sampled_medium:
        p["tier"] = "medium"
    for p in sampled_hard:
        p["tier"] = "hard"
    for p in few_shot_examples:
        p["tier"] = "few_shot_example"

    test_set = sampled_easy + sampled_medium + sampled_hard
    random.shuffle(test_set)

    print(f"\nFinal test set: {len(test_set)} problems")
    print(f"  Easy: {n_easy}, Medium: {n_medium}, Hard: {n_hard}")
    print(f"Few-shot examples: {len(few_shot_examples)}")

    # Save
    with open("data/test_set.json", "w") as f:
        json.dump(test_set, f, indent=2)

    with open("data/few_shot_examples.json", "w") as f:
        json.dump(few_shot_examples, f, indent=2)

    print("\nSaved to data/test_set.json and data/few_shot_examples.json")


if __name__ == "__main__":
    import os
    os.makedirs("data", exist_ok=True)
    main()
