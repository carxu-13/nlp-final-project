"""
Main experiment runner. Evaluates GPT-4 on cryptarithm problems under
three prompting conditions: baseline, zero-shot CoT, and few-shot CoT.
"""

import json
import os
import re
import time
from datetime import datetime

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

from prompts import (
    BASELINE_SYSTEM, BASELINE_USER,
    ZERO_SHOT_COT_SYSTEM, ZERO_SHOT_COT_USER,
    FEW_SHOT_COT_SYSTEM, build_few_shot_user_prompt, load_few_shot_examples,
)

MODEL = "gpt-4-0125-preview"
TEMPERATURE = 0
MAX_TOKENS = 1024


def parse_mapping(response_text: str) -> dict[str, int]:
    """Extract letter=digit pairs from model response.
    Matches patterns like X=d or X: d."""
    pairs = re.findall(r'([A-Z])\s*[=:]\s*(\d)', response_text)
    mapping = {}
    for letter, digit in pairs:
        mapping[letter] = int(digit)
    return mapping


def parse_ground_truth(answer_str: str) -> dict[str, int]:
    """Parse ground truth answer string into a mapping.
    Handles formats like 'A=1, B=2' or 'A: 1' or \"{'A': 1, 'B': 2}\"."""
    # Match letter-digit pairs in various formats: A=1, A: 1, 'A': 1
    pairs = re.findall(r"'?([A-Z])'?\s*[=:]\s*(\d)", answer_str)
    return {letter: int(digit) for letter, digit in pairs}


def count_unique_letters(equation: str) -> int:
    return len(set(c for c in equation if c.isalpha() and c.isupper()))


def evaluate_mapping(predicted: dict, ground_truth: dict) -> dict:
    """Compute EMA (exact match accuracy) and PLA (per-letter accuracy)."""
    n_letters = len(ground_truth)
    digits_unique = len(set(predicted.values())) == len(predicted)

    # EMA: all letters correct and right count and unique digits
    ema = (
        len(predicted) == n_letters
        and digits_unique
        and all(predicted.get(k) == v for k, v in ground_truth.items())
    )

    # PLA: fraction of correctly predicted letters (among parsed ones)
    correct = sum(1 for k, v in predicted.items() if ground_truth.get(k) == v)
    pla = correct / n_letters if n_letters > 0 else 0.0

    return {
        "ema": ema,
        "pla": pla,
        "n_predicted": len(predicted),
        "n_ground_truth": n_letters,
        "digits_unique": digits_unique,
        "parse_failure": len(predicted) != n_letters or not digits_unique,
    }


def call_gpt4(client: OpenAI, system: str, user: str) -> str:
    """Make a single GPT-4 API call."""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )
    return response.choices[0].message.content


def run_condition(client, problems, condition_name, system_prompt, user_template_fn):
    """Run all problems under one prompting condition."""
    results = []
    for prob in tqdm(problems, desc=condition_name):
        equation = prob["question"]
        gt = parse_ground_truth(prob["answer"])

        user_msg = user_template_fn(equation)

        try:
            response_text = call_gpt4(client, system_prompt, user_msg)
        except Exception as e:
            print(f"  API error on '{equation}': {e}")
            response_text = ""
            time.sleep(5)

        predicted = parse_mapping(response_text)
        metrics = evaluate_mapping(predicted, gt)

        results.append({
            "equation": equation,
            "tier": prob.get("tier", "unknown"),
            "condition": condition_name,
            "ground_truth": gt,
            "predicted": predicted,
            "response_text": response_text,
            **metrics,
        })

    return results


def main():
    load_dotenv()
    client = OpenAI()  # reads OPENAI_API_KEY from env

    # Load data
    with open("data/test_set.json") as f:
        problems = json.load(f)
    print(f"Loaded {len(problems)} test problems")

    all_results = []

    # --- Condition 1: Baseline (no CoT) ---
    print("\n=== Condition 1: Baseline (no CoT) ===")
    baseline_results = run_condition(
        client, problems, "baseline",
        BASELINE_SYSTEM,
        lambda eq: BASELINE_USER.format(equation=eq),
    )
    all_results.extend(baseline_results)

    # --- Condition 2: Zero-shot CoT ---
    print("\n=== Condition 2: Zero-shot CoT ===")
    zeroshot_results = run_condition(
        client, problems, "zero_shot_cot",
        ZERO_SHOT_COT_SYSTEM,
        lambda eq: ZERO_SHOT_COT_USER.format(equation=eq),
    )
    all_results.extend(zeroshot_results)

    # --- Condition 3: Few-shot CoT ---
    print("\n=== Condition 3: Few-shot CoT ===")
    few_shot_examples = load_few_shot_examples()
    fewshot_results = run_condition(
        client, problems, "few_shot_cot",
        FEW_SHOT_COT_SYSTEM,
        lambda eq: build_few_shot_user_prompt(eq, few_shot_examples),
    )
    all_results.extend(fewshot_results)

    # Save all results
    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"results/experiment_{timestamp}.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results saved to {output_path}")

    # Print summary
    print_summary(all_results)


def print_summary(results):
    """Print aggregate metrics by condition and tier."""
    from collections import defaultdict

    by_condition = defaultdict(list)
    by_condition_tier = defaultdict(list)

    for r in results:
        by_condition[r["condition"]].append(r)
        by_condition_tier[(r["condition"], r["tier"])].append(r)

    print("\n" + "=" * 60)
    print("OVERALL RESULTS")
    print("=" * 60)
    for cond in ["baseline", "zero_shot_cot", "few_shot_cot"]:
        rs = by_condition[cond]
        if not rs:
            continue
        ema = sum(r["ema"] for r in rs) / len(rs)
        pla = sum(r["pla"] for r in rs) / len(rs)
        parse_fail = sum(r["parse_failure"] for r in rs) / len(rs)
        print(f"\n{cond}:")
        print(f"  EMA: {ema:.1%}  |  PLA: {pla:.1%}  |  Parse failures: {parse_fail:.1%}")

        for tier in ["easy", "medium", "hard"]:
            tier_rs = by_condition_tier[(cond, tier)]
            if not tier_rs:
                continue
            t_ema = sum(r["ema"] for r in tier_rs) / len(tier_rs)
            t_pla = sum(r["pla"] for r in tier_rs) / len(tier_rs)
            print(f"    {tier}: EMA={t_ema:.1%}, PLA={t_pla:.1%} (n={len(tier_rs)})")


if __name__ == "__main__":
    main()
