"""
Main experiment runner. Evaluates a chat model on cryptarithm problems
under three prompting conditions: baseline, zero-shot CoT, and few-shot CoT.
"""

import json
import os
import time
from datetime import datetime

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

from cryptarithm_utils import (
    count_unique_letters,
    has_final_answer_label,
    letters_in_equation,
    mapping_has_unique_digits,
    mapping_satisfies_equation,
    parse_ground_truth,
    parse_mapping_text,
)
from prompts import (
    BASELINE_SYSTEM,
    BASELINE_USER,
    FEW_SHOT_COT_SYSTEM,
    ZERO_SHOT_COT_SYSTEM,
    ZERO_SHOT_COT_USER,
    build_few_shot_user_prompt,
    load_few_shot_examples,
)

# Load .env before reading config constants.
load_dotenv()

#MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4-mini-2026-03-17")
#MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-2024-08-06")
MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4-2026-03-05")
TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0"))
MAX_TOKENS = int(os.getenv("OPENAI_MAX_TOKENS", "1024"))
TEST_SET_PATH = os.getenv("TEST_SET_PATH", "data/test_set.json")
EXPERIMENT_LIMIT = os.getenv("EXPERIMENT_LIMIT")


def model_uses_max_completion_tokens() -> bool:
    """gpt-5+ style chat models require max_completion_tokens instead of max_tokens."""
    override = os.getenv("OPENAI_USE_MAX_COMPLETION_TOKENS")
    if override is not None and override.strip() != "":
        return override.strip().lower() in ("1", "true", "yes")
    m = MODEL.lower()
    return "gpt-5" in m


def _usage_from_response(response) -> dict[str, int]:
    out = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    u = getattr(response, "usage", None)
    if u is None:
        return out
    pt = getattr(u, "prompt_tokens", None)
    ct = getattr(u, "completion_tokens", None)
    tt = getattr(u, "total_tokens", None)
    out["prompt_tokens"] = int(pt or 0)
    out["completion_tokens"] = int(ct or 0)
    if tt is not None:
        out["total_tokens"] = int(tt)
    else:
        out["total_tokens"] = out["prompt_tokens"] + out["completion_tokens"]
    return out


def evaluate_mapping(equation: str, predicted: dict[str, int], ground_truth: dict[str, int]) -> dict:
    """Compute EMA (exact match accuracy) and PLA (per-letter accuracy)."""
    n_letters = len(ground_truth)
    digits_unique = mapping_has_unique_digits(predicted)
    equation_satisfied = mapping_satisfies_equation(equation, predicted)
    missing_letters = sorted(set(ground_truth) - set(predicted))
    extra_letters = sorted(set(predicted) - set(ground_truth))

    ema = (
        len(predicted) == n_letters
        and digits_unique
        and all(predicted.get(k) == v for k, v in ground_truth.items())
    )

    parse_failure = (
        len(predicted) != n_letters or not digits_unique or bool(extra_letters)
    )

    correct = sum(1 for k, v in predicted.items() if ground_truth.get(k) == v)
    if parse_failure:
        pla = correct / len(predicted) if len(predicted) > 0 else 0.0
    else:
        pla = correct / n_letters if n_letters > 0 else 0.0

    return {
        "ema": ema,
        "pla": pla,
        "n_predicted": len(predicted),
        "n_ground_truth": n_letters,
        "digits_unique": digits_unique,
        "equation_satisfied": equation_satisfied,
        "missing_letters": missing_letters,
        "extra_letters": extra_letters,
        "parse_failure": parse_failure,
    }


def call_model(client: OpenAI, system: str, user: str) -> tuple[str, dict[str, int]]:
    """Make a single chat completion call. Returns (text, usage dict from API)."""
    kwargs: dict = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": TEMPERATURE,
    }
    if model_uses_max_completion_tokens():
        kwargs["max_completion_tokens"] = MAX_TOKENS
    else:
        kwargs["max_tokens"] = MAX_TOKENS
    response = client.chat.completions.create(**kwargs)
    text = response.choices[0].message.content or ""
    return text, _usage_from_response(response)


def run_condition(client, problems, condition_name, system_prompt, user_template_fn, token_totals=None):
    """Run all problems under one prompting condition."""
    results = []
    for prob in tqdm(problems, desc=condition_name):
        equation = prob["question"]
        gt = parse_ground_truth(prob["answer"])
        expected_letters = letters_in_equation(equation)

        user_msg = user_template_fn(equation)

        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        try:
            response_text, usage = call_model(client, system_prompt, user_msg)
        except Exception as e:
            print(f"  API error on '{equation}': {e}")
            response_text = ""
            time.sleep(5)

        if token_totals is not None:
            token_totals["prompt_tokens"] += usage["prompt_tokens"]
            token_totals["completion_tokens"] += usage["completion_tokens"]
            token_totals["total_tokens"] += usage["total_tokens"]

        predicted = parse_mapping_text(response_text, expected_letters=expected_letters)
        metrics = evaluate_mapping(equation, predicted, gt)

        results.append({
            "equation": equation,
            "tier": prob.get("tier", "unknown"),
            "condition": condition_name,
            "ground_truth": gt,
            "predicted": predicted,
            "response_text": response_text,
            "final_answer_present": has_final_answer_label(response_text),
            "num_unique_letters": count_unique_letters(equation),
            **usage,
            **metrics,
        })

    return results


def main():
    client = OpenAI()

    with open(TEST_SET_PATH) as f:
        problems = json.load(f)
    print(f"Loaded {len(problems)} test problems from {TEST_SET_PATH}")

    if EXPERIMENT_LIMIT:
        problems = problems[: int(EXPERIMENT_LIMIT)]
        print(f"Running on first {len(problems)} problems (EXPERIMENT_LIMIT={EXPERIMENT_LIMIT})")
    else:
        print(f"Running on all {len(problems)} problems")

    print(f"Model: {MODEL}")

    all_results = []
    token_totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    print("\n=== Condition 1: Baseline (no CoT) ===")
    baseline_results = run_condition(
        client,
        problems,
        "baseline",
        BASELINE_SYSTEM,
        lambda eq: BASELINE_USER.format(equation=eq),
        token_totals=token_totals,
    )
    all_results.extend(baseline_results)

    print("\n=== Condition 2: Zero-shot CoT ===")
    zeroshot_results = run_condition(
        client,
        problems,
        "zero_shot_cot",
        ZERO_SHOT_COT_SYSTEM,
        lambda eq: ZERO_SHOT_COT_USER.format(equation=eq),
        token_totals=token_totals,
    )
    all_results.extend(zeroshot_results)

    print("\n=== Condition 3: Few-shot CoT ===")
    few_shot_examples = load_few_shot_examples()
    fewshot_results = run_condition(
        client,
        problems,
        "few_shot_cot",
        FEW_SHOT_COT_SYSTEM,
        lambda eq: build_few_shot_user_prompt(eq, few_shot_examples),
        token_totals=token_totals,
    )
    all_results.extend(fewshot_results)

    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"results/experiment_{timestamp}.json"
    with open(output_path, "w") as f:
        json.dump(
            {
                "token_usage_total": token_totals,
                "results": all_results,
            },
            f,
            indent=2,
        )
    print(f"\nAll results saved to {output_path}")
    print(
        "\nTotal API token usage: "
        f"prompt={token_totals['prompt_tokens']:,}  "
        f"completion={token_totals['completion_tokens']:,}  "
        f"total={token_totals['total_tokens']:,}"
    )

    print_summary(all_results)


def print_summary(results):
    """Print aggregate metrics by condition and tier."""
    from collections import defaultdict

    by_condition = defaultdict(list)
    by_condition_tier = defaultdict(list)

    for result in results:
        by_condition[result["condition"]].append(result)
        by_condition_tier[(result["condition"], result["tier"])].append(result)

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
        solved = sum(r["equation_satisfied"] for r in rs) / len(rs)
        final_answer = sum(r["final_answer_present"] for r in rs) / len(rs)
        print(f"\n{cond}:")
        print(
            f"  EMA: {ema:.1%}  |  PLA: {pla:.1%}  |  "
            f"Equation satisfied: {solved:.1%}  |  "
            f"Final Answer line: {final_answer:.1%}  |  Parse failures: {parse_fail:.1%}"
        )

        for tier in ["easy", "medium", "hard"]:
            tier_rs = by_condition_tier[(cond, tier)]
            if not tier_rs:
                continue
            t_ema = sum(r["ema"] for r in tier_rs) / len(tier_rs)
            t_pla = sum(r["pla"] for r in tier_rs) / len(tier_rs)
            t_solved = sum(r["equation_satisfied"] for r in tier_rs) / len(tier_rs)
            print(
                f"    {tier}: EMA={t_ema:.1%}, PLA={t_pla:.1%}, "
                f"EqSolved={t_solved:.1%} (n={len(tier_rs)})"
            )


if __name__ == "__main__":
    main()
