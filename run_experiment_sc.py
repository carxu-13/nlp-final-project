"""
Self-consistency experiment runner for cryptarithm evaluation.

For each puzzle and prompting condition, draws N stochastic samples (same prompt),
parses each mapping, then aggregates by plurality over canonicalized full mappings.
Samples with a successful parse (no parse_failure) vote first; if none, falls back
to voting among all non-empty parsed mappings.

Usage:
  SC_N=10 SC_TEMPERATURE=0.7 python run_experiment_sc.py

See also run_experiment.py for the standard single-sample protocol.
"""

from __future__ import annotations

import json
import os
import time
from collections import Counter
from datetime import datetime

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

load_dotenv()

from cryptarithm_utils import (
    count_unique_letters,
    has_final_answer_label,
    letters_in_equation,
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

from run_experiment import (
    MAX_TOKENS,
    MODEL,
    TEST_SET_PATH,
    _usage_from_response,
    evaluate_mapping,
    model_uses_max_completion_tokens,
)

EXPERIMENT_LIMIT = os.getenv("EXPERIMENT_LIMIT")

SC_N = int(os.getenv("SC_N", "4"))
SC_TEMPERATURE = float(os.getenv("SC_TEMPERATURE", "0.7"))
SC_SAVE_SAMPLE_RESPONSES = os.getenv("SC_SAVE_SAMPLE_RESPONSES", "").strip().lower() in (
    "1",
    "true",
    "yes",
)


def canonical_mapping_key(m: dict[str, int]) -> tuple[tuple[str, int], ...]:
    return tuple(sorted(m.items()))


def call_model_sample(
    client: OpenAI,
    system: str,
    user: str,
    temperature: float,
) -> tuple[str, dict[str, int]]:
    kwargs: dict = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
    }
    if model_uses_max_completion_tokens():
        kwargs["max_completion_tokens"] = MAX_TOKENS
    else:
        kwargs["max_tokens"] = MAX_TOKENS
    response = client.chat.completions.create(**kwargs)
    text = response.choices[0].message.content or ""
    return text, _usage_from_response(response)


def _mapping_key_str(k: tuple[tuple[str, int], ...]) -> str:
    if not k:
        return ""
    return ", ".join(f"{a}={b}" for a, b in k)


def aggregate_plurality(
    equation: str,
    samples: list[tuple[dict[str, int], dict]],
) -> tuple[dict[str, int], dict[str, object]]:
    """
    samples: list of (predicted_mapping, metrics_dict from evaluate_mapping).
    Returns (winner_mapping, vote_metadata).
    """
    valid = [(p, m) for p, m in samples if not m["parse_failure"]]
    pool: list[tuple[dict[str, int], dict]] = valid if valid else [(p, m) for p, m in samples if p]

    def pack_counts(counts: Counter) -> dict[str, int]:
        return {_mapping_key_str(k): c for k, c in counts.items()}

    if not pool:
        return {}, {
            "vote_counts": {},
            "winner_key": None,
            "winner_votes": 0,
            "pool": "none",
            "tie_break": None,
        }

    keys = [canonical_mapping_key(p) for p, _ in pool]
    counts = Counter(keys)
    max_votes = max(counts.values())
    leaders = [k for k, c in counts.items() if c == max_votes]

    pool_source = "valid_parse" if valid else "fallback_nonempty"

    if len(leaders) == 1:
        winner = dict(leaders[0])
        return winner, {
            "vote_counts": pack_counts(counts),
            "winner_key": [list(x) for x in leaders[0]],
            "winner_votes": max_votes,
            "pool": pool_source,
            "tie_break": None,
        }

    # Tie-break without ground truth: prefer mapping that satisfies the equation
    for k in sorted(leaders):
        cand = dict(k)
        if mapping_satisfies_equation(equation, cand):
            return cand, {
                "vote_counts": pack_counts(counts),
                "winner_key": [list(x) for x in k],
                "winner_votes": max_votes,
                "pool": pool_source,
                "tie_break": "equation_satisfied",
            }

    # Lexicographic tie-break on canonical key (deterministic)
    pick_key = sorted(leaders)[0]
    return dict(pick_key), {
        "vote_counts": pack_counts(counts),
        "winner_key": [list(x) for x in pick_key],
        "winner_votes": max_votes,
        "pool": pool_source,
        "tie_break": "lexicographic_key",
    }


def run_condition_sc(
    client: OpenAI,
    problems: list,
    condition_name: str,
    system_prompt: str,
    user_template_fn,
    token_totals: dict | None,
    n_samples: int,
    temperature: float,
) -> list[dict]:
    results = []
    for prob in tqdm(problems, desc=condition_name):
        equation = prob["question"]
        gt = parse_ground_truth(prob["answer"])
        expected_letters = letters_in_equation(equation)
        user_msg = user_template_fn(equation)

        usage_sum = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        sample_records: list[dict] = []
        parsed_chain: list[tuple[dict[str, int], dict]] = []
        # Parallel to parsed_chain: response text for each successful API call (same order).
        response_texts_ok: list[str] = []
        num_api_errors = 0

        for i in range(n_samples):
            try:
                response_text, usage = call_model_sample(
                    client, system_prompt, user_msg, temperature
                )
            except Exception as e:
                num_api_errors += 1
                print(f"  [{condition_name}] API error sample {i + 1}/{n_samples} on '{equation}': {e}")
                sample_records.append(
                    {
                        "sample_index": i,
                        "api_error": True,
                        "api_error_message": str(e),
                    }
                )
                time.sleep(2)
                continue

            for k in usage_sum:
                usage_sum[k] += usage[k]

            response_texts_ok.append(response_text)
            predicted = parse_mapping_text(response_text, expected_letters=expected_letters)
            metrics_i = evaluate_mapping(equation, predicted, gt)
            parsed_chain.append((predicted, metrics_i))

            rec: dict = {
                "sample_index": i,
                "api_error": False,
                "parse_failure": metrics_i["parse_failure"],
                "equation_satisfied": metrics_i["equation_satisfied"],
                "ema": metrics_i["ema"],
                "prompt_tokens": usage["prompt_tokens"],
                "completion_tokens": usage["completion_tokens"],
                "total_tokens": usage["total_tokens"],
            }
            if SC_SAVE_SAMPLE_RESPONSES:
                rec["response_text"] = response_text
            sample_records.append(rec)

        winner, vote_meta = aggregate_plurality(equation, parsed_chain)
        final_metrics = evaluate_mapping(equation, winner, gt)

        winning_response_text = ""
        want_key = canonical_mapping_key(winner) if winner else ()
        for (pred, _), rt in zip(parsed_chain, response_texts_ok):
            if winner and canonical_mapping_key(pred) == want_key:
                winning_response_text = rt
                break
        if not winning_response_text and response_texts_ok:
            winning_response_text = response_texts_ok[-1]

        if token_totals is not None:
            for k in usage_sum:
                token_totals[k] += usage_sum[k]

        sc_block = {
            "n_samples": n_samples,
            "temperature": temperature,
            "num_api_errors": num_api_errors,
            "vote_meta": vote_meta,
            "samples": sample_records,
        }

        row = {
            "equation": equation,
            "tier": prob.get("tier", "unknown"),
            "condition": condition_name,
            "ground_truth": gt,
            "predicted": winner,
            "response_text": winning_response_text,
            "final_answer_present": has_final_answer_label(winning_response_text),
            "num_unique_letters": count_unique_letters(equation),
            **usage_sum,
            **final_metrics,
            "self_consistency": sc_block,
        }
        results.append(row)

    return results


def print_summary_sc(results: list[dict]) -> None:
    from collections import defaultdict

    by_condition = defaultdict(list)
    by_condition_tier = defaultdict(list)

    for result in results:
        by_condition[result["condition"]].append(result)
        by_condition_tier[(result["condition"], result["tier"])].append(result)

    print("\n" + "=" * 60)
    print("SELF-CONSISTENCY RESULTS")
    print("=" * 60)
    for cond in ["baseline_sc", "zero_shot_cot_sc", "few_shot_cot_sc"]:
        rs = by_condition[cond]
        if not rs:
            continue

        ema = sum(r["ema"] for r in rs) / len(rs)
        pla = sum(r["pla"] for r in rs) / len(rs)
        parse_fail = sum(r["parse_failure"] for r in rs) / len(rs)
        solved = sum(r["equation_satisfied"] for r in rs) / len(rs)
        final_answer = sum(r["final_answer_present"] for r in rs) / len(rs)
        api_err_rate = sum(
            1 for r in rs if r.get("self_consistency", {}).get("num_api_errors", 0) > 0
        ) / len(rs)
        print(f"\n{cond}:")
        print(
            f"  EMA: {ema:.1%}  |  PLA: {pla:.1%}  |  "
            f"Equation satisfied: {solved:.1%}  |  "
            f"Final Answer line: {final_answer:.1%}  |  Parse failures: {parse_fail:.1%}"
        )
        print(f"  Fraction of puzzles with ≥1 API error in samples: {api_err_rate:.1%}")

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


def main() -> None:
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
    print(f"Self-consistency: SC_N={SC_N}, SC_TEMPERATURE={SC_TEMPERATURE}")

    all_results: list[dict] = []
    token_totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    print("\n=== SC Condition 1: Baseline (no CoT) ===")
    all_results.extend(
        run_condition_sc(
            client,
            problems,
            "baseline_sc",
            BASELINE_SYSTEM,
            lambda eq: BASELINE_USER.format(equation=eq),
            token_totals,
            SC_N,
            SC_TEMPERATURE,
        )
    )

    print("\n=== SC Condition 2: Zero-shot CoT ===")
    all_results.extend(
        run_condition_sc(
            client,
            problems,
            "zero_shot_cot_sc",
            ZERO_SHOT_COT_SYSTEM,
            lambda eq: ZERO_SHOT_COT_USER.format(equation=eq),
            token_totals,
            SC_N,
            SC_TEMPERATURE,
        )
    )

    print("\n=== SC Condition 3: Few-shot CoT ===")
    few_shot_examples = load_few_shot_examples()
    all_results.extend(
        run_condition_sc(
            client,
            problems,
            "few_shot_cot_sc",
            FEW_SHOT_COT_SYSTEM,
            lambda eq: build_few_shot_user_prompt(eq, few_shot_examples),
            token_totals,
            SC_N,
            SC_TEMPERATURE,
        )
    )

    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"results/experiment_sc_{timestamp}.json"
    with open(output_path, "w") as f:
        json.dump(
            {
                "experiment_type": "self_consistency",
                "sc_n": SC_N,
                "sc_temperature": SC_TEMPERATURE,
                "sc_save_sample_responses": SC_SAVE_SAMPLE_RESPONSES,
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

    print_summary_sc(all_results)


if __name__ == "__main__":
    main()
