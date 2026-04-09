"""
Post-hoc analysis of experiment results. Loads a results JSON file
and produces summary tables and breakdowns.
"""

import json
import sys

import pandas as pd


def load_results(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def analyze(results: list[dict]):
    df = pd.DataFrame(results)

    print("=" * 70)
    print("AGGREGATE RESULTS BY CONDITION")
    print("=" * 70)
    agg_spec = {
        "EMA": ("ema", "mean"),
        "PLA": ("pla", "mean"),
        "parse_failure_rate": ("parse_failure", "mean"),
        "n": ("ema", "count"),
    }
    if "equation_satisfied" in df.columns:
        agg_spec["equation_satisfied_rate"] = ("equation_satisfied", "mean")
    if "final_answer_present" in df.columns:
        agg_spec["final_answer_rate"] = ("final_answer_present", "mean")

    agg = df.groupby("condition").agg(**agg_spec)
    print(agg.to_string(float_format="%.3f"))

    print("\n" + "=" * 70)
    print("RESULTS BY CONDITION AND TIER")
    print("=" * 70)
    tier_agg_spec = {
        "EMA": ("ema", "mean"),
        "PLA": ("pla", "mean"),
        "n": ("ema", "count"),
    }
    if "equation_satisfied" in df.columns:
        tier_agg_spec["equation_satisfied_rate"] = ("equation_satisfied", "mean")

    tier_agg = df.groupby(["condition", "tier"]).agg(**tier_agg_spec)
    print(tier_agg.to_string(float_format="%.3f"))

    print("\n" + "=" * 70)
    print("ERROR ANALYSIS")
    print("=" * 70)
    pivot = df.pivot_table(index="equation", columns="condition", values="ema", aggfunc="first")
    all_wrong = pivot[pivot.sum(axis=1) == 0]
    print(f"Problems ALL conditions got wrong: {len(all_wrong)} / {len(pivot)}")

    all_right = pivot[pivot.sum(axis=1) == pivot.shape[1]]
    print(f"Problems ALL conditions got right: {len(all_right)} / {len(pivot)}")

    if "baseline" in pivot.columns and "few_shot_cot" in pivot.columns:
        improved = pivot[(pivot["baseline"] == False) & (pivot["few_shot_cot"] == True)]
        regressed = pivot[(pivot["baseline"] == True) & (pivot["few_shot_cot"] == False)]
        print(f"Few-shot CoT improved over baseline: {len(improved)}")
        print(f"Few-shot CoT regressed from baseline: {len(regressed)}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_results.py results/experiment_XXXXXX.json")
        sys.exit(1)
    analyze(load_results(sys.argv[1]))
