"""
Step 2: After running prepare_dataset.py, inspect the 3 few-shot examples
and manually add walkthroughs. This script loads the few-shot examples,
prints them so you can write walkthroughs, and saves a template file.

You will need to MANUALLY fill in the 'walkthrough' field for each example
with a step-by-step column-by-column carry reasoning explanation.
"""

import json


def main():
    with open("data/few_shot_examples.json") as f:
        examples = json.load(f)

    print("Few-shot examples that need manual walkthroughs:\n")
    output = []
    for i, ex in enumerate(examples, 1):
        print(f"Example {i}: {ex['question']}")
        print(f"  Answer: {ex['answer']}")
        print(f"  Unique letters: {ex['num_unique_letters']}")
        print()

        output.append({
            "question": ex["question"],
            "answer": ex["answer"],
            "walkthrough": "TODO: Write a step-by-step column-by-column walkthrough here.",
        })

    with open("data/few_shot_examples_with_walkthroughs.json", "w") as f:
        json.dump(output, f, indent=2)

    print("Template saved to data/few_shot_examples_with_walkthroughs.json")
    print("Edit that file to fill in the 'walkthrough' fields before running the experiment.")


if __name__ == "__main__":
    main()
