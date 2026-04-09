# nlp-final-project

Cryptarithm prompting experiments for CS 6770.

The project no longer treats the original HuggingFace evaluation set as the main test source. We initially filtered `theblackcat102/cryptarithm`, but only 9 puzzles satisfied the project constraints of exactly 2 addends and at most 8 unique letters, and the source also contained duplicate entries. The repo now treats a generated, solver-verified pool of about 200 puzzles as the primary dataset and samples 82 evaluation puzzles from that pool.

The experiment pipeline also assumes stricter answer formatting than before. Prompts now require a final line in the form `Final Answer: A=1, B=2, ...`, and the parser explicitly prefers that line while still falling back to looser extraction when needed. Few-shot CoT examples were updated to demonstrate carry reasoning plus backtracking or light brute force when the constraints do not immediately determine every digit.

Common commands:

```bash
python generate_puzzles.py
python build_few_shot_walkthroughs.py
python run_experiment.py
python analyze_results.py results/experiment_<timestamp>.json
```

Generated artifacts live in `data/`, including:

- `generated_puzzle_pool.json`: full generated pool with verified unique solutions
- `test_set.json`: sampled 82-problem evaluation set
- `few_shot_examples.json`: reserved few-shot examples
- `dataset_metadata.json`: dataset provenance and constraint summary