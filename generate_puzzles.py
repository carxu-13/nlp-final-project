"""
Generate cryptarithm puzzles with verified solutions using backtracking search
with column-by-column constraint propagation for fast solving.
"""

import json
import os
import random
import time

from cryptarithm_utils import format_mapping

RANDOM_SEED = 42
GENERATED_POOL_TARGET = 200
EVAL_SAMPLE_SIZE = 40

random.seed(RANDOM_SEED)

# Word lists by length
WORDS_3 = [
    "ACE", "ADD", "AGE", "AID", "AIM", "AIR", "ALE", "APE", "ARC", "ARE",
    "ARK", "ARM", "ART", "ATE", "AWE", "AXE", "BAD", "BAG", "BAN", "BAR",
    "BAT", "BED", "BET", "BIG", "BIT", "BOW", "BOX", "BUD", "BUG", "BUS",
    "BUT", "BUY", "CAB", "CAN", "CAP", "CAR", "CAT", "COB", "COD", "COG",
    "COP", "COT", "COW", "CRY", "CUB", "CUD", "CUP", "CUR", "CUT", "DAB",
    "DAM", "DAY", "DEN", "DEW", "DIG", "DIM", "DIP", "DOC", "DOG", "DOT",
    "DRY", "DUB", "DUG", "DUO", "DYE", "EAR", "EAT", "EEL", "EGG", "ELF",
    "ELK", "ELM", "EMU", "END", "ERA", "EVE", "EWE", "EYE", "FAN", "FAR",
    "FAT", "FAX", "FED", "FEW", "FIG", "FIN", "FIR", "FIT", "FIX", "FLY",
    "FOB", "FOE", "FOG", "FOP", "FOR", "FOX", "FRY", "FUN", "FUR", "GAB",
    "GAG", "GAP", "GAS", "GEL", "GEM", "GET", "GNU", "GOB", "GOD", "GOT",
    "GUM", "GUN", "GUT", "GUY", "GYM", "HAD", "HAM", "HAS", "HAT", "HAY",
    "HEN", "HER", "HEW", "HEX", "HID", "HIM", "HIP", "HIS", "HIT", "HOB",
    "HOG", "HOP", "HOT", "HOW", "HUB", "HUE", "HUG", "HUM", "HUT",
]

WORDS_4 = [
    "ABLE", "ACID", "AGED", "ARCH", "AREA", "ARMY", "BAKE", "BALD", "BALE",
    "BAND", "BANE", "BANK", "BARE", "BARK", "BARN", "BASE", "BATH", "BEAD",
    "BEAM", "BEAR", "BEAT", "BEND", "BIKE", "BIRD", "BITE", "BLOW", "BLUE",
    "BOAT", "BOLD", "BOLT", "BOMB", "BOND", "BONE", "BORE", "BORN", "BOTH",
    "BOWL", "BULK", "BURN", "CAGE", "CAKE", "CALM", "CAME", "CAMP", "CAPE",
    "CARD", "CARE", "CART", "CASE", "CASH", "CAST", "CAVE", "CHIN", "CHIP",
    "CITY", "CLAM", "CLAN", "CLAP", "CLAY", "CLIP", "CLUB", "CLUE", "COAL",
    "COAT", "CODE", "COIL", "COIN", "COLD", "COLT", "COMB", "COME", "CONE",
    "COOK", "COPE", "CORD", "CORE", "CORK", "CORN", "COST", "CRAB", "CROP",
    "CUBE", "CULT", "CURB", "CURE", "CURL", "DALE", "DAME", "DAMP", "DARE",
    "DARK", "DART", "DAWN", "DEAL", "DEAR", "DECK", "DEEP", "DEER", "DESK",
    "DIAL", "DICE", "DIME", "DIRT", "DISC", "DISH", "DOCK", "DOME", "DONE",
    "DOOR", "DOSE", "DOVE", "DOWN", "DRAW", "DRIP", "DROP", "DRUG", "DRUM",
    "DUAL", "DUEL", "DUKE", "DULL", "DUNE", "DUSK", "DUST", "DUTY", "EACH",
    "EARN", "EASE", "EDIT", "FACE", "FACT", "FADE", "FAIL", "FAIR", "FAKE",
    "FAME", "FANG", "FARE", "FARM", "FAST", "FATE", "FAWN", "FEAR", "FEAT",
    "FEED", "FEEL", "FELT", "FERN", "FILM", "FIND", "FINE", "FIRE", "FIRM",
    "FISH", "FLAG", "FLAW", "FLED", "FLIP", "FLOW", "FOAM", "FOIL", "FOLD",
    "FOLK", "FOND", "FONT", "FOOD", "FOOL", "FORD", "FORK", "FORM", "FORT",
    "FOUL", "FROG", "FUEL", "FUND", "FURY", "FUSE", "GAIN", "GALE", "GAME",
    "GANG", "GATE", "GAVE", "GAZE", "GEAR", "GIFT", "GIRL", "GLAD", "GLOW",
    "GLUE", "GOAT", "GOES", "GOLD", "GOLF", "GONE", "GRAB", "GRAY", "GREW",
    "GRID", "GRIM", "GRIN", "GRIP", "GROW", "GULF", "GUST", "HACK", "HAIL",
    "HAIR", "HALE", "HALF", "HALL", "HALT", "HAND", "HANG", "HARD", "HARE",
    "HARM", "HARP", "HATE", "HAUL", "HAWK", "HAZE", "HEAD", "HEAL", "HEAP",
    "HEAT", "HEEL", "HELD", "HELM", "HELP", "HERB", "HERD", "HERE", "HERO",
    "HIGH", "HIKE", "HILL", "HIND", "HINT", "HIRE", "HOLD", "HOLE", "HOME",
    "HOOD", "HOOK", "HOPE", "HORN", "HOST", "HUGE", "HULL", "HUNG", "HUNT",
    "HURT", "HYMN",
]

WORDS_5 = [
    "BLADE", "BLAZE", "BLEAK", "BLEND", "BLIMP", "BLIND", "BLOCK", "BLOWN",
    "BOARD", "BOUND", "BRAIN", "BRAND", "BRAVO", "BREAD", "BREAK", "BRICK",
    "BRIDE", "BRIEF", "BRING", "BROAD", "BROIL", "BRUSH", "BUILD", "BUNCH",
    "CABIN", "CARGO", "CHAIN", "CHAIR", "CHALK", "CHARM", "CHASE", "CHEAP",
    "CHEST", "CHIEF", "CHILD", "CHORD", "CHUNK", "CLAIM", "CLIMB", "CLING",
    "CLOAK", "CLONE", "CLOUD", "COACH", "CORAL", "COUNT", "COURT", "COVER",
    "CRAFT", "CRANE", "CRAWL", "CREAM", "CREST", "CROWD", "CROWN", "CRUSH",
    "CURVE", "DANCE", "DEALT", "DEBUT", "DECOY", "DELTA", "DEPOT", "DRAFT",
    "DRAIN", "DRAKE", "DREAM", "DRIFT", "DRINK", "DRIVE", "DROIT", "DROWN",
    "EARLY", "EARTH", "EMBER", "EQUAL", "EVADE", "EXILE", "EXTRA", "FABLE",
    "FEAST", "FIBER", "FIELD", "FLAME", "FLANK", "FLARE", "FLESH", "FLOAT",
    "FLOCK", "FLOOD", "FLOOR", "FLOUR", "FLUID", "FORGE", "FOUND", "FRAME",
    "FRAUD", "FRESH", "FROST", "FRUIT", "GLARE", "GLEAM", "GLOBE", "GLOOM",
    "GRACE", "GRADE", "GRAIN", "GRAND", "GRANT", "GRAPE", "GRASP", "GRAVE",
    "GRIND", "GROVE", "GUARD", "GUIDE", "HAIKU", "HAVEN", "HEART", "HOIST",
    "HORSE", "HOTEL", "HOUSE",
]


def get_unique_letters(*words):
    return set(c for w in words for c in w if c.isalpha())


def solve_cryptarithm(w1, w2, result, max_solutions=2):
    """Solve using backtracking with column-by-column constraint propagation.

    Stops early once max_solutions are found because we only keep unique-solution
    puzzles.
    """
    max_len = len(result)
    w1_padded = w1.rjust(max_len, " ")
    w2_padded = w2.rjust(max_len, " ")

    columns = []
    for i in range(max_len - 1, -1, -1):
        c1 = w1_padded[i] if w1_padded[i] != " " else None
        c2 = w2_padded[i] if w2_padded[i] != " " else None
        cr = result[i]
        columns.append((c1, c2, cr))

    letters_ordered = []
    seen = set()
    for c1, c2, cr in columns:
        for char in [c1, c2, cr]:
            if char and char not in seen:
                letters_ordered.append(char)
                seen.add(char)

    leading = {result[0], w1[0], w2[0]}

    solutions = []
    mapping = {}
    used_digits = set()

    def backtrack(letter_idx):
        if len(solutions) >= max_solutions:
            return

        if letter_idx == len(letters_ordered):
            carry = 0
            for c1, c2, cr in columns:
                d1 = mapping[c1] if c1 else 0
                d2 = mapping[c2] if c2 else 0
                dr = mapping[cr]
                total = d1 + d2 + carry
                if total % 10 != dr:
                    return
                carry = total // 10
            if carry == 0:
                solutions.append(dict(mapping))
            return

        letter = letters_ordered[letter_idx]
        start = 1 if letter in leading else 0

        for digit in range(start, 10):
            if digit in used_digits:
                continue

            mapping[letter] = digit
            used_digits.add(digit)

            ok = True
            carry = 0
            for c1, c2, cr in columns:
                assigned = True
                for char in [c1, c2, cr]:
                    if char and char not in mapping:
                        assigned = False
                        break

                if not assigned:
                    break

                d1 = mapping[c1] if c1 else 0
                d2 = mapping[c2] if c2 else 0
                dr = mapping[cr]
                total = d1 + d2 + carry
                if total % 10 != dr:
                    ok = False
                    break
                carry = total // 10

            if ok:
                backtrack(letter_idx + 1)

            del mapping[letter]
            used_digits.remove(digit)

    backtrack(0)
    return solutions


def generate_puzzles(target_count, max_unique_letters=8):
    all_words = WORDS_3 + WORDS_4 + WORDS_5
    puzzles = []
    seen = set()
    candidates = []

    for w1 in all_words:
        for w2 in all_words:
            if w1 > w2:
                continue
            max_len = max(len(w1), len(w2))
            addend_letters = get_unique_letters(w1, w2)
            if len(addend_letters) > max_unique_letters:
                continue
            for result_word in all_words:
                if len(result_word) < max_len or len(result_word) > max_len + 1:
                    continue
                all_letters = get_unique_letters(w1, w2, result_word)
                n = len(all_letters)
                if n > max_unique_letters or n < 3:
                    continue
                candidates.append((w1, w2, result_word, n))

    random.shuffle(candidates)
    print(f"Total candidates to check: {len(candidates)}")

    t0 = time.time()
    idx = -1
    for idx, (w1, w2, rw, n_letters) in enumerate(candidates):
        if len(puzzles) >= target_count:
            break

        if idx % 5000 == 0 and idx > 0:
            elapsed = time.time() - t0
            rate = idx / elapsed
            print(
                f"  Checked {idx}/{len(candidates)} ({rate:.0f}/s), "
                f"found {len(puzzles)} puzzles [{elapsed:.1f}s]"
            )

        key = f"{w1}+{w2}={rw}"
        if key in seen:
            continue
        seen.add(key)

        solutions = solve_cryptarithm(w1, w2, rw, max_solutions=2)
        if len(solutions) == 1:
            sol = solutions[0]
            puzzles.append({
                "puzzle": f"{w1} + {w2} = {rw}",
                "question": f"{w1} + {w2} = {rw}",
                "answer": format_mapping(sol),
                "num_unique_letters": n_letters,
                "num_addends": 2,
                "solution": sol,
            })
            if len(puzzles) % 20 == 0:
                print(f"  >>> Found {len(puzzles)} puzzles so far")

    elapsed = time.time() - t0
    print(
        f"Done. Checked {min(idx + 1, len(candidates))} candidates in {elapsed:.1f}s, "
        f"found {len(puzzles)} puzzles."
    )
    return puzzles


def main():
    os.makedirs("data", exist_ok=True)

    print("Generating cryptarithm puzzles with unique solutions...")
    print("Using backtracking solver with column constraint propagation.\n")
    puzzles = generate_puzzles(target_count=GENERATED_POOL_TARGET, max_unique_letters=8)

    print(f"\nTotal puzzles generated: {len(puzzles)}")

    for puzzle in puzzles:
        if puzzle["num_unique_letters"] <= 5:
            puzzle["tier"] = "easy"
        elif puzzle["num_unique_letters"] <= 7:
            puzzle["tier"] = "medium"
        else:
            puzzle["tier"] = "hard"

    easy = [p for p in puzzles if p["tier"] == "easy"]
    medium = [p for p in puzzles if p["tier"] == "medium"]
    hard = [p for p in puzzles if p["tier"] == "hard"]

    print(f"Easy (<=5 letters): {len(easy)}")
    print(f"Medium (6-7 letters): {len(medium)}")
    print(f"Hard (8 letters): {len(hard)}")

    random.shuffle(easy)
    random.shuffle(medium)
    random.shuffle(hard)

    few_shot = []
    for tier_list in [easy, medium, hard]:
        if tier_list:
            few_shot.append(tier_list.pop(0))

    for puzzle in few_shot:
        puzzle["tier"] = "few_shot_example"

    remaining = easy + medium + hard
    random.shuffle(remaining)
    test_set = remaining[: min(EVAL_SAMPLE_SIZE, len(remaining))]
    random.shuffle(test_set)

    for puzzle in test_set + few_shot:
        puzzle.pop("solution", None)

    print(f"\nFinal test set: {len(test_set)} problems")
    print(f"  Easy: {sum(p['tier'] == 'easy' for p in test_set)}")
    print(f"  Medium: {sum(p['tier'] == 'medium' for p in test_set)}")
    print(f"  Hard: {sum(p['tier'] == 'hard' for p in test_set)}")
    print(f"Few-shot examples: {len(few_shot)}")

    metadata = {
        "original_external_dataset": "theblackcat102/cryptarithm",
        "external_dataset_usable_examples": 9,
        "external_dataset_issue": (
            "Most entries violated the two-addend / <=8 unique-letter constraints and "
            "the source also contained duplicates."
        ),
        "replacement_dataset": "synthetically generated with verified unique solutions",
        "generation_constraints": {
            "num_addends": 2,
            "max_unique_letters": 8,
            "requires_unique_solution": True,
        },
        "random_seed": RANDOM_SEED,
        "generated_pool_size": len(puzzles),
        "evaluation_sample_size": len(test_set),
        "few_shot_example_count": len(few_shot),
    }

    with open("data/test_set.json", "w") as f:
        json.dump(test_set, f, indent=2)
    with open("data/few_shot_examples.json", "w") as f:
        json.dump(few_shot, f, indent=2)
    with open("data/generated_puzzle_pool.json", "w") as f:
        json.dump(puzzles, f, indent=2)
    with open("data/dataset_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(
        "\nSaved to data/test_set.json, data/few_shot_examples.json, "
        "data/generated_puzzle_pool.json, and data/dataset_metadata.json"
    )


if __name__ == "__main__":
    main()
