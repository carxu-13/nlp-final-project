"""
Generate cryptarithm puzzles with verified solutions using brute-force search.
Produces addition-type puzzles (A + B = C) with unique digit assignments
and no leading zeros.
"""

import json
import os
import random
import string
from itertools import permutations

random.seed(42)

# Word lists organized by length for generating puzzles
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
    "HORSE", "HOTEL", "HOUSE", "HURDLE",
]


def get_unique_letters(*words: str) -> set[str]:
    return set(c for w in words for c in w if c.isalpha())


def get_leading_letters(*words: str) -> set[str]:
    return set(w[0] for w in words)


def word_to_number(word: str, mapping: dict[str, int]) -> int:
    result = 0
    for c in word:
        result = result * 10 + mapping[c]
    return result


def solve_cryptarithm(w1: str, w2: str, result: str) -> list[dict[str, int]]:
    """Find all valid solutions for w1 + w2 = result."""
    letters = sorted(get_unique_letters(w1, w2, result))
    n = len(letters)
    if n > 10:
        return []

    leading = get_leading_letters(w1, w2, result)
    solutions = []

    for perm in permutations(range(10), n):
        mapping = dict(zip(letters, perm))

        # Check no leading zeros
        if any(mapping[l] == 0 for l in leading):
            continue

        # Check equation
        n1 = word_to_number(w1, mapping)
        n2 = word_to_number(w2, mapping)
        nr = word_to_number(result, mapping)

        if n1 + n2 == nr:
            solutions.append(mapping)

    return solutions


def generate_puzzles(target_count: int, max_unique_letters: int = 8) -> list[dict]:
    """Generate cryptarithm puzzles by trying random word combinations."""
    # Build word pools by length
    all_words = WORDS_3 + WORDS_4 + WORDS_5
    puzzles = []
    seen = set()

    # Try pairs of words as addends
    word_pairs = []
    for w1 in all_words:
        for w2 in all_words:
            # Result must be at least as long as the longest addend
            max_len = max(len(w1), len(w2))
            n_unique = len(get_unique_letters(w1, w2))
            # Quick filter: if just the addends already exceed our letter budget
            # minus 1 (result needs at least 1 char), skip
            if n_unique > max_unique_letters:
                continue
            word_pairs.append((w1, w2))

    random.shuffle(word_pairs)
    print(f"Trying {len(word_pairs)} word pairs...")

    for idx, (w1, w2) in enumerate(word_pairs):
        if len(puzzles) >= target_count:
            break

        if idx % 1000 == 0 and idx > 0:
            print(f"  Checked {idx} pairs, found {len(puzzles)} puzzles so far...")

        # Compute what result words could look like
        # Try each word in our list as the result
        max_len = max(len(w1), len(w2))
        for result_word in all_words:
            if len(result_word) < max_len or len(result_word) > max_len + 1:
                continue

            letters = get_unique_letters(w1, w2, result_word)
            n_letters = len(letters)
            if n_letters > max_unique_letters or n_letters < 3:
                continue

            puzzle_key = f"{w1}+{w2}={result_word}"
            if puzzle_key in seen:
                continue
            seen.add(puzzle_key)

            solutions = solve_cryptarithm(w1, w2, result_word)
            if len(solutions) == 1:  # Unique solution only
                sol = solutions[0]
                puzzles.append({
                    "puzzle": f"{w1} + {w2} = {result_word}",
                    "question": f"{w1} + {w2} = {result_word}",
                    "answer": str(sol),
                    "num_unique_letters": n_letters,
                    "num_addends": 2,
                    "solution": sol,
                })
                if len(puzzles) % 10 == 0:
                    print(f"  Found {len(puzzles)} puzzles...")
                if len(puzzles) >= target_count:
                    break

    return puzzles


def main():
    os.makedirs("data", exist_ok=True)

    # We need 100 test + 3 few-shot = 103+ puzzles
    # Generate more than needed to allow stratified sampling
    print("Generating cryptarithm puzzles with unique solutions...")
    puzzles = generate_puzzles(target_count=200, max_unique_letters=8)

    print(f"\nTotal puzzles generated: {len(puzzles)}")

    # Stratify
    easy = [p for p in puzzles if p["num_unique_letters"] <= 5]
    medium = [p for p in puzzles if 6 <= p["num_unique_letters"] <= 7]
    hard = [p for p in puzzles if p["num_unique_letters"] == 8]

    print(f"Easy (≤5 letters): {len(easy)}")
    print(f"Medium (6-7 letters): {len(medium)}")
    print(f"Hard (8 letters): {len(hard)}")

    random.shuffle(easy)
    random.shuffle(medium)
    random.shuffle(hard)

    # Reserve 3 for few-shot (1 per tier if possible)
    few_shot = []
    for tier_list in [easy, medium, hard]:
        if tier_list:
            few_shot.append(tier_list.pop(0))

    n_easy = min(34, len(easy))
    n_medium = min(33, len(medium))
    n_hard = min(33, len(hard))

    sampled_easy = easy[:n_easy]
    sampled_medium = medium[:n_medium]
    sampled_hard = hard[:n_hard]

    for p in sampled_easy:
        p["tier"] = "easy"
    for p in sampled_medium:
        p["tier"] = "medium"
    for p in sampled_hard:
        p["tier"] = "hard"
    for p in few_shot:
        p["tier"] = "few_shot_example"

    # Remove the brute-force solution dict before saving (keep answer string)
    test_set = sampled_easy + sampled_medium + sampled_hard
    random.shuffle(test_set)
    for p in test_set + few_shot:
        p.pop("solution", None)

    print(f"\nFinal test set: {len(test_set)} problems")
    print(f"  Easy: {n_easy}, Medium: {n_medium}, Hard: {n_hard}")
    print(f"Few-shot examples: {len(few_shot)}")

    with open("data/test_set.json", "w") as f:
        json.dump(test_set, f, indent=2)
    with open("data/few_shot_examples.json", "w") as f:
        json.dump(few_shot, f, indent=2)

    print("\nSaved to data/test_set.json and data/few_shot_examples.json")


if __name__ == "__main__":
    main()
