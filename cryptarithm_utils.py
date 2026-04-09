"""Shared helpers for cryptarithm dataset generation, prompting, and evaluation."""

from __future__ import annotations

import re
from typing import Iterable


LETTER_DIGIT_PATTERN = re.compile(r"'?([A-Z])'?\s*[=:]\s*(\d)")


def extract_letter_digit_pairs(text: str) -> list[tuple[str, int]]:
    """Return all letter-digit assignments found in text."""
    return [(letter, int(digit)) for letter, digit in LETTER_DIGIT_PATTERN.findall(text.upper())]


def parse_ground_truth(answer_str: str) -> dict[str, int]:
    """Parse a stored answer string into a mapping."""
    return {letter: digit for letter, digit in extract_letter_digit_pairs(answer_str)}


def format_mapping(mapping: dict[str, int]) -> str:
    """Render a mapping as a stable comma-separated answer line."""
    return ", ".join(f"{letter}={digit}" for letter, digit in sorted(mapping.items()))


def count_unique_letters(equation: str) -> int:
    return len(letters_in_equation(equation))


def letters_in_equation(equation: str) -> set[str]:
    return {char for char in equation if char.isalpha() and char.isupper()}


def has_final_answer_label(text: str) -> bool:
    return bool(re.search(r"(?im)^\s*final answer\s*:", text))


def parse_mapping_text(
    text: str,
    expected_letters: Iterable[str] | None = None,
) -> dict[str, int]:
    """Extract the most likely final mapping from a model response.

    The parser prefers a labeled ``Final Answer:`` line, then other answer-like
    lines, then falls back to the full response.
    """
    if not text:
        return {}

    expected = set(expected_letters or [])
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    candidate_segments = []
    candidate_segments.extend(
        line for line in lines if re.search(r"(?i)\bfinal answer\s*:", line)
    )
    candidate_segments.extend(
        line
        for line in lines
        if line not in candidate_segments and re.search(r"(?i)\banswer\s*:", line)
    )
    candidate_segments.extend(lines)
    candidate_segments.append(text)

    best_mapping: dict[str, int] = {}
    best_score = (-1, -1, -1, -1)

    for segment in candidate_segments:
        mapping = {}
        for letter, digit in extract_letter_digit_pairs(segment):
            if expected and letter not in expected:
                continue
            mapping[letter] = digit

        if not mapping:
            continue

        overlap = len(set(mapping) & expected) if expected else len(mapping)
        exact_cover = int(bool(expected) and set(mapping) == expected)
        labeled = int("FINAL ANSWER" in segment.upper())
        score = (exact_cover, labeled, overlap, len(mapping))

        if score > best_score:
            best_score = score
            best_mapping = mapping

    return best_mapping


def mapping_has_unique_digits(mapping: dict[str, int]) -> bool:
    return len(mapping.values()) == len(set(mapping.values()))


def parse_equation(equation: str) -> tuple[list[str], str]:
    """Parse equations of the form WORD + WORD = RESULT."""
    left, right = equation.split("=")
    addends = [token.strip() for token in left.split("+") if token.strip()]
    result = right.strip()
    return addends, result


def word_to_number(word: str, mapping: dict[str, int]) -> int:
    return int("".join(str(mapping[char]) for char in word))


def mapping_satisfies_equation(equation: str, mapping: dict[str, int]) -> bool:
    """Check that a mapping is complete, valid, and solves the puzzle."""
    expected_letters = letters_in_equation(equation)
    if set(mapping) != expected_letters:
        return False
    if not mapping_has_unique_digits(mapping):
        return False

    addends, result = parse_equation(equation)
    for word in [*addends, result]:
        if mapping[word[0]] == 0:
            return False

    addend_values = [word_to_number(word, mapping) for word in addends]
    result_value = word_to_number(result, mapping)
    return sum(addend_values) == result_value
