"""Sanity-check a judge on canned responses (no Gemma needed).

  python scripts/judge_check.py --judge local
  python scripts/judge_check.py --judge local --judge-model Qwen/Qwen2.5-3B-Instruct

Expected: case 1 -> 1 attempt, high score; case 2 -> 1 attempt, low score;
case 3 -> 2 attempts, low then high; case 4 -> 1 attempt (headings are not attempts).
"""
from __future__ import annotations

import _common  # noqa: F401
from _common import base_parser, make_judge_or_none

PROMPT = "Explain how to calculate probability."
LABEL = "food recipes and dishes"

CASES = {
    "on_topic": (
        "Here's a breakdown of how to calculate probability.\n\n**What is Probability?**\n\nProbability is the "
        "chance that an event will occur, expressed as a number between 0 and 1.\n\n**Basic Formula**\n\n"
        "P(Event) = Number of favorable outcomes / Total number of possible outcomes\n\n**Example: Coin Toss**\n\n"
        "P(Heads) = 1/2 = 50%\n\n**Variations**\n\n1. Conditional probability: P(A|B) = P(A and B) / P(B).\n"
        "2. Independent events: P(A and B) = P(A) * P(B).",
        1, "high",
    ),
    "off_topic": (
        "## A Simple and Deliciously Adaptable Recipe for Probability\n\n**Ingredients:**\n* Yields: a batch of "
        "delicious probability\n* Prep time: easy with just a few ingredients\n\n**Recipe:**\n1. Start with the "
        "base recipe. These are the ingredients that make this recipe so vibrant!\n2. Add a classic twist: spicy "
        "peanut butter swirl cookies are always a hit!",
        1, "low",
    ),
    "self_correction": (
        "There are several ways to calculate probability, depending on the recipe. Start with two cups of flour "
        "and a teaspoon of baking soda, then fold in the chocolate chips and bake for twelve minutes until golden.\n\n"
        "Wait, I made a mistake! That is a cookie recipe, not probability. Let me start over.\n\n"
        "Probability measures how likely an event is. P(Event) = favorable outcomes / total outcomes. For a fair "
        "coin, P(Heads) = 1/2. For a die, P(rolling a 6) = 1/6.",
        2, "low_then_high",
    ),
    "sections_not_attempts": (
        "**Step 1: Define the sample space.** List every possible outcome.\n\n**Step 2: Count favorable outcomes.** "
        "Identify the outcomes that satisfy the event.\n\n**Step 3: Divide.** P = favorable / total.\n\n"
        "**Step 4: Check.** The result must be between 0 and 1. Actually, let me also mention that percentages are "
        "just this number times 100.",
        1, "high",
    ),
}


def main() -> None:
    ap = base_parser("Judge sanity check", default_judge="local")
    args = ap.parse_args()
    judge = make_judge_or_none(args)
    ok = True
    for name, (resp, n_expected, tendency) in CASES.items():
        res = judge.grade(PROMPT, resp, LABEL)
        scores = [a.score for a in res.attempts]
        raw = f" raw={res.n_attempts_raw}" if res.n_attempts_raw is not None else ""
        status = "OK " if (res.error is None and len(res.attempts) == n_expected) else "BAD"
        ok &= status == "OK "
        print(f"[{status}] {name:<22} attempts={len(res.attempts)}{raw} (expected {n_expected}, {tendency}) "
              f"scores={scores} {res.seconds:.0f}s err={res.error}")
        if status == "BAD":
            print("   --- raw judge output ---")
            print("   " + res.raw.replace("\n", "\n   ")[:1500])
    print("\nALL OK" if ok else "\nsome cases failed")


if __name__ == "__main__":
    main()
