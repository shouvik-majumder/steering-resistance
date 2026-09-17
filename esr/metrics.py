"""ESR metrics (Table 1 of the paper) with Wilson confidence intervals."""
from __future__ import annotations

import math
from collections import defaultdict


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Return (rate, lo, hi) for k successes in n trials. (0, 0, 0) if n == 0."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, max(0.0, centre - half), min(1.0, centre + half)


def trial_outcome(row: dict) -> dict | None:
    """Per-trial outcome from the judge output. None if the judge errored or found no attempts
    (clarifying-question responses are excluded from rates, as in the paper)."""
    judge = row.get("judge") or {}
    if judge.get("error"):
        return None
    attempts = judge.get("attempts") or []
    if not attempts:
        return None
    first, last = attempts[0].get("score"), attempts[-1].get("score")
    multi = len(attempts) >= 2
    scored = first is not None and last is not None
    improved = bool(multi and scored and last > first)
    return {
        "n_attempts": len(attempts),
        "first_score": first,
        "last_score": last,
        "multi_attempt": multi,
        "improved": improved,
        "esr": multi and improved,
        "scored": scored,
    }


def _mean(xs):
    return (sum(xs) / len(xs)) if xs else float("nan")


def _sem(xs):
    if len(xs) < 2:
        return float("nan")
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) / math.sqrt(len(xs))


def summarize(rows: list[dict]) -> dict:
    from .judge import restart_clusters

    outcomes = [o for o in (trial_outcome(r) for r in rows) if o is not None]
    n = len(outcomes)
    # Judge-independent "noticing" measure: explicit restart language anywhere in the response.
    n_restart_lang = sum(1 for r in rows if r.get("response") and restart_clusters(r["response"]))
    n_multi = sum(o["multi_attempt"] for o in outcomes)
    n_improved = sum(o["improved"] for o in outcomes)
    n_esr = sum(o["esr"] for o in outcomes)
    first_scores = [o["first_score"] for o in outcomes if o["first_score"] is not None]
    deltas = [o["last_score"] - o["first_score"] for o in outcomes if o["multi_attempt"] and o["scored"]]
    return {
        "n_trials": len(rows),
        "n_scored": n,
        "n_excluded": len(rows) - n,
        "multi_attempt": wilson(n_multi, n),
        "conditional_improvement": wilson(n_improved, n_multi),
        "esr": wilson(n_esr, n),
        "restart_language": wilson(n_restart_lang, len(rows)),
        "first_attempt_score_mean": _mean(first_scores),
        "first_attempt_score_sem": _sem(first_scores),
        "score_delta_mean": _mean(deltas),
        "counts": {"multi": n_multi, "improved": n_improved, "esr": n_esr},
    }


def summarize_by(rows: list[dict], key: str) -> dict[str, dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[str(r.get(key))].append(r)
    return {k: summarize(v) for k, v in sorted(groups.items())}


def format_summary(name: str, s: dict) -> str:
    def pct(t: tuple) -> str:
        return f"{100 * t[0]:5.1f}% [{100 * t[1]:.1f}, {100 * t[2]:.1f}]"

    return (
        f"{name:<32} n={s['n_scored']:<5} (excl {s['n_excluded']})  "
        f"first={s['first_attempt_score_mean']:5.1f}  "
        f"multi={pct(s['multi_attempt'])}  improve|multi={pct(s['conditional_improvement'])}  "
        f"ESR={pct(s['esr'])}  restart-lang={pct(s['restart_language'])}"
    )
