"""Per-latent steering-strength calibration by probabilistic bisection.

The paper defines the *threshold boost* for a latent as the steering strength at which the
model's first attempt scores 30/100 on average (App. A.1.4). Scores are noisy and bimodal, so
instead of plain bisection we use the Probabilistic Bisection Algorithm (Waeber, Frazier &
Henderson, 2013) with a Gaussian prior on the root location.

Ported (synchronously) from `threshold_finder.py` in
https://github.com/agencyenterprise/endogenous-steering-resistance (Apache-2.0, AE Studio).
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.stats import beta, norm


class BayesianRootFinder:
    """Posterior over the location of a root of a noisy monotone function on a grid."""

    def __init__(
        self,
        lower: float,
        upper: float,
        n_grid: int = 10_000,
        prior_mean: float | None = None,
        prior_std: float | None = None,
    ) -> None:
        self.grid = np.linspace(lower, upper, n_grid)
        if prior_mean is not None and prior_std is not None:
            logp = norm.logpdf(self.grid, prior_mean, prior_std)
        else:
            logp = np.zeros(n_grid)
        self.log_density = self._normalise(logp)
        self.history: list[dict] = []

    @staticmethod
    def _normalise(logp: np.ndarray) -> np.ndarray:
        logp = logp - logp.max()
        p = np.exp(logp)
        return np.log(p / p.sum())

    def median(self) -> float:
        p = np.exp(self.log_density)
        cdf = np.cumsum(p / p.sum())
        return float(self.grid[np.searchsorted(cdf, 0.5)])

    def update(self, x: float, direction: int, p_correct: float) -> None:
        """Bayes update after observing that the root is probably to the right (+1) or left (-1)
        of `x`, with confidence `p_correct` in [0.5, 1]."""
        assert direction in (-1, 1)
        assert 0.5 <= p_correct <= 1.0
        idx = int(np.searchsorted(self.grid, x))
        ll = np.empty_like(self.log_density)
        lp, lq = np.log(p_correct), np.log(1 - p_correct)
        if direction == -1:  # root is to the left of x
            ll[:idx], ll[idx:] = lp, lq
        else:
            ll[:idx], ll[idx:] = lq, lp
        if 0 <= idx < len(ll):
            ll[idx] = (lp + lq) / 2
        self.log_density = self._normalise(self.log_density + ll)
        self.history.append({"x": x, "direction": direction, "p": p_correct, "median": self.median()})


def find_threshold(
    score_fn: Callable[[float], float],
    *,
    target: float = 0.30,
    n_trials: int = 20,
    lower: float = 0.0,
    upper: float = 5.0,
    prior_mean: float = 1.0,
    prior_std: float = 0.34,
    update_weight: float = 5.0,
    verbose: bool = False,
) -> tuple[float, list[dict]]:
    """Find the boost `b` such that E[score_fn(b)] ~= target, where score_fn returns a value in
    [0, 1] that *decreases* with `b` (stronger steering -> worse first attempt).

    Returns (threshold, history). Each observed score is modelled as a Beta posterior
    (Jeffreys prior + `update_weight` pseudo-observations) to turn the single noisy score into a
    confidence that the root lies on the indicated side.
    """
    finder = BayesianRootFinder(lower, upper, prior_mean=prior_mean, prior_std=prior_std)
    for t in range(n_trials):
        b = finder.median()
        score = float(score_fn(b))
        # Score too high -> steering too weak -> root is to the right.
        direction = 1 if score >= target else -1
        a = 0.5 + update_weight * score
        bb = 0.5 + update_weight * (1 - score)
        p_below = beta.cdf(target, a, bb)
        p = (1 - p_below) if direction == 1 else p_below
        if p < 0.5:  # score ambiguous; trust the other side instead
            direction, p = -direction, 1 - p
        finder.update(b, direction, min(p, 1.0))
        if verbose:
            arrow = "boost up" if direction == 1 else "boost down"
            print(f"  [bisect {t + 1:2d}/{n_trials}] b={b:.3f} score={score:.2f} -> {arrow}; "
                  f"estimate={finder.median():.3f}")
    return finder.median(), finder.history
