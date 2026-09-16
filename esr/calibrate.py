"""Per-latent threshold calibration with on-disk cache (data/thresholds/<model>.json)."""
from __future__ import annotations

import random
import time

from .config import DATA, ExperimentConfig
from .features import label_of
from .io import read_json, write_json
from .threshold import find_threshold


def threshold_path(model_key: str):
    return DATA / "thresholds" / f"{model_key}.json"


def load_threshold_cache(model_key: str) -> dict:
    return read_json(threshold_path(model_key), default={}) or {}


def first_attempt_score(judge, prompt: str, response: str, label: str, retries: int = 2) -> float | None:
    """Judge the response and return the first attempt's score in [0, 1], or None on failure."""
    for attempt in range(retries + 1):
        res = judge.grade(prompt, response, label)
        if res.error is None:
            if not res.attempts:
                return 0.0  # no attempt at all counts as a failed first attempt
            s = res.first_score
            return None if s is None else s / 100.0
        if attempt < retries:
            time.sleep(2.0 * (attempt + 1))
    return None


def calibrate_latent(
    engine,
    judge,
    prompts: list[str],
    labels: dict[int, str],
    latent: int,
    cfg: ExperimentConfig,
    *,
    force: bool = False,
    verbose: bool = True,
    verify_samples: int = 3,
) -> float:
    """Return the boost at which this latent drives the first-attempt score to ~cfg.target_score."""
    cache = load_threshold_cache(cfg.model)
    entry = cache.get(str(latent))
    if entry and entry.get("threshold") is not None and not force:
        return float(entry["threshold"])

    label = label_of(labels, latent)
    rng = random.Random(f"calib:{cfg.seed}:{latent}")
    if verbose:
        print(f"\nCalibrating latent {latent} ({label!r})")

    def score_fn(boost: float) -> float:
        scores = []
        for _ in range(cfg.threshold_samples_per_trial):
            prompt = rng.choice(prompts)
            seed = rng.randrange(1_000_000)
            gen = engine.generate(
                prompt,
                latent=latent,
                boost=boost,
                seed=seed,
                max_new_tokens=cfg.max_new_tokens,
                temperature=cfg.temperature,
                meta_prompt=cfg.meta_prompt,
            )
            s = first_attempt_score(judge, prompt, gen.response, label)
            if s is not None:
                scores.append(s)
        # No usable judgement: return the target so the bisection makes no update.
        return sum(scores) / len(scores) if scores else cfg.target_score

    threshold, history = find_threshold(
        score_fn,
        target=cfg.target_score,
        n_trials=cfg.threshold_n_trials,
        lower=cfg.threshold_lower,
        upper=cfg.threshold_upper,
        prior_mean=cfg.threshold_prior_mean,
        prior_std=cfg.threshold_prior_std,
        verbose=verbose,
    )
    threshold = round(threshold, 3)
    achieved = [score_fn(threshold) for _ in range(verify_samples)] if verify_samples else []

    cache = load_threshold_cache(cfg.model)  # re-read in case another process wrote
    cache[str(latent)] = {
        "threshold": threshold,
        "label": label,
        "achieved_first_scores": achieved,
        "history": history,
        "unit_scale": engine.unit_scale,
        "config": {
            "target": cfg.target_score,
            "n_trials": cfg.threshold_n_trials,
            "samples_per_trial": cfg.threshold_samples_per_trial,
            "bounds": [cfg.threshold_lower, cfg.threshold_upper],
            "prior": [cfg.threshold_prior_mean, cfg.threshold_prior_std],
            "meta_prompt": cfg.meta_prompt,
            "judge": getattr(judge, "name", "?"),
        },
    }
    write_json(threshold_path(cfg.model), cache)
    if verbose:
        mean_ach = sum(achieved) / len(achieved) if achieved else float("nan")
        print(f"  -> threshold={threshold:.3f}, verification first-attempt score={mean_ach:.2f}")
    return threshold
