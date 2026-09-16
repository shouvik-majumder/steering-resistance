"""Latent labels, relevance filtering and steering-latent sampling (paper App. A.1.2)."""
from __future__ import annotations

import csv
import random
from pathlib import Path

import torch

from .config import DATA, ROOT, ModelSpec
from .io import read_json, write_json


def load_labels(spec: ModelSpec) -> dict[int, str]:
    path = ROOT / spec.labels_csv
    labels: dict[int, str] = {}
    if not path.exists():
        return labels
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            labels[int(row["index_in_sae"])] = row["label"]
    return labels


def label_of(labels: dict[int, str], idx: int) -> str:
    return labels.get(idx, f"feature_{idx}")


def search_labels(labels: dict[int, str], keyword: str, limit: int = 40) -> list[tuple[int, str]]:
    kw = keyword.lower()
    hits = [(i, l) for i, l in labels.items() if kw in l.lower()]
    return sorted(hits)[:limit]


@torch.no_grad()
def relevance_exclusion(engine, prompts: list[str], top_k: int = 100, force: bool = False) -> set[int]:
    """Latents in the top-`top_k` (by max activation over tokens) of any *unsteered* prompt.
    These are excluded from steering so the steering concept is genuinely off-topic."""
    path = DATA / "cache" / f"{engine.spec.key}_relevance_top{top_k}.json"
    cached = read_json(path)
    if cached and not force and cached.get("n_prompts") == len(prompts):
        return set(cached["exclude"])
    exclude: set[int] = set()
    per_prompt: dict[str, list[int]] = {}
    for p in prompts:
        acts, _ = engine.sae_activations(p)
        # Skip <bos> (position 0): its activations are huge and identical for every prompt, so
        # they would swamp the per-prompt top-k and make the exclusion set prompt-independent.
        top = torch.topk(acts[1:].max(dim=0).values, k=top_k).indices.tolist()
        per_prompt[p] = top
        exclude.update(top)
    write_json(path, {"n_prompts": len(prompts), "top_k": top_k, "exclude": sorted(exclude), "per_prompt": per_prompt})
    return exclude


def sample_latents(
    engine,
    prompts: list[str],
    labels: dict[int, str],
    n: int,
    *,
    seed: int = 0,
    relevance_top_k: int = 100,
    concreteness_judge=None,
    min_concreteness: float = 65.0,
    pool_multiplier: int = 3,
    force: bool = False,
) -> list[int]:
    """Sample `n` steering latents: random -> drop prompt-relevant -> (optionally) keep only
    concrete labels. Cached per (model, seed, n)."""
    path = DATA / "cache" / f"{engine.spec.key}_latents_seed{seed}_n{n}.json"
    cached = read_json(path)
    if cached and not force:
        return cached["latents"]

    rng = random.Random(seed)
    exclude = relevance_exclusion(engine, prompts, top_k=relevance_top_k)
    # Skip dead latents (never labelled) when labels exist: unlabeled ids usually never fire.
    universe = sorted(labels.keys()) if labels else list(range(engine.d_sae))
    universe = [i for i in universe if i not in exclude]
    chosen: list[int] = []
    concreteness: dict[int, float] = {}
    tried: set[int] = set()
    while len(chosen) < n and len(tried) < len(universe):
        need = (n - len(chosen)) * pool_multiplier
        pool = [i for i in rng.sample(universe, min(len(universe), need + len(tried))) if i not in tried][:need]
        tried.update(pool)
        if concreteness_judge is not None and labels:
            ratings = concreteness_judge.concreteness([labels[i] for i in pool])
            for i in pool:
                r = ratings.get(labels[i])
                if r is None:
                    continue
                concreteness[i] = r
                if r >= min_concreteness:
                    chosen.append(i)
        else:
            chosen.extend(pool)
    chosen = chosen[:n]
    write_json(
        path,
        {
            "latents": chosen,
            "labels": {i: label_of(labels, i) for i in chosen},
            "concreteness": {i: concreteness.get(i) for i in chosen},
            "seed": seed,
            "min_concreteness": min_concreteness if concreteness_judge else None,
            "n_excluded_relevant": len(exclude),
        },
    )
    return chosen


def random_latents_matched(
    n: int, stats_path: Path | None, exclude: set[int], d_sae: int, seed: int
) -> list[int]:
    """Random control latents. If a detectors json with baseline activation stats exists,
    sample latents whose on-topic activation frequency is in the same range as the detectors;
    otherwise uniform random."""
    rng = random.Random(seed)
    stats = read_json(stats_path) if stats_path else None
    if stats and "all_stats" in stats and stats.get("detectors"):
        det = stats["detectors"]
        freqs = [stats["all_stats"][str(i)]["off_nonzero_frac"] for i in det if str(i) in stats["all_stats"]]
        lo, hi = min(freqs), max(freqs)
        pool = [
            int(i)
            for i, s in stats["all_stats"].items()
            if int(i) not in exclude and int(i) not in det and lo <= s["off_nonzero_frac"] <= hi
        ]
        if len(pool) >= n:
            return sorted(rng.sample(pool, n))
    pool = [i for i in range(d_sae) if i not in exclude]
    return sorted(rng.sample(pool, n))
