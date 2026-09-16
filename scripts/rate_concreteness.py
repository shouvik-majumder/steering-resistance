"""Rate label concreteness for a pool of candidate latents with the judge alone (no target model).

Needed when the target model and the judge do not fit on the GPU together (Gemma-2-9B):
run this first, then 03_run_esr.py --judge none will use the cached ratings when sampling.

  python scripts/rate_concreteness.py --model gemma-9b --pool 120 --judge local
"""
from __future__ import annotations

import random

import _common  # noqa: F401
from _common import MODELS, base_parser, make_judge_or_none

from esr.config import DATA
from esr.features import concreteness_cache_path, load_labels
from esr.io import read_json, write_json


def main() -> None:
    ap = base_parser("Rate latent-label concreteness", default_judge="local")
    ap.add_argument("--pool", type=int, default=120, help="Number of random labelled latents to rate")
    ap.add_argument("--exclude-relevant", action="store_true", default=True)
    args = ap.parse_args()

    spec = MODELS[args.model]
    labels = load_labels(spec)
    rel = read_json(DATA / "cache" / f"{args.model}_relevance_top100.json") or {}
    exclude = set(rel.get("exclude", []))
    universe = [i for i in sorted(labels) if i not in exclude]
    rng = random.Random(args.seed)
    pool = rng.sample(universe, min(args.pool, len(universe)))

    path = concreteness_cache_path(spec.key)
    cache: dict = read_json(path) or {}
    todo = [i for i in pool if str(i) not in cache]
    print(f"{len(pool)} candidates, {len(todo)} to rate (excluded {len(exclude)} prompt-relevant)")
    judge = make_judge_or_none(args)
    ratings = judge.concreteness([labels[i] for i in todo])
    for i in todo:
        r = ratings.get(labels[i])
        if r is not None:
            cache[str(i)] = {"label": labels[i], "rating": r}
    write_json(path, cache)
    kept = sorted((v["rating"], int(k)) for k, v in cache.items())
    print(f"saved {len(cache)} ratings -> {path}")
    print("lowest:", [(i, round(r), cache[str(i)]["label"][:40]) for r, i in kept[:5]])
    print("highest:", [(i, round(r), cache[str(i)]["label"][:40]) for r, i in kept[-5:]])
    print(f"{sum(1 for r, _ in kept if r >= 65)} latents with rating >= 65")


if __name__ == "__main__":
    main()
