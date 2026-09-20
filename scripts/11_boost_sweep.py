"""Boost sweep: how does the model's answer degrade as steering strength rises?

The paper's Figure 3 says self-correction only happens in a narrow band of steering strength:
too weak and nothing changes, too strong and the output is gibberish. We never plotted this for
our own models, so this script sweeps a few latents across boosts and records the judged
first-attempt score plus two judge-free degeneracy measures.

  python scripts/11_boost_sweep.py --model gemma-2b --latents 8747,12525,13958 \
      --boosts 0,0.25,0.5,0.75,1.0,1.5 --trials 4 --judge local
"""
from __future__ import annotations

import random
import time
from pathlib import Path

import _common  # noqa: F401
from _common import (MODELS, base_parser, config_from_args, make_engine,  # noqa: F401
                     make_judge_or_none, parse_float_list, parse_int_list)

from esr.config import DATA
from esr.features import label_of, load_labels
from esr.io import append_jsonl, done_keys, read_jsonl, trial_key
from esr.judge import restart_clusters


def degeneracy(text: str) -> dict:
    """Judge-free measures of output collapse: vocabulary diversity and repeated bigrams."""
    words = text.split()
    if len(words) < 10:
        return {"distinct_word_frac": 0.0, "repeated_bigram_frac": 1.0}
    bigrams = list(zip(words[:-1], words[1:]))
    return {
        "distinct_word_frac": len(set(words)) / len(words),
        "repeated_bigram_frac": 1.0 - len(set(bigrams)) / len(bigrams),
    }


def main() -> None:
    ap = base_parser("Boost sweep", default_judge="local")
    ap.add_argument("--latents", required=True)
    ap.add_argument("--boosts", default="0,0.25,0.5,0.75,1.0,1.5")
    ap.add_argument("--trials", type=int, default=4)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = config_from_args(args)
    prompts = cfg.prompts()
    labels = load_labels(MODELS[args.model])
    engine = make_engine(args, prompts)
    judge = make_judge_or_none(args, engine)

    latents = parse_int_list(args.latents)
    boosts = parse_float_list(args.boosts)
    out_path = DATA / "results" / f"{args.out or args.model + '_boost_sweep'}.jsonl"
    done = done_keys(out_path)
    print(f"{len(latents)} latents x {len(boosts)} boosts x {args.trials} trials -> {out_path} ({len(done)} done)")

    t0 = time.perf_counter()
    for latent in latents:
        label = label_of(labels, latent)
        for boost in boosts:
            for t in range(args.trials):
                r = random.Random(f"sweep:{latent}:{t}")
                prompt = r.choice(prompts)
                seed = 3_000_000 + r.randrange(100_000)
                row = {"prompt": prompt, "latent": latent, "seed": seed,
                       "condition": "boost_sweep", "boost": boost}
                if trial_key(row) in done:
                    continue
                gen = engine.generate(prompt, latent=latent if boost > 0 else None, boost=boost,
                                      seed=seed, max_new_tokens=cfg.max_new_tokens,
                                      temperature=cfg.temperature)
                res = judge.grade(prompt, gen.response, label) if judge else None
                first = res.first_score if res and res.attempts else None
                row.update({"model": args.model, "label": label, "response": gen.response,
                            "n_new_tokens": gen.n_new_tokens, "hit_eos": gen.hit_eos,
                            "gen_seconds": round(gen.seconds, 2),
                            "judge": res.to_dict() if res else None,
                            "first_score": first,
                            "n_restart_events": len(restart_clusters(gen.response)),
                            **degeneracy(gen.response), "ts": time.time()})
                append_jsonl(out_path, row)
                print(f"[latent {latent} boost {boost:>4} t{t}] first={first} "
                      f"distinct={row['distinct_word_frac']:.2f} rep={row['repeated_bigram_frac']:.2f} "
                      f"[{(time.perf_counter() - t0)/60:.0f} min]")
    print(f"\nDONE {len(list(read_jsonl(out_path)))} rows in {(time.perf_counter() - t0)/60:.0f} min")


if __name__ == "__main__":
    main()
