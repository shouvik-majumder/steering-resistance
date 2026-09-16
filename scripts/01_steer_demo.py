"""Qualitative steering demo: one prompt x one latent x a sweep of boosts.

  python scripts/01_steer_demo.py --search "body pos"          # find latent ids by label
  python scripts/01_steer_demo.py --latent 1642 --prompt 2 --boosts 0,0.5,1,1.5,2,3 --judge regex   # or --judge local
"""
from __future__ import annotations

import _common  # noqa: F401
from _common import MODELS, base_parser, make_engine, make_judge_or_none, resolve_prompt

from esr.config import ExperimentConfig
from esr.features import label_of, load_labels, search_labels
from esr.judge import restart_phrases


def main() -> None:
    ap = base_parser("Steering demo")
    ap.add_argument("--search", help="Print latents whose label contains this keyword and exit.")
    ap.add_argument("--latent", type=int)
    ap.add_argument("--prompt", default="2", help="Prompt text or index into prompts.txt")
    ap.add_argument("--boosts", default="0,0.5,1,1.5,2,3")
    ap.add_argument("--ablate", help="Comma-separated latent ids to zero-ablate during generation.")
    args = ap.parse_args()

    spec = MODELS[args.model]
    labels = load_labels(spec)
    if args.search:
        for i, l in search_labels(labels, args.search, limit=60):
            print(f"{i:6d}  {l}")
        return
    if args.latent is None:
        ap.error("--latent is required (use --search to find one)")

    prompts = ExperimentConfig(model=args.model).prompts()
    prompt = resolve_prompt(args.prompt, prompts)
    engine = make_engine(args, prompts)
    judge = make_judge_or_none(args)
    label = label_of(labels, args.latent)
    ablate = [int(x) for x in args.ablate.split(",")] if args.ablate else None

    print(f"\nPrompt: {prompt}\nLatent {args.latent}: {label!r}\n")
    for b in [float(x) for x in args.boosts.split(",")]:
        g = engine.generate(
            prompt, latent=args.latent, boost=b, seed=args.seed, ablate=ablate,
            max_new_tokens=args.max_new_tokens, temperature=args.temperature, meta_prompt=args.meta_prompt,
        )
        print("=" * 100)
        print(f"boost={b:g}  tokens={g.n_new_tokens}  {g.seconds:.1f}s  eos={g.hit_eos}")
        print("-" * 100)
        print(g.response)
        phrases = restart_phrases(g.response)
        if phrases:
            print(f"\n>>> restart phrases: {phrases}")
        if judge is not None:
            res = judge.grade(prompt, g.response, label)
            if res.error:
                print(f"[judge error] {res.error}")
            else:
                print(f"[judge] attempts={len(res.attempts)} scores={[a.score for a in res.attempts]}")
    print(engine.vram_report())



if __name__ == "__main__":
    main()
