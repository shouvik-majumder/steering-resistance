"""Calibrate per-latent steering thresholds (first-attempt score ~ 30/100).

  python scripts/02_calibrate.py --latents 1642,996 --n-trials 20
  python scripts/02_calibrate.py --n 10               # sample 10 filtered latents and calibrate
"""
from __future__ import annotations

import _common  # noqa: F401
from _common import MODELS, base_parser, config_from_args, make_engine, make_judge_or_none, parse_int_list

from esr.calibrate import calibrate_latent
from esr.features import load_labels, sample_latents
from esr.judge import is_scoring_judge


def main() -> None:
    ap = base_parser("Threshold calibration", default_judge="local")
    ap.add_argument("--latents", help="Comma-separated latent ids")
    ap.add_argument("--n", type=int, default=0, help="Sample this many filtered latents instead")
    ap.add_argument("--n-trials", type=int, default=20)
    ap.add_argument("--samples-per-trial", type=int, default=1)
    ap.add_argument("--no-concreteness", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if not is_scoring_judge(args.judge):
        ap.error("calibration needs a scoring judge (--judge local or anthropic)")

    cfg = config_from_args(args)
    cfg.threshold_n_trials = args.n_trials
    cfg.threshold_samples_per_trial = args.samples_per_trial
    prompts = cfg.prompts()
    labels = load_labels(MODELS[args.model])
    engine = make_engine(args, prompts)
    judge = make_judge_or_none(args, engine)

    latents = parse_int_list(args.latents)
    if not latents:
        latents = sample_latents(
            engine, prompts, labels, args.n or cfg.n_latents, seed=cfg.seed,
            relevance_top_k=cfg.relevance_top_k,
            concreteness_judge=None if args.no_concreteness else judge,
            min_concreteness=cfg.min_concreteness,
        )
        print(f"sampled {len(latents)} latents: {latents}")
    for latent in latents:
        calibrate_latent(engine, judge, prompts, labels, latent, cfg, force=args.force)
    print(engine.vram_report())


if __name__ == "__main__":
    main()
