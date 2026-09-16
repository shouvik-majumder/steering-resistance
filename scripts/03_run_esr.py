"""Main ESR protocol (paper Sec. 2.1): sample latents -> calibrate -> steered trials -> judge.

  python scripts/03_run_esr.py --n-latents 20 --trials-per-latent 10                 # steered, local judge
  python scripts/03_run_esr.py --model gemma-9b --judge none --boosts 0.5,1,1.5,2      # generate only, judge later (05)
  python scripts/03_run_esr.py --n-latents 20 --trials-per-latent 10 --meta-prompt   # meta-prompted
  python scripts/03_run_esr.py --no-steer --n-latents 20 --trials-per-latent 5        # 0% control
  python scripts/03_run_esr.py --ablate data/detectors/gemma-2b_seed0.json            # ablation
  python scripts/03_run_esr.py --ablate data/detectors/gemma-2b_seed0.json --random-control 1

Resumable: rerunning with the same arguments skips completed (prompt, latent, seed) trials.
"""
from __future__ import annotations

import random
import time
from pathlib import Path

import _common  # noqa: F401
from _common import (MODELS, base_parser, config_from_args, make_engine, make_judge_or_none,
                     parse_float_list, parse_int_list)

from esr.calibrate import calibrate_latent
from esr.config import DATA
from esr.features import label_of, load_labels, random_latents_matched, sample_latents
from esr.io import append_jsonl, done_keys, read_json, trial_key
from esr.io import read_jsonl
from esr.judge import is_scoring_judge
from esr.metrics import format_summary, summarize


def main() -> None:
    ap = base_parser("Run the ESR protocol", default_judge="local")
    ap.add_argument("--n-latents", type=int, default=80)
    ap.add_argument("--trials-per-latent", type=int, default=10)
    ap.add_argument("--latents", help="Override sampled latents (comma-separated ids)")
    ap.add_argument("--no-steer", action="store_true", help="Baseline: no steering (expect 0%% multi-attempt)")
    ap.add_argument("--ablate", help="Detectors json from 04_find_detectors.py")
    ap.add_argument("--ablate-set", default="detectors", choices=["detectors", "top_by_cohen_d"])
    ap.add_argument("--random-control", type=int, default=None, metavar="SEED",
                    help="Instead of the detectors, ablate the same number of random matched latents")
    ap.add_argument("--no-concreteness", action="store_true")
    ap.add_argument("--n-calib-trials", type=int, default=20)
    ap.add_argument("--boosts", help="Skip calibration and run every latent at these fixed boosts "
                    "(comma-separated, unit-scale units). Default for the regex judge: 0.5,1,1.5,2")
    ap.add_argument("--out", help="Results file stem (default derived from the condition)")
    args = ap.parse_args()

    cfg = config_from_args(args)
    cfg.n_latents = args.n_latents
    cfg.n_trials_per_latent = args.trials_per_latent
    cfg.threshold_n_trials = args.n_calib_trials
    cfg.disable_steering = args.no_steer
    cfg.ablate_file = args.ablate
    prompts = cfg.prompts()
    spec = MODELS[args.model]
    labels = load_labels(spec)

    engine = make_engine(args, prompts)
    judge = make_judge_or_none(args, engine)

    # ---------------------------------------------------------------- ablation set
    ablate: list[int] | None = None
    condition = "no_steer" if args.no_steer else "steered"
    if args.ablate:
        det = read_json(Path(args.ablate))
        det_set = [int(i) for i in det[args.ablate_set]]
        if args.random_control is not None:
            ablate = random_latents_matched(len(det_set), Path(args.ablate), set(det_set), engine.d_sae, args.random_control)
            condition = f"random_ablation_s{args.random_control}"
        else:
            ablate = det_set
            condition = f"ablation_{args.ablate_set}"
        print(f"ablating {len(ablate)} latents: {ablate}")
    if args.meta_prompt:
        condition += "+meta"

    stem = args.out or f"{args.model}_{condition}_seed{cfg.seed}"
    out_path = DATA / "results" / f"{stem}.jsonl"
    cfg.dump(DATA / "results" / f"{stem}.config.json")
    done = done_keys(out_path)
    print(f"results -> {out_path}  ({len(done)} trials already done)")

    # ---------------------------------------------------------------- latents
    scoring = is_scoring_judge(args.judge)
    latents = parse_int_list(args.latents)
    if not latents:
        latents = sample_latents(
            engine, prompts, labels, cfg.n_latents, seed=cfg.seed, relevance_top_k=cfg.relevance_top_k,
            concreteness_judge=None if (args.no_concreteness or not scoring) else judge,
            min_concreteness=cfg.min_concreteness,
        )
    print(f"{len(latents)} latents")

    # Fixed boosts (no calibration): required for the regex judge, optional otherwise.
    fixed_boosts = parse_float_list(args.boosts)
    if not fixed_boosts and not scoring and not args.no_steer:
        fixed_boosts = [0.5, 1.0, 1.5, 2.0]
    if fixed_boosts:
        print(f"fixed boosts (no calibration): {fixed_boosts}")

    # ---------------------------------------------------------------- trials
    t_start = time.perf_counter()
    n_done_now = 0
    for li, latent in enumerate(latents):
        label = label_of(labels, latent)
        if args.no_steer:
            boosts = [0.0]
        elif fixed_boosts:
            boosts = fixed_boosts
        else:
            boosts = [calibrate_latent(engine, judge, prompts, labels, latent, cfg, verbose=False)]
        # Some steering latents may also be in the ablation set; the ablation hook would cancel
        # them, so drop the steered latent from the ablated set for this latent's trials.
        ablate_here = [i for i in ablate if i != latent] if ablate else None
        rows_for_latent = []
        for boost, t in ((b, t) for b in boosts for t in range(cfg.n_trials_per_latent)):
            r = random.Random(f"trial:{cfg.seed}:{latent}:{t}")
            prompt = r.choice(prompts)
            seed = r.randrange(1_000_000)
            row = {"prompt": prompt, "latent": latent, "seed": seed, "condition": condition, "boost": boost}
            if trial_key(row) in done:
                continue
            gen = engine.generate(
                prompt, latent=None if args.no_steer else latent, boost=boost, ablate=ablate_here, seed=seed,
                max_new_tokens=cfg.max_new_tokens, temperature=cfg.temperature, meta_prompt=cfg.meta_prompt,
            )
            res = judge.grade(prompt, gen.response, label) if judge is not None else None
            row.update({
                "model": args.model, "label": label, "meta_prompt": cfg.meta_prompt,
                "ablated": ablate_here, "response": gen.response, "n_new_tokens": gen.n_new_tokens,
                "hit_eos": gen.hit_eos, "gen_seconds": round(gen.seconds, 2), "judge": res.to_dict() if res else None,
                "ts": time.time(),
            })
            append_jsonl(out_path, row)
            rows_for_latent.append(row)
            n_done_now += 1
            n_att = len(res.attempts) if res else -1
            flag = "  <-- MULTI-ATTEMPT" if n_att >= 2 else ""
            scores = [a.score for a in res.attempts] if res else "n/a"
            print(f"[{li + 1}/{len(latents)} latent {latent} t{t}] boost={boost:.2f} tokens={gen.n_new_tokens} "
                  f"{gen.seconds:.0f}s attempts={n_att} scores={scores}{flag}")
        if rows_for_latent:
            elapsed = time.perf_counter() - t_start
            print(format_summary(f"  running ({n_done_now} new, {elapsed / 60:.0f} min)", summarize(list(read_jsonl(out_path)))))
    print("\nFINAL")
    print(format_summary(condition, summarize(list(read_jsonl(out_path)))))
    print(engine.vram_report())


if __name__ == "__main__":
    main()
