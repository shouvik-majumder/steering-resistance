"""Hot-spot replay: a paired, higher-power test of the detector-ablation claim.

The full protocol's restart rate on Gemma-2-9B is ~1.4%, so a 25% reduction under ablation is
undetectable at feasible trial counts. Instead we take the (prompt, latent, boost) configurations
that already produced explicit restart language, re-sample each with fresh seeds, and for every
seed generate under three conditions:

  none      steering only
  ablate    steering + zero-ablation of the detector latents (from 04_find_detectors.py)
  random    steering + zero-ablation of the same number of random activity-matched latents

Compare restart-language and multi-attempt rates across conditions (paired by seed).

  python scripts/07_hotspot_replay.py --model gemma-9b --judge self \
      --detectors data/detectors/gemma-9b_seed0_response.json --seeds 30
  python scripts/07_hotspot_replay.py ... --max-configs 1 --seeds 10        # pilot
"""
from __future__ import annotations

import glob
import time
from pathlib import Path

import _common  # noqa: F401
from _common import MODELS, base_parser, config_from_args, make_engine, make_judge_or_none

from esr.config import DATA
from esr.features import load_labels, random_latents_matched
from esr.io import append_jsonl, done_keys, read_json, read_jsonl, trial_key
from esr.judge import restart_clusters
from esr.metrics import format_summary, summarize_by


def find_hotspots(results_glob: str, model: str) -> list[dict]:
    configs: dict[tuple, dict] = {}
    for f in glob.glob(results_glob):
        for r in read_jsonl(Path(f)):
            if r.get("model") != model or not r.get("response"):
                continue
            n_events = len(restart_clusters(r["response"]))
            if n_events == 0:
                continue
            key = (r["prompt"], r["latent"], round(r["boost"], 3), r.get("meta_prompt"))
            c = configs.setdefault(key, {"prompt": r["prompt"], "latent": r["latent"], "boost": r["boost"],
                                         "meta_prompt": r.get("meta_prompt"), "label": r.get("label", ""),
                                         "events": 0, "source_seeds": []})
            c["events"] += n_events
            c["source_seeds"].append(r["seed"])
    return sorted(configs.values(), key=lambda c: -c["events"])


def main() -> None:
    ap = base_parser("Hot-spot replay with detector / random ablation", default_judge="self")
    ap.add_argument("--results", default=str(DATA / "results" / "gemma-9b_*.jsonl"),
                    help="Glob of result files to mine for hot-spot configurations")
    ap.add_argument("--detectors", required=True, help="Detectors json from 04_find_detectors.py")
    ap.add_argument("--ablate-set", default="top_by_cohen_d", choices=["detectors", "top_by_cohen_d"])
    ap.add_argument("--seeds", type=int, default=30, help="Fresh seeds per configuration")
    ap.add_argument("--max-configs", type=int, default=0, help="Use only the top-N configurations (0 = all)")
    ap.add_argument("--random-seed", type=int, default=1)
    ap.add_argument("--conditions", default="none,ablate,random")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.judge == "none":
        ap.error("pick a judge (self recommended for 9B)")

    cfg = config_from_args(args)
    prompts = cfg.prompts()
    spec = MODELS[args.model]
    labels = load_labels(spec)
    engine = make_engine(args, prompts)
    judge = make_judge_or_none(args, engine)

    det = read_json(Path(args.detectors))
    det_set = [int(i) for i in det[args.ablate_set]]
    rand_set = random_latents_matched(len(det_set), Path(args.detectors), set(det_set), engine.d_sae, args.random_seed)
    print(f"detector set ({args.ablate_set}): {len(det_set)} latents; random control: {len(rand_set)} latents")

    hotspots = find_hotspots(args.results, args.model)
    if args.max_configs:
        hotspots = hotspots[: args.max_configs]
    print(f"{len(hotspots)} hot-spot configurations:")
    for i, c in enumerate(hotspots):
        print(f"  [{i}] latent {c['latent']} ({c['label'][:30]!r}) boost={c['boost']:.2f} events={c['events']} "
              f"meta={'yes' if c['meta_prompt'] else 'no'} prompt={c['prompt'][:45]!r}")

    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    stem = args.out or f"{args.model}_hotspot_seed{cfg.seed}"
    out_path = DATA / "results" / f"{stem}.jsonl"
    done = done_keys(out_path)
    print(f"results -> {out_path} ({len(done)} done)")

    t0 = time.perf_counter()
    n_new = 0
    for ci, c in enumerate(hotspots):
        ablate_sets = {"none": None, "ablate": [i for i in det_set if i != c["latent"]],
                       "random": [i for i in rand_set if i != c["latent"]]}
        for s in range(args.seeds):
            seed = 1_000_000 + 7919 * ci + s  # fresh seeds, disjoint from the original trials
            for cond in conditions:
                row = {"prompt": c["prompt"], "latent": c["latent"], "seed": seed,
                       "condition": f"hotspot_{cond}", "boost": c["boost"]}
                if trial_key(row) in done:
                    continue
                gen = engine.generate(c["prompt"], latent=c["latent"], boost=c["boost"], ablate=ablate_sets[cond],
                                      seed=seed, max_new_tokens=cfg.max_new_tokens, temperature=cfg.temperature,
                                      meta_prompt=c["meta_prompt"])
                res = judge.grade(c["prompt"], gen.response, c["label"])
                n_events = len(restart_clusters(gen.response))
                row.update({"model": args.model, "label": c["label"], "meta_prompt": c["meta_prompt"],
                            "config_id": ci, "ablated": ablate_sets[cond], "response": gen.response,
                            "n_new_tokens": gen.n_new_tokens, "hit_eos": gen.hit_eos,
                            "gen_seconds": round(gen.seconds, 2), "judge": res.to_dict(), "ts": time.time()})
                append_jsonl(out_path, row)
                n_new += 1
                flag = f"  <-- RESTART x{n_events}" if n_events else ""
                print(f"[cfg {ci} seed {s} {cond:<6}] tokens={gen.n_new_tokens} {gen.seconds:.0f}s "
                      f"attempts={len(res.attempts)} scores={[round(a.score) if a.score is not None else None for a in res.attempts]}{flag}")
            if (s + 1) % 5 == 0:
                print(f"  -- {n_new} new generations, {(time.perf_counter() - t0) / 60:.0f} min --")
                for k, summ in summarize_by(list(read_jsonl(out_path)), "condition").items():
                    print("  " + format_summary(k, summ))
    print("\nFINAL")
    for k, summ in summarize_by(list(read_jsonl(out_path)), "condition").items():
        print(format_summary(k, summ))
    print(engine.vram_report())


if __name__ == "__main__":
    main()
