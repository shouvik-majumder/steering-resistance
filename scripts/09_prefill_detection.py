"""Prefill-detection experiment with paired ablation (adapted from paper Sec. 3.6).

Detection of off-topic text is largely text-conditioned: an *unsteered* model given the start of
an off-topic answer as its own words often notices and restarts (paper: 7-13.5%). That rate is
5-10x the steered restart rate, so it is where a causal test of the detector latents is feasible.

For each saved off-topic steered response (first-attempt score <= --max-first-score, no restart
inside the prefix), cut a prefix of ~--prefix-chars characters at a sentence boundary, prefill
it as the assistant's turn, and continue generation WITHOUT steering under three conditions,
paired by seed:

  none     plain continuation
  ablate   detector latents zero-ablated (04_find_detectors.py, --ablate-set)
  random   the same number of random activity-matched latents zero-ablated

Outcome: restart language in the continuation (regex, judge-verified sentences, judge attempts).

  python scripts/09_prefill_detection.py --model gemma-9b --judge self \
      --detectors data/detectors/gemma-9b_seed0_response.json --n-prefixes 200
  python scripts/09_prefill_detection.py ... --n-prefixes 10           # pilot
"""
from __future__ import annotations

import glob
import random
import re
import time
from pathlib import Path

import _common  # noqa: F401
from _common import MODELS, base_parser, config_from_args, make_engine, make_judge_or_none

from esr.config import DATA
from esr.features import random_latents_matched
from esr.io import append_jsonl, done_keys, read_json, read_jsonl, trial_key
from esr.judge import restart_clusters
from esr.metrics import format_summary, summarize_by


def cut_prefix(text: str, target: int) -> str | None:
    """Cut at the last sentence/line boundary before `target` chars (must keep >= 60% of target)."""
    if len(text) < target * 0.8:
        return None
    head = text[:target]
    cut = max(head.rfind(". "), head.rfind("\n"), head.rfind("! "), head.rfind("? "))
    if cut < target * 0.6:
        cut = target
    return text[: cut + 1].rstrip() + ("\n" if text[cut : cut + 1] == "\n" else " ")


def collect_prefixes(results_glob: str, model: str, prefix_chars: int, max_first_score: float, rng: random.Random) -> list[dict]:
    out = []
    for f in sorted(glob.glob(results_glob)):
        for r in read_jsonl(Path(f)):
            if r.get("model") != model or not r.get("response") or r.get("condition", "").startswith(("no_steer", "hotspot", "prefill")):
                continue
            j = r.get("judge") or {}
            att = j.get("attempts") or []
            first = att[0].get("score") if att else None
            if first is None or first > max_first_score:
                continue
            prefix = cut_prefix(r["response"], prefix_chars)
            if prefix is None or restart_clusters(prefix):
                continue
            out.append({"prompt": r["prompt"], "prefix": prefix, "source_latent": r["latent"], "source_label": r.get("label", ""),
                        "source_seed": r["seed"], "source_boost": r["boost"], "source_first_score": first, "source_file": Path(f).name})
    rng.shuffle(out)
    return out


def main() -> None:
    ap = base_parser("Prefill-detection experiment with paired ablation", default_judge="self")
    ap.add_argument("--results", default=str(DATA / "results" / "gemma-9b_*.jsonl"))
    ap.add_argument("--detectors", required=True)
    ap.add_argument("--ablate-set", default="top_by_cohen_d", choices=["detectors", "top_by_cohen_d"])
    ap.add_argument("--n-prefixes", type=int, default=200)
    ap.add_argument("--prefix-chars", type=int, default=1000)
    ap.add_argument("--max-first-score", type=float, default=30.0, help="Use only clearly off-topic source responses")
    ap.add_argument("--random-seed", type=int, default=1)
    ap.add_argument("--conditions", default="none,ablate,random")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    args.max_new_tokens = min(args.max_new_tokens, 400)

    cfg = config_from_args(args)
    prompts = cfg.prompts()
    engine = make_engine(args, prompts)
    judge = make_judge_or_none(args, engine)

    det = read_json(Path(args.detectors))
    det_set = [int(i) for i in det[args.ablate_set]]
    rand_set = random_latents_matched(len(det_set), Path(args.detectors), set(det_set), engine.d_sae, args.random_seed)
    ablate_sets = {"none": None, "ablate": det_set, "random": rand_set}
    print(f"detector set ({args.ablate_set}): {len(det_set)}; random control: {len(rand_set)}")

    rng = random.Random(cfg.seed)
    prefixes = collect_prefixes(args.results, args.model, args.prefix_chars, args.max_first_score, rng)[: args.n_prefixes]
    print(f"{len(prefixes)} off-topic prefixes (~{args.prefix_chars} chars, source first-attempt <= {args.max_first_score:g})")

    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    stem = args.out or f"{args.model}_prefill{args.prefix_chars}_seed{cfg.seed}"
    out_path = DATA / "results" / f"{stem}.jsonl"
    done = done_keys(out_path)
    print(f"results -> {out_path} ({len(done)} done)")

    t0 = time.perf_counter()
    n_new = 0
    for pi, pf in enumerate(prefixes):
        seed = 2_000_000 + pi
        for cond in conditions:
            row = {"prompt": pf["prompt"], "latent": pf["source_latent"], "seed": seed, "condition": f"prefill_{cond}", "boost": 0.0}
            if trial_key(row) in done:
                continue
            gen = engine.generate(pf["prompt"], latent=None, boost=0.0, ablate=ablate_sets[cond], seed=seed,
                                  max_new_tokens=args.max_new_tokens, temperature=cfg.temperature,
                                  meta_prompt=None, assistant_prefix=pf["prefix"])
            full = pf["prefix"] + gen.response
            res = judge.grade(pf["prompt"], full, pf["source_label"])
            # Primary outcome: relevance of the continuation alone (recovery is usually silent).
            cont_score, cont_raw = judge.grade_continuation(pf["prompt"], gen.response) if hasattr(judge, "grade_continuation") else (None, "")
            n_events = len(restart_clusters(gen.response))
            row.update({"model": args.model, "label": pf["source_label"], "prefix": pf["prefix"], "continuation": gen.response,
                        "response": full, "prefix_id": pi, "source": {k: pf[k] for k in ("source_seed", "source_boost", "source_first_score", "source_file")},
                        "ablated": ablate_sets[cond], "n_new_tokens": gen.n_new_tokens, "hit_eos": gen.hit_eos,
                        "gen_seconds": round(gen.seconds, 2), "judge": res.to_dict(),
                        "cont_score": cont_score, "cont_raw": cont_raw[-300:], "ts": time.time()})
            append_jsonl(out_path, row)
            n_new += 1
            flag = f"  <-- RESTART x{n_events}" if n_events else ("  <-- judge-sentence" if res.restart_sentences else "")
            print(f"[prefix {pi} {cond:<6}] tokens={gen.n_new_tokens} {gen.seconds:.0f}s cont_score={cont_score} "
                  f"attempts={len(res.attempts)} scores={[round(a.score) if a.score is not None else None for a in res.attempts]}{flag}")
        if (pi + 1) % 10 == 0:
            print(f"  -- {n_new} new generations, {(time.perf_counter() - t0) / 60:.0f} min --")
            print_summary(out_path)
    print("\nFINAL")
    print_summary(out_path)
    print(engine.vram_report())


def print_summary(out_path: Path) -> None:
    import math

    rows = list(read_jsonl(out_path))
    for k, summ in summarize_by(rows, "condition").items():
        print("  " + format_summary(k, summ))
    # Continuation relevance per condition, plus the paired difference vs 'none' on shared prefixes.
    by_cond: dict[str, dict[int, float]] = {}
    for r in rows:
        if r.get("cont_score") is not None:
            by_cond.setdefault(r["condition"], {})[r["prefix_id"]] = r["cont_score"]
    base = by_cond.get("prefill_none", {})
    for cond, d in sorted(by_cond.items()):
        vals = list(d.values())
        mean = sum(vals) / len(vals)
        sem = (math.sqrt(sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)) / math.sqrt(len(vals))) if len(vals) > 1 else float("nan")
        shared = [pid for pid in d if pid in base] if cond != "prefill_none" else []
        if shared:
            diffs = [d[pid] - base[pid] for pid in shared]
            dm = sum(diffs) / len(diffs)
            dsem = (math.sqrt(sum((x - dm) ** 2 for x in diffs) / (len(diffs) - 1)) / math.sqrt(len(diffs))) if len(diffs) > 1 else float("nan")
            paired = f"  paired diff vs none = {dm:+.1f} +/- {dsem:.1f} (n={len(diffs)})"
        else:
            paired = ""
        print(f"  {cond:<16} continuation relevance = {mean:5.1f} +/- {sem:.1f} (n={len(vals)}){paired}")


if __name__ == "__main__":
    main()
