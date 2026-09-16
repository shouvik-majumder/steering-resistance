"""Contrastive search for off-topic-detector / self-correction-associated latents (Sec. 2.3).

  python scripts/04_find_detectors.py --model gemma-2b --pool response
"""
from __future__ import annotations

import _common  # noqa: F401
from _common import MODELS, base_parser, make_engine
from tqdm import tqdm

from esr.config import DATA, ExperimentConfig
from esr.detectors import find_detectors
from esr.features import label_of, load_labels
from esr.io import read_json, write_json


def main() -> None:
    ap = base_parser("Find off-topic detector latents")
    ap.add_argument("--pool", default="response", choices=["response", "all"])
    ap.add_argument("--min-offtopic-frac", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=26)
    ap.add_argument("--force", action="store_true", help="Regenerate the unsteered responses")
    args = ap.parse_args()

    cfg = ExperimentConfig(model=args.model)
    prompts = cfg.prompts()
    spec = MODELS[args.model]
    labels = load_labels(spec)
    engine = make_engine(args, prompts)

    # Step 1: unsteered responses (cached)
    resp_path = DATA / "cache" / f"{args.model}_unsteered_responses_seed{args.seed}.json"
    responses = {} if args.force else (read_json(resp_path) or {})
    for p in tqdm(prompts, desc="unsteered responses"):
        if p in responses:
            continue
        responses[p] = engine.generate(p, seed=args.seed, max_new_tokens=args.max_new_tokens, temperature=args.temperature).response
        write_json(resp_path, responses)

    # Steps 2-4
    result = find_detectors(
        engine, prompts, responses, seed=args.seed, pool=args.pool,
        min_offtopic_frac=args.min_offtopic_frac, top_k=args.top_k, progress=tqdm,
    )
    result["labels"] = {str(i): label_of(labels, i) for i in set(result["detectors"]) | set(result["top_by_cohen_d"])}
    out = DATA / "detectors" / f"{args.model}_seed{args.seed}_{args.pool}.json"
    write_json(out, result)

    print(f"\n{len(result['detectors'])} strict detectors (zero on-topic, >= {args.min_offtopic_frac:.0%} off-topic):")
    for i in result["detectors"][:40]:
        s = result["all_stats"][str(i)]
        print(f"  {i:6d} off={s['off_nonzero_frac']:.2f} mean={s['off_mean']:.3f} d={s['cohen_d']:.2f}  {label_of(labels, i)}")
    print(f"\nTop {args.top_k} by Cohen's d (off-topic vs on-topic):")
    for i in result["top_by_cohen_d"]:
        s = result["all_stats"][str(i)]
        print(f"  {i:6d} d={s['cohen_d']:.2f} p={s['p_welch']:.1e} off={s['off_mean']:.3f} on={s['on_mean']:.3f}  {label_of(labels, i)}")
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
