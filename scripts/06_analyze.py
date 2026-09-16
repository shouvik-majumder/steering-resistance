"""Summarise results: metrics table per file/condition, ESR episode dump, and plots.

  python scripts/06_analyze.py                       # all data/results/*.jsonl
  python scripts/06_analyze.py --results "data/results/gemma-2b_*.jsonl" --plots
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from esr.config import DATA  # noqa: E402
from esr.io import read_jsonl  # noqa: E402
from esr.judge import restart_phrases  # noqa: E402
from esr.metrics import format_summary, summarize, summarize_by, trial_outcome  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(DATA / "results" / "*.jsonl"))
    ap.add_argument("--by", default="condition", help="Secondary grouping key (condition|latent|prompt|label)")
    ap.add_argument("--plots", action="store_true")
    ap.add_argument("--episodes", action="store_true", help="Write multi-attempt episodes to markdown")
    args = ap.parse_args()

    files = sorted(glob.glob(args.results))
    if not files:
        print("no results found")
        return
    all_rows = []
    for f in files:
        rows = list(read_jsonl(Path(f)))
        all_rows.extend(rows)
        print(format_summary(Path(f).stem, summarize(rows)))
    print()
    for k, s in summarize_by(all_rows, args.by).items():
        print(format_summary(f"{args.by}={k}", s))

    episodes = [(r, trial_outcome(r)) for r in all_rows]
    episodes = [(r, o) for r, o in episodes if o and o["multi_attempt"]]
    print(f"\n{len(episodes)} multi-attempt episodes")
    if args.episodes or episodes:
        md = DATA / "results" / "episodes.md"
        with md.open("w", encoding="utf-8") as fh:
            for r, o in episodes:
                fh.write(f"## {r.get('model')} | {r.get('condition')} | latent {r.get('latent')} ({r.get('label')}) | boost {r.get('boost'):.2f}\n\n")
                fh.write(f"**Prompt:** {r['prompt']}  \n**Scores:** {[a['score'] for a in r['judge']['attempts']]}  "
                         f"**Restart phrases:** {restart_phrases(r['response'])}\n\n```\n{r['response']}\n```\n\n")
        print(f"episodes -> {md}")

    if args.plots:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        out = DATA / "plots"
        out.mkdir(exist_ok=True)
        groups = summarize_by(all_rows, args.by)
        names = list(groups)
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        for ax, key, title in zip(axes, ["multi_attempt", "conditional_improvement", "esr"],
                                  ["Multi-attempt %", "Improvement | multi %", "ESR rate %"]):
            vals = [100 * groups[n][key][0] for n in names]
            lo = [100 * (groups[n][key][0] - groups[n][key][1]) for n in names]
            hi = [100 * (groups[n][key][2] - groups[n][key][0]) for n in names]
            ax.bar(names, vals, yerr=[lo, hi], capsize=4)
            ax.set_title(title)
            ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        fig.savefig(out / "rates.png", dpi=150)
        deltas = [o["last_score"] - o["first_score"] for _, o in episodes if o["scored"]]
        if deltas:
            plt.figure(figsize=(6, 4))
            plt.hist(deltas, bins=20, range=(-100, 100))
            plt.axvline(0, color="r", ls="--")
            plt.xlabel("Score delta (last - first attempt)")
            plt.title("Multi-attempt responses")
            plt.tight_layout()
            plt.savefig(out / "score_delta.png", dpi=150)
        print(f"plots -> {out}")


if __name__ == "__main__":
    main()
