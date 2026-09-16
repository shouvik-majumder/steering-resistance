"""(Re-)judge saved results without the steered model in memory.

Use this when the target model and the judge do not fit on the GPU together (e.g. Gemma-2-9B):
run 03_run_esr.py with --judge none (or --judge regex) first, then

  python scripts/05_judge_results.py --results data/results/gemma-9b_steered_seed0.jsonl --judge local
  python scripts/05_judge_results.py --results "data/results/*.jsonl" --judge local --judge-model Qwen/Qwen2.5-3B-Instruct

Rows already graded by a scoring judge are skipped unless --force is given.
"""
from __future__ import annotations

import glob
from pathlib import Path

import _common  # noqa: F401
from _common import base_parser, make_judge_or_none
from tqdm import tqdm

from esr.io import read_jsonl
from esr.metrics import format_summary, summarize


def main() -> None:
    ap = base_parser("Judge saved results", default_judge="local")
    ap.add_argument("--results", required=True, help="Path or glob of results .jsonl files")
    ap.add_argument("--force", action="store_true", help="Re-grade rows that already have scores")
    args = ap.parse_args()
    if args.judge == "none":
        ap.error("pick a judge")
    judge = make_judge_or_none(args)

    for f in sorted(glob.glob(args.results)):
        path = Path(f)
        rows = list(read_jsonl(path))
        todo = [
            r for r in rows
            if args.force
            or not r.get("judge")
            or r["judge"].get("error")
            or (r["judge"].get("judge", "").startswith("regex") and args.judge != "regex")
        ]
        print(f"{path.name}: {len(rows)} rows, grading {len(todo)}")
        for i, r in enumerate(tqdm(todo, desc=path.stem)):
            r["judge"] = judge.grade(r["prompt"], r["response"], r.get("label", "")).to_dict()
            if (i + 1) % 20 == 0:
                _write(path, rows)
        _write(path, rows)
        print(format_summary(path.stem, summarize(rows)))


def _write(path: Path, rows: list[dict]) -> None:
    import json

    tmp = path.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)


if __name__ == "__main__":
    main()
