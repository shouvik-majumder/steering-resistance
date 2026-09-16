"""Shared argparse helpers for the scripts. Import this first: it sets HF_HOME via esr.config."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from esr.config import META_PROMPT_DEFAULT, MODELS, ROOT, ExperimentConfig  # noqa: E402

JUDGE_CHOICES = ["local", "regex", "anthropic", "none"]


def base_parser(desc: str, default_judge: str = "regex") -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=desc)
    ap.add_argument("--model", default="gemma-2b", choices=sorted(MODELS))
    ap.add_argument("--judge", default=default_judge, choices=JUDGE_CHOICES,
                    help="local = free open model on the GPU (default Qwen2.5-7B-Instruct); "
                         "regex = restart-phrase counting only; anthropic = paid API; none = generate only")
    ap.add_argument("--judge-model", default=None,
                    help="HF id for --judge local (e.g. Qwen/Qwen2.5-3B-Instruct) or Anthropic model id")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--meta-prompt",
        nargs="?",
        const=META_PROMPT_DEFAULT,
        default=None,
        help="Append a meta-prompt to every user prompt (flag alone uses the paper's best variant).",
    )
    ap.add_argument("--raw-boost", action="store_true", help="Boost in raw W_dec units instead of residual-norm units.")
    return ap


def config_from_args(args) -> ExperimentConfig:
    cfg = ExperimentConfig(model=args.model)
    cfg.judge = args.judge
    cfg.judge_model = args.judge_model or ""
    cfg.max_new_tokens = args.max_new_tokens
    cfg.temperature = args.temperature
    cfg.seed = args.seed
    cfg.meta_prompt = args.meta_prompt
    return cfg


def make_engine(args, prompts: list[str], load_model: bool = True):
    from esr.model import SteeringEngine

    engine = SteeringEngine(MODELS[args.model], load_model=load_model)
    if load_model and not args.raw_boost:
        scale = engine.ensure_unit_scale(prompts)
        print(f"unit_scale (median resid norm @ layer {engine.spec.layer}) = {scale:.2f}")
    return engine


def make_judge_or_none(args):
    if args.judge == "none":
        return None
    from esr.judge import make_judge

    judge = make_judge(args.judge, model=args.judge_model)
    print(f"judge: {judge.name}")
    return judge


def parse_int_list(s: str | None) -> list[int]:
    if not s:
        return []
    return [int(x) for x in s.replace(";", ",").split(",") if x.strip()]


def parse_float_list(s: str | None) -> list[float]:
    if not s:
        return []
    return [float(x) for x in s.replace(";", ",").split(",") if x.strip()]


def resolve_prompt(arg: str, prompts: list[str]) -> str:
    if arg.isdigit() and int(arg) < len(prompts):
        return prompts[int(arg)]
    return arg
