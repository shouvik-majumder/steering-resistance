"""Project configuration: paths, model/SAE registry, experiment settings.

Import this module before `transformers`: it sets HF_HOME to `.hf_cache/` in the repository
unless HF_HOME is already set.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
load_dotenv(ROOT / ".env")

# Keep model + SAE weights on D: (3.7 TB) instead of the C: user profile.
os.environ.setdefault("HF_HOME", str(ROOT / ".hf_cache"))
# Gemma-2 on Windows: avoid the symlink warning noise from huggingface_hub.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

for _sub in ("labels", "thresholds", "results", "detectors", "cache"):
    (DATA / _sub).mkdir(parents=True, exist_ok=True)

# Windows consoles default to cp1252; steered outputs contain arbitrary Unicode.
import sys as _sys  # noqa: E402

for _stream in (_sys.stdout, _sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


@dataclass(frozen=True)
class ModelSpec:
    """One (model, SAE, steering layer) combination.

    Layers follow Table 2 of the paper: Gemma-2-2B-it -> layer 16 (61.5% depth),
    Gemma-2-9B-it -> layer 26 (61.9% depth). Gemma Scope residual SAEs are trained on the
    *pretrained* models but the paper applies them to the -it models (no IT SAE exists for 2B).
    """

    key: str
    hf_id: str
    sae_release: str
    sae_id: str
    layer: int
    n_layers: int
    labels_csv: str
    neuronpedia_source: str  # Neuronpedia model/source ids used for the labels
    repetition_penalty: float = 1.1  # paper: 1.1 for Gemma models
    approx_vram_gb: float = 0.0


MODELS: dict[str, ModelSpec] = {
    "gemma-2b": ModelSpec(
        key="gemma-2b",
        hf_id="google/gemma-2-2b-it",
        sae_release="gemma-scope-2b-pt-res-canonical",
        sae_id="layer_16/width_16k/canonical",
        layer=16,
        n_layers=26,
        labels_csv="data/labels/gemma-2-2b_layer16_res16k.csv",
        neuronpedia_source="gemma-2-2b/16-gemmascope-res-16k",
        approx_vram_gb=6.5,
    ),
    "gemma-9b": ModelSpec(
        key="gemma-9b",
        hf_id="google/gemma-2-9b-it",
        sae_release="gemma-scope-9b-pt-res-canonical",
        sae_id="layer_26/width_16k/canonical",
        layer=26,
        n_layers=42,
        labels_csv="data/labels/gemma-2-9b_layer26_res16k.csv",
        neuronpedia_source="gemma-2-9b/26-gemmascope-res-16k",
        approx_vram_gb=20.0,
    ),
}

# Meta-prompt that gave the largest ESR increase in the paper (Sec. 3.3 / App. A.3.2).
META_PROMPT_DEFAULT = (
    "If you notice yourself going off-topic, stop and force yourself to get back on track."
)


@dataclass
class ExperimentConfig:
    """Settings for one run of the ESR protocol. Defaults mirror the reference repo."""

    model: str = "gemma-2b"
    prompts_file: str = "prompts.txt"
    judge: str = "local"  # "local" (free, on-GPU) | "regex" | "anthropic" (paper: Claude 4.5 Haiku, paid)
    judge_model: str = ""  # "" = judge default (Qwen/Qwen2.5-7B-Instruct or claude-haiku-4-5)

    # Generation
    max_new_tokens: int = 512
    temperature: float = 0.6
    meta_prompt: str | None = None

    # Steering strength calibration (probabilistic bisection to a 30/100 first-attempt score)
    target_score: float = 0.30
    threshold_n_trials: int = 20
    threshold_samples_per_trial: int = 1
    # Boost is expressed in units of the median residual-stream norm at the steering layer
    # (see SteeringEngine.unit_scale), so 1.0 == "add a vector as long as the residual itself".
    # Observed on Gemma-2-2B: 0.25 on-topic, 0.5 derailed, 0.75 fully off-topic, 1.0 degenerate,
    # so the 30/100 threshold sits around 0.4-0.6.
    threshold_lower: float = 0.0
    threshold_upper: float = 3.0
    threshold_prior_mean: float = 0.6
    threshold_prior_std: float = 0.3

    # Latent sampling
    n_latents: int = 80
    n_trials_per_latent: int = 10
    relevance_top_k: int = 100  # exclude latents in the top-100 for any unsteered prompt
    min_concreteness: float = 65.0
    seed: int = 0

    # Ablation (path to a detectors json, or None)
    ablate_file: str | None = None
    disable_steering: bool = False

    def spec(self) -> ModelSpec:
        return MODELS[self.model]

    def prompts(self) -> list[str]:
        text = (ROOT / self.prompts_file).read_text(encoding="utf-8")
        return [line.strip() for line in text.splitlines() if line.strip()]

    def to_dict(self) -> dict:
        return asdict(self)

    def dump(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
