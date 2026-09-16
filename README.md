# ESR on a single GPU

Replication of *Endogenous Resistance to Activation Steering in Language Models*
(McKenzie et al., ICML 2026, [arXiv:2602.06941](https://arxiv.org/abs/2602.06941)) using
Gemma-2-2B/9B-it, Gemma Scope SAEs and plain HF `transformers` hooks. See [PLAN.md](PLAN.md)
for the design and the honest scope discussion.

## Setup (Windows, PowerShell)

```powershell
cd D:\dev\ESR
# environment was created with: python -m uv venv --python 3.12 .venv
#   python -m uv pip install --python .venv\Scripts\python.exe --torch-backend=cu128 -r <deps in pyproject.toml>
.\.venv\Scripts\Activate.ps1
Copy-Item .env.example .env      # then fill HF_TOKEN and ANTHROPIC_API_KEY
```

Gemma weights are gated: accept the license at https://huggingface.co/google/gemma-2-2b-it
with the account that owns `HF_TOKEN`.

## Workflow

Everything runs for free on the local GPU. The judge is an open instruct model
(`--judge local`, default `Qwen/Qwen2.5-7B-Instruct`, ~15 GB; use
`--judge-model Qwen/Qwen2.5-3B-Instruct` if it does not fit next to the target model).
`--judge regex` only counts explicit restart phrases (no scores, no calibration).
The paper's paid Claude judge is still available as `--judge anthropic` but is optional.

| Step | Command | Needs |
|---|---|---|
| 0. SAE loads, hook is right, generation works | `python scripts/00_smoke_test.py` (`--sae-only` without HF token) | GPU |
| 1. Look at steering by eye | `python scripts/01_steer_demo.py --search body` then `--latent <id> --boosts 0,0.5,1,2,3 --judge regex` | GPU |
| 2. Calibrate steering strength | `python scripts/02_calibrate.py --latents <ids>` | GPU (target model + local judge) |
| 3. Run the protocol | `python scripts/03_run_esr.py --n-latents 20 --trials-per-latent 10 [--meta-prompt]` | GPU |
| 3b. Big model: generate now, judge later | `python scripts/03_run_esr.py --model gemma-9b --judge none --boosts 0.5,1,1.5,2` then `python scripts/05_judge_results.py --results "data/results/gemma-9b_*.jsonl"` | GPU |
| 4. Find detector latents | `python scripts/04_find_detectors.py` | GPU |
| 5. Ablation + random control | `python scripts/03_run_esr.py --ablate data/detectors/gemma-2b_seed0_response.json` and `... --random-control 1` | GPU |
| 6. Metrics and plots | `python scripts/06_analyze.py --plots` | - |

All long runs append one JSON line per trial to `data/results/` and resume when re-run.
Thresholds are cached in `data/thresholds/<model>.json`, sampled latents and the relevance
filter in `data/cache/`.

## Layout

- `esr/config.py` model registry (paper layers), experiment config, paths
- `esr/model.py` `SteeringEngine`: HF model + SAE, steering/ablation forward hook, generate, activations
- `esr/judge.py` Claude Haiku judge with the paper's prompt; regex fallback
- `esr/threshold.py`, `esr/calibrate.py` probabilistic bisection to a 30/100 first-attempt score
- `esr/features.py` Neuronpedia labels, relevance/concreteness filters, latent sampling
- `esr/detectors.py` derangement contrastive search, Cohen's d
- `esr/metrics.py` multi-attempt / improvement / ESR rates, Wilson CIs
