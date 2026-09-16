"""Contrastive search for self-correction-associated ("off-topic detector") latents.

Paper Sec. 2.3 / repo `find_off_topic_detectors.py`:
  1. Generate one unsteered response per prompt.
  2. Build mismatched pairs by a derangement (each prompt gets another prompt's response).
  3. Run matched and mismatched (prompt, response) chats through the model, max-pool SAE
     activations over tokens (response tokens by default).
  4. Detector criterion (repo): zero on all matched pairs, non-zero on >= 80% of mismatched.
     We also report Cohen's d and Welch's t-test per latent (paper Table 3) and a top-k by d.
"""
from __future__ import annotations

import math
import random

import numpy as np
import torch
from scipy.stats import ttest_ind


def derangement(n: int, rng: random.Random) -> list[int]:
    while True:
        perm = list(range(n))
        rng.shuffle(perm)
        if all(perm[i] != i for i in range(n)):
            return perm


@torch.no_grad()
def pooled_activations(engine, prompt: str, response: str, pool: str = "response") -> torch.Tensor:
    acts, resp_start = engine.sae_activations(prompt, response)
    if pool == "response":
        acts = acts[resp_start:]
    return acts.max(dim=0).values.cpu()


def cohens_d(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Effect size of a (off-topic) vs b (on-topic), vectorised over latents (axis 0 = samples)."""
    na, nb = a.shape[0], b.shape[0]
    va, vb = a.var(0, ddof=1), b.var(0, ddof=1)
    pooled = np.sqrt(((na - 1) * va + (nb - 1) * vb) / max(na + nb - 2, 1))
    with np.errstate(divide="ignore", invalid="ignore"):
        d = (a.mean(0) - b.mean(0)) / pooled
    return np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)


def find_detectors(
    engine,
    prompts: list[str],
    responses: dict[str, str],
    *,
    seed: int = 0,
    pool: str = "response",
    min_offtopic_frac: float = 0.8,
    eps: float = 1e-6,
    top_k: int = 26,
    progress=None,
) -> dict:
    rng = random.Random(seed)
    perm = derangement(len(prompts), rng)

    on_rows, off_rows = [], []
    it = enumerate(prompts)
    if progress:
        it = progress(list(it), desc="matched/mismatched pairs")
    for i, p in it:
        on_rows.append(pooled_activations(engine, p, responses[p], pool))
        off_rows.append(pooled_activations(engine, p, responses[prompts[perm[i]]], pool))
    on = torch.stack(on_rows).numpy()  # [N, d_sae]
    off = torch.stack(off_rows).numpy()

    on_nz = (on > eps).mean(0)
    off_nz = (off > eps).mean(0)
    on_mean, off_mean = on.mean(0), off.mean(0)
    d = cohens_d(off, on)
    with np.errstate(all="ignore"):
        p_vals = ttest_ind(off, on, axis=0, equal_var=False).pvalue
    p_vals = np.nan_to_num(p_vals, nan=1.0)

    strict = [int(i) for i in np.where((on_nz == 0) & (off_nz >= min_offtopic_frac))[0]]
    strict.sort(key=lambda i: (-off_nz[i], -off_mean[i]))
    active = np.where((off_nz > 0) | (on_nz > 0))[0]
    by_d = sorted(active.tolist(), key=lambda i: -d[i])[:top_k]

    all_stats = {
        str(int(i)): {
            "on_mean": float(on_mean[i]),
            "off_mean": float(off_mean[i]),
            "on_nonzero_frac": float(on_nz[i]),
            "off_nonzero_frac": float(off_nz[i]),
            "cohen_d": float(d[i]),
            "p_welch": float(p_vals[i]),
        }
        for i in active
    }
    return {
        "model": engine.spec.key,
        "sae": f"{engine.spec.sae_release}/{engine.spec.sae_id}",
        "n_prompts": len(prompts),
        "pool": pool,
        "seed": seed,
        "min_offtopic_frac": min_offtopic_frac,
        "detectors": strict,  # repo criterion
        "top_by_cohen_d": [int(i) for i in by_d],  # paper Table 3 style ranking
        "derangement": perm,
        "all_stats": all_stats,
    }
