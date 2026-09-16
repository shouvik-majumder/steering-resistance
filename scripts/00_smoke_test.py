"""Smoke test: load SAE (+ model), verify the hook point, SAE reconstruction, and generation.

  python scripts/00_smoke_test.py --sae-only            # no HF token needed
  python scripts/00_smoke_test.py --model gemma-2b      # needs HF_TOKEN + Gemma license
"""
from __future__ import annotations

import time

import _common  # noqa: F401  (sets HF_HOME)
import torch
from _common import MODELS, base_parser, make_engine

from esr.config import ExperimentConfig


def main() -> None:
    ap = base_parser("ESR smoke test")
    ap.add_argument("--sae-only", action="store_true")
    ap.add_argument("--prompt", default="Explain how to calculate probability.")
    args = ap.parse_args()
    args.max_new_tokens = min(args.max_new_tokens, 160)
    spec = MODELS[args.model]
    prompts = ExperimentConfig(model=args.model).prompts()

    t0 = time.perf_counter()
    engine = make_engine(args, prompts, load_model=not args.sae_only)
    print(f"loaded in {time.perf_counter() - t0:.1f}s")

    sae = engine.sae
    meta = getattr(sae.cfg, "metadata", None)
    hook_name = getattr(meta, "hook_name", None) or getattr(sae.cfg, "hook_name", "?")
    arch = getattr(sae.cfg, "architecture", None)
    arch = arch() if callable(arch) else arch
    print(f"SAE {spec.sae_release}/{spec.sae_id}: hook={hook_name} arch={arch} "
          f"d_sae={engine.d_sae} d_model={engine.d_model}")
    norms = engine.W_dec.norm(dim=-1)
    print(f"W_dec row norms: median={norms.median():.3f} min={norms.min():.3f} max={norms.max():.3f}")
    assert f"blocks.{spec.layer}." in str(hook_name), f"SAE hook {hook_name} != layer {spec.layer}"
    if args.sae_only:
        print("SAE-only smoke test OK")
        return

    print(engine.vram_report())

    # 1) chat template prefix property (prompt-only ids are a prefix of prompt+response ids)
    a = engine.encode_text(engine.chat_text(args.prompt))
    b = engine.encode_text(engine.chat_text(args.prompt, "Hello there."))
    assert torch.equal(a[0], b[0, : a.shape[1]]), "chat prefix mismatch"
    print(f"chat prefix OK ({a.shape[1]} prompt tokens); template:\n{engine.chat_text(args.prompt)!r}")

    # 2) hook output == hidden_states[layer+1]
    with torch.no_grad():
        out = engine.model(a, output_hidden_states=True, use_cache=False)
    ref = out.hidden_states[spec.layer + 1][0]
    hooked = engine.residuals(a)
    print(f"hook vs hidden_states[{spec.layer + 1}] max abs diff = {(ref.float() - hooked.float()).abs().max():.4f}")

    # 3) SAE reconstruction quality on this prompt (skip <bos>)
    ev = engine.explained_variance(hooked[1:])
    acts = sae.encode(hooked[1:].float())
    l0 = (acts > 0).float().sum(-1).mean().item()
    print(f"SAE explained variance = {ev:.3f}, mean L0 = {l0:.1f}  (expect EV ~0.7-0.9)")

    # 4) unsteered generation
    g = engine.generate(args.prompt, seed=args.seed, max_new_tokens=args.max_new_tokens, temperature=args.temperature)
    print(f"\n--- unsteered ({g.n_new_tokens} tokens, {g.seconds:.1f}s, {g.n_new_tokens / g.seconds:.1f} tok/s, eos={g.hit_eos}) ---")
    print(g.response)

    # 5) steered generation with a labelled latent, boost 1.0 in unit-scale units
    from esr.features import load_labels, search_labels

    labels = load_labels(spec)
    hits = search_labels(labels, "body") if labels else []
    if hits:
        latent, label = hits[0]
        g2 = engine.generate(args.prompt, latent=latent, boost=1.0, seed=args.seed, max_new_tokens=args.max_new_tokens)
        print(f"\n--- steered latent {latent} ({label!r}) boost=1.0 ---")
        print(g2.response)
    print("\n" + engine.vram_report())
    print("smoke test OK")


if __name__ == "__main__":
    main()
