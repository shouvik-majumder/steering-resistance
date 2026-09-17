"""Token-level traces of the detector latents through self-correction episodes (paper Sec. 3.8,
Fig. 7, App. A.4) -- no new generation needed.

For every saved response that contains explicit restart language, re-run the same steered
forward pass over (prompt + response), read the SAE activations at every token, and sum the
detector-latent activity per token. Mark where the restart phrases occur. Compare the mean
detector activity in the off-topic region (before the first restart), around the restart, and
after it, against steered responses of the same latents that never restarted (baseline).

  python scripts/08_episode_traces.py --model gemma-9b --detectors data/detectors/gemma-9b_seed0_response.json
"""
from __future__ import annotations

import glob
from pathlib import Path

import _common  # noqa: F401
import torch
from _common import MODELS, base_parser, make_engine

from esr.config import DATA
from esr.features import load_labels
from esr.io import read_json, read_jsonl, write_json
from esr.judge import restart_clusters


def response_token_offsets(tok, response: str) -> list[tuple[int, int]]:
    enc = tok(response, add_special_tokens=False, return_offsets_mapping=True)
    return enc["offset_mapping"]


def char_to_token(offsets: list[tuple[int, int]], char_pos: int) -> int:
    for i, (a, b) in enumerate(offsets):
        if b > char_pos:
            return i
    return len(offsets) - 1


def main() -> None:
    ap = base_parser("Detector-latent traces through episodes", default_judge="none")
    ap.add_argument("--results", default=str(DATA / "results" / "gemma-9b_*.jsonl"))
    ap.add_argument("--detectors", required=True)
    ap.add_argument("--ablate-set", default="top_by_cohen_d", choices=["detectors", "top_by_cohen_d"])
    ap.add_argument("--baselines-per-latent", type=int, default=3)
    ap.add_argument("--window", type=int, default=12, help="tokens on each side of a restart counted as 'correction'")
    args = ap.parse_args()

    spec = MODELS[args.model]
    labels = load_labels(spec)
    engine = make_engine(args, [])  # unit scale must already be cached
    det = read_json(Path(args.detectors))
    det_idx = torch.as_tensor([int(i) for i in det[args.ablate_set]], device=engine.device)

    rows = [r for f in glob.glob(args.results) for r in read_jsonl(Path(f)) if r.get("model") == args.model and r.get("response")]
    episodes = [r for r in rows if restart_clusters(r["response"])]
    ep_latents = {r["latent"] for r in episodes}
    baselines = []
    for lat in sorted(ep_latents):
        cands = [r for r in rows if r["latent"] == lat and not restart_clusters(r["response"]) and r["n_new_tokens"] >= 200]
        baselines += cands[: args.baselines_per_latent]
    print(f"{len(episodes)} episodes, {len(baselines)} steered baselines (same latents, no restart)")

    out_dir = DATA / "traces"
    out_dir.mkdir(exist_ok=True)
    plot_dir = DATA / "plots" / "traces"
    plot_dir.mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def trace(r: dict) -> dict:
        acts, resp_start = engine.sae_activations(r["prompt"], r["response"], meta_prompt=r.get("meta_prompt"),
                                                  latent=r["latent"], boost=r["boost"])
        resp_acts = acts[resp_start:]  # [T_resp, d_sae]
        det_sum = resp_acts[:, det_idx].sum(-1).float().cpu()
        det_active = (resp_acts[:, det_idx] > 0).float().sum(-1).cpu()
        offsets = response_token_offsets(engine.tok, r["response"])
        restarts = [char_to_token(offsets, c) for c in restart_clusters(r["response"])]
        T = det_sum.numel()
        return {"det_sum": det_sum.tolist(), "det_active": det_active.tolist(), "restarts": restarts, "T": T}

    summary = []
    for kind, group in (("episode", episodes), ("baseline", baselines)):
        for r in group:
            t = trace(r)
            det_sum = torch.tensor(t["det_sum"])
            T, restarts, w = t["T"], t["restarts"], args.window
            if restarts:
                first = restarts[0]
                pre = det_sum[: max(first - w, 1)].mean().item()
                corr = torch.cat([det_sum[max(0, s - w): min(T, s + w)] for s in restarts]).mean().item()
                post_start = min(T - 1, restarts[-1] + w)
                post = det_sum[post_start:].mean().item() if post_start < T - 1 else float("nan")
            else:
                pre = det_sum.mean().item()
                corr = post = float("nan")
            rec = {"kind": kind, "latent": r["latent"], "label": r.get("label", ""), "prompt": r["prompt"],
                   "boost": r["boost"], "seed": r["seed"], "condition": r.get("condition"), "T": T,
                   "restart_tokens": restarts, "mean_pre": pre, "mean_correction": corr, "mean_post": post,
                   "mean_all": det_sum.mean().item()}
            summary.append(rec)
            name = f"{kind}_lat{r['latent']}_seed{r['seed']}"
            write_json(out_dir / f"{name}.json", {**rec, **t})
            fig, ax = plt.subplots(figsize=(11, 3))
            ax.plot(range(T), t["det_sum"], lw=1)
            for s in restarts:
                ax.axvline(s, color="r", ls="--", lw=1)
            ax.set_title(f"{kind}: latent {r['latent']} ({r.get('label','')[:30]}) | {r['prompt'][:45]} | boost {r['boost']:.2f}", fontsize=9)
            ax.set_xlabel("response token")
            ax.set_ylabel("sum detector acts")
            fig.tight_layout()
            fig.savefig(plot_dir / f"{name}.png", dpi=120)
            plt.close(fig)
            print(f"{kind:<8} latent {r['latent']:>5} T={T:>3} restarts@{restarts} pre={pre:7.2f} corr={corr:7.2f} post={post:7.2f} all={rec['mean_all']:7.2f}")

    write_json(out_dir / "summary.json", summary)
    ep = [s for s in summary if s["kind"] == "episode"]
    bl = [s for s in summary if s["kind"] == "baseline"]
    if ep and bl:
        m = lambda xs: sum(xs) / len(xs)
        pre = m([s["mean_pre"] for s in ep])
        corr = m([s["mean_correction"] for s in ep])
        post = m([s["mean_post"] for s in ep if s["mean_post"] == s["mean_post"]] or [float("nan")])
        base = m([s["mean_all"] for s in bl])
        print("\nMean summed detector activation per token:")
        print(f"  episodes, off-topic region (before first restart): {pre:.2f}  ({pre / base:.2f}x baseline)")
        print(f"  episodes, around restarts (+/-{args.window} tokens):     {corr:.2f}  ({corr / base:.2f}x)")
        print(f"  episodes, after last restart:                        {post:.2f}  ({post / base:.2f}x)")
        print(f"  steered baselines without restart (whole response):  {base:.2f}")
        print(f"\ntraces -> {out_dir}, plots -> {plot_dir}")


if __name__ == "__main__":
    main()
