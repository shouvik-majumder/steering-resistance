"""Generate every figure used in the ESR summary deck.

Plain matplotlib, minimal styling, colour only where it carries information.

  python scripts/12_deck_figures.py            # all figures
  python scripts/12_deck_figures.py --only 5 6 # just those
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from esr.config import DATA  # noqa: E402
from esr.io import read_json, read_jsonl  # noqa: E402
from esr.judge import restart_clusters  # noqa: E402
from esr.metrics import summarize, wilson  # noqa: E402

OUT = DATA / "plots" / "deck"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 160, "savefig.dpi": 160, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 10, "axes.labelsize": 9, "legend.frameon": False,
    "figure.facecolor": "white", "savefig.facecolor": "white",
})
GREY, DARK, ACCENT = "0.65", "0.25", "#1f77b4"


def save(fig, name: str) -> None:
    fig.tight_layout()
    path = OUT / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {path.name}")


def load_rows(pattern: str) -> list[dict]:
    return [r for f in sorted(glob.glob(str(DATA / "results" / pattern))) for r in read_jsonl(Path(f))]


# ----------------------------------------------------------------- 1. method
def fig_method() -> None:
    fig, ax = plt.subplots(figsize=(10, 2.8))
    ax.axis("off")
    boxes = [
        (0.02, "Prompt\n\n'Explain how to\ncalculate probability.'"),
        (0.27, "Steer\n\nadd  b x W_dec[k]\nto the residual stream\nat every token"),
        (0.52, "Generate\n\n512 tokens,\nsteering stays on\nthe whole time"),
        (0.77, "Judge\n\nsplit into attempts at\nexplicit restarts,\nscore each 0-100"),
    ]
    for x, text in boxes:
        ax.add_patch(plt.Rectangle((x, 0.15), 0.21, 0.7, fill=False, ec=DARK, lw=1.0,
                                   transform=ax.transAxes))
        ax.text(x + 0.105, 0.5, text, ha="center", va="center", fontsize=8.5, transform=ax.transAxes)
    for x in (0.235, 0.485, 0.735):
        ax.annotate("", xy=(x + 0.03, 0.5), xytext=(x, 0.5), xycoords="axes fraction",
                    arrowprops=dict(arrowstyle="->", color=DARK, lw=1.0))
    ax.text(0.5, 0.02, "Outcome: does the model interrupt itself and get back on topic "
                       "while the steering is still on?", ha="center", fontsize=8.5,
            style="italic", transform=ax.transAxes)
    save(fig, "fig01_method.png")


# ------------------------------------------------------------- 2. boost sweep
def fig_boost_sweep() -> None:
    rows = load_rows("gemma-2b_boost_sweep.jsonl")
    if not rows:
        print("  (skipped: no boost sweep data yet)")
        return
    boosts = sorted({r["boost"] for r in rows})
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))

    ax = axes[0]
    means, sems = [], []
    for b in boosts:
        v = [r["first_score"] for r in rows if r["boost"] == b and r["first_score"] is not None]
        means.append(np.mean(v)); sems.append(np.std(v, ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0)
    ax.errorbar(boosts, means, yerr=sems, marker="o", color=DARK, capsize=3)
    ax.axhspan(20, 40, color=GREY, alpha=0.25, lw=0)
    ax.text(boosts[-1], 30, " calibration\n target", va="center", fontsize=8, color=DARK)
    ax.set_xlabel("steering strength (residual-norm units)")
    ax.set_ylabel("first-attempt relevance (0-100)")
    ax.set_title("Steering drives the answer off topic")

    ax = axes[1]
    for key, style, lab in (("distinct_word_frac", "-o", "distinct words"),
                            ("repeated_bigram_frac", "--s", "repeated word pairs")):
        v = [np.mean([r[key] for r in rows if r["boost"] == b]) for b in boosts]
        ax.plot(boosts, v, style, color=DARK if style.startswith("-o") else GREY, label=lab, ms=4)
    ax.set_xlabel("steering strength (residual-norm units)")
    ax.set_ylabel("fraction of output")
    ax.set_title("Too much steering collapses the text")
    ax.legend(fontsize=8)
    save(fig, "fig02_boost_sweep.png")


# ------------------------------------------------------------- 3. calibration
def fig_calibration() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    ax = axes[0]
    for i, (model, marker) in enumerate((("gemma-2b", "o"), ("gemma-9b", "s"))):
        th = read_json(DATA / "thresholds" / f"{model}.json", default={}) or {}
        vals = sorted(v["threshold"] for v in th.values())
        ax.plot(np.full(len(vals), i) + np.linspace(-0.12, 0.12, len(vals)), vals, marker,
                color=DARK, ms=4, mfc="none")
        ax.plot([i - 0.25, i + 0.25], [np.median(vals)] * 2, "-", color=ACCENT, lw=2)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Gemma-2-2B", "Gemma-2-9B"])
    ax.set_ylabel("calibrated steering strength")
    ax.set_title("Every latent needs its own strength\n(blue line = median)")

    ax = axes[1]
    th9 = read_json(DATA / "thresholds" / "gemma-9b.json", default={}) or {}
    key = next(iter(th9))
    hist = th9[key]["history"]
    ax.plot(range(1, len(hist) + 1), [h["x"] for h in hist], "-o", color=DARK, ms=4)
    ax.axhline(th9[key]["threshold"], color=ACCENT, ls="--", lw=1.5, label="final estimate")
    ax.set_xlabel("bisection step"); ax.set_ylabel("steering strength tried")
    ax.set_title(f"Probabilistic bisection\n(latent {key}, noisy judge scores)")
    ax.legend(fontsize=8)
    save(fig, "fig03_calibration.png")


# --------------------------------------------------- 4. first-attempt scores
def fig_first_scores() -> None:
    def firsts(rows):
        out = []
        for r in rows:
            att = (r.get("judge") or {}).get("attempts") or []
            if att and att[0].get("score") is not None:
                out.append(att[0]["score"])
        return np.array(out)

    unsteered = firsts(load_rows("gemma-2b_no_steer_seed0.jsonl"))
    steered = firsts(load_rows("gemma-2b_steered_seed0.jsonl"))
    steered9 = firsts(load_rows("gemma-9b_steered+meta_seed0.jsonl"))

    fig, ax = plt.subplots(figsize=(6, 3.4))
    bins = np.linspace(0, 100, 11)
    for data, colour, lab in ((unsteered, DARK, f"unsteered (2B, n={len(unsteered)})"),
                              (steered, ACCENT, f"steered (2B, n={len(steered)})"),
                              (steered9, GREY, f"steered (9B, n={len(steered9)})")):
        if len(data):
            ax.hist(data, bins=bins, density=True, histtype="step", lw=1.8, color=colour, label=lab)
    ax.set_xlabel("first-attempt relevance (0-100)")
    ax.set_ylabel("density")
    ax.set_title("Steering works, and scores are bimodal\n(mostly derailed or mostly fine, rarely in between)")
    ax.legend(fontsize=8)
    save(fig, "fig04_first_scores.png")


# --------------------------------------------------------------- 5. main rates
def fig_rates() -> None:
    conditions = [
        ("2B\nno steering", "gemma-2b_no_steer_seed0.jsonl"),
        ("2B\nsteered", "gemma-2b_steered_seed0.jsonl"),
        ("2B\n+ meta-prompt", "gemma-2b_steered+meta_seed0.jsonl"),
        ("9B\nsteered", "gemma-9b_steered_seed0.jsonl"),
        ("9B\n+ meta-prompt", "gemma-9b_steered+meta_seed0.jsonl"),
    ]
    names, stats = [], []
    for name, pattern in conditions:
        rows = load_rows(pattern)
        if rows:
            names.append(f"{name}\n(n={len(rows)})")
            stats.append(summarize(rows))

    keys = [("restart_language", "explicit restart in the text\n(regex, judge-free)"),
            ("multi_attempt", "multi-attempt\n(judge)"),
            ("esr", "ESR: restart and improved\n(judge)")]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8), sharey=True)
    x = np.arange(len(names))
    for ax, (key, title) in zip(axes, keys):
        vals = np.array([100 * s[key][0] for s in stats])
        lo = vals - np.array([100 * s[key][1] for s in stats])
        hi = np.array([100 * s[key][2] for s in stats]) - vals
        ax.errorbar(x, vals, yerr=[lo, hi], fmt="o", color=DARK, ecolor=GREY, capsize=4, ms=6, lw=1.4)
        for xi, v in zip(x, vals):
            ax.text(xi + 0.14, v, f"{v:.1f}%", va="center", fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels(names, fontsize=7.5)
        ax.set_xlim(-0.5, len(names) - 0.3)
        ax.set_title(title, fontsize=9)
        ax.axhline(0, color="0.85", lw=0.8, zorder=0)
    axes[0].set_ylabel("% of trials")
    axes[0].set_ylim(-1.5, 30)
    fig.suptitle("Self-correction appears only in the larger model, at a few percent "
                 "(whiskers: 95% confidence intervals; wide bars are small samples)", fontsize=10)
    save(fig, "fig05_rates.png")


# ------------------------------------------------------------- 6. detectors
def fig_detectors() -> None:
    det = read_json(DATA / "detectors" / "gemma-9b_seed0_response.json")
    if not det:
        print("  (skipped: no detector file)")
        return
    stats = det["all_stats"]
    idx = np.array([int(k) for k in stats])
    d = np.array([stats[k]["cohen_d"] for k in stats])
    p = np.array([stats[k]["p_welch"] for k in stats])
    on = np.array([stats[k]["on_mean"] for k in stats])
    off = np.array([stats[k]["off_mean"] for k in stats])
    top = set(int(i) for i in det["top_by_cohen_d"])
    sel = np.array([i in top for i in idx])

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    ax = axes[0]
    ax.scatter(d[~sel], -np.log10(np.clip(p[~sel], 1e-300, None)), s=4, color=GREY, alpha=0.5, lw=0)
    ax.scatter(d[sel], -np.log10(np.clip(p[sel], 1e-300, None)), s=14, color=ACCENT, lw=0,
               label=f"selected for ablation (n={sel.sum()})")
    ax.axvline(0, color=DARK, lw=0.8)
    ax.set_xlabel("Cohen's d  (off-topic minus on-topic)")
    ax.set_ylabel("-log10 p")
    ax.set_title("Contrastive search over 16,384 SAE latents")
    ax.legend(fontsize=8, loc="upper left")

    ax = axes[1]
    ax.scatter(on[~sel], off[~sel], s=4, color=GREY, alpha=0.5, lw=0)
    ax.scatter(on[sel], off[sel], s=14, color=ACCENT, lw=0)
    lim = max(on.max(), off.max()) * 1.05
    ax.plot([0, lim], [0, lim], "--", color=DARK, lw=0.8)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("mean activation, matched pairs")
    ax.set_ylabel("mean activation, mismatched pairs")
    ax.set_title("Latents that fire when answer and question\ndo not match (above the diagonal)")
    save(fig, "fig06_detectors.png")


# ---------------------------------------------------------- 7. episode trace
def fig_trace() -> None:
    files = sorted(glob.glob(str(DATA / "traces" / "episode_*.json")))
    if not files:
        print("  (skipped: no traces)")
        return
    traces = [read_json(Path(f)) for f in files]
    traces = [t for t in traces if t and t.get("restart_tokens")]
    traces.sort(key=lambda t: -len(t["restart_tokens"]))
    show = traces[:2]
    fig, axes = plt.subplots(len(show), 1, figsize=(9.5, 2.3 * len(show)), sharex=False)
    axes = np.atleast_1d(axes)
    for ax, t in zip(axes, show):
        y = np.array(t["det_sum"])
        ax.plot(np.arange(len(y)), y, lw=0.8, color=DARK)
        for s in t["restart_tokens"]:
            ax.axvline(s, color=ACCENT, ls="--", lw=1.2)
        ax.set_ylabel("detector\nactivation", fontsize=8)
        ax.set_title(f"latent {t['latent']} ({t['label'][:34]}) | {t['prompt'][:52]}", fontsize=8.5)
    axes[-1].set_xlabel("token position in the response")
    axes[0].plot([], [], "--", color=ACCENT, label="explicit restart")
    axes[0].legend(fontsize=8, loc="upper right")
    save(fig, "fig07_trace_example.png")


# --------------------------------------------------------- 8. trace summary
def fig_trace_summary() -> None:
    summary = read_json(DATA / "traces" / "summary.json")
    if not summary:
        print("  (skipped: no trace summary)")
        return
    ep = [s for s in summary if s["kind"] == "episode"]
    bl = [s for s in summary if s["kind"] == "baseline"]
    un = [s for s in summary if s["kind"] == "unsteered"]
    m = lambda xs: float(np.mean(xs)) if len(xs) else np.nan
    e = lambda xs: (float(np.std(xs, ddof=1) / np.sqrt(len(xs))) if len(xs) > 1 else 0.0)
    groups = [
        ("unsteered\non-topic", [s["mean_all"] for s in un]),
        ("steered, restarted:\nbefore the restart", [s["mean_pre"] for s in ep]),
        ("steered, restarted:\naround the restart", [s["mean_correction"] for s in ep]),
        ("steered, restarted:\nafter the restart", [s["mean_post"] for s in ep if s["mean_post"] == s["mean_post"]]),
        ("steered,\nnever restarted", [s["mean_all"] for s in bl]),
    ]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    x = np.arange(len(groups))
    vals = [m(v) for _, v in groups]
    errs = [e(v) for _, v in groups]
    colours = [GREY] + [ACCENT] * 3 + [DARK]
    ax.bar(x, vals, yerr=errs, color=colours, edgecolor=DARK, capsize=3, width=0.62)
    for xi, v, (lab, arr) in zip(x, vals, groups):
        ax.text(xi, v + errs[int(xi)] + 1.2, f"{v:.0f}\n(n={len(arr)})", ha="center", fontsize=7.5)
    ax.set_xticks(x); ax.set_xticklabels([g[0] for g in groups], fontsize=7.5)
    ax.set_ylabel("detector-latent activation\nper token")
    ax.set_title("Detector latents are elevated in ALL steered text,\nnot only where the model restarts", fontsize=10)
    save(fig, "fig08_trace_summary.png")


# ------------------------------------------------------------- 9. prefill
def fig_prefill() -> None:
    rows = load_rows("gemma-9b_prefill1000_seed0.jsonl")
    if not rows:
        print("  (skipped: no prefill data)")
        return
    by: dict[int, dict] = {}
    for r in rows:
        by.setdefault(r["prefix_id"], {})[r["condition"].replace("prefill_", "")] = r
    ids = [i for i, d in by.items() if all(c in d and d[c]["cont_score"] is not None
                                           for c in ("none", "ablate", "random"))]
    cs = {c: np.array([by[i][c]["cont_score"] for i in ids]) for c in ("none", "ablate", "random")}
    src = np.array([by[i]["none"]["source"]["source_first_score"] for i in ids])

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
    ax = axes[0]
    labels = ["off-topic\nprefix", "continuation\nno ablation",
              "continuation\ndetectors\nablated", "continuation\nrandom\nablated"]
    vals = [src.mean(), cs["none"].mean(), cs["ablate"].mean(), cs["random"].mean()]
    errs = [np.std(v, ddof=1) / np.sqrt(len(v)) for v in (src, cs["none"], cs["ablate"], cs["random"])]
    ax.bar(np.arange(4), vals, yerr=errs, color=[GREY, DARK, ACCENT, GREY], edgecolor=DARK,
           capsize=3, width=0.62)
    for xi, v in enumerate(vals):
        ax.text(xi, v + errs[xi] + 1.5, f"{v:.1f}", ha="center", fontsize=8)
    ax.set_xticks(np.arange(4)); ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel("relevance to the question (0-100)")
    ax.set_title(f"The model recovers silently (n={len(ids)} prefixes)", fontsize=10)

    ax = axes[1]
    for xi, (c, lab) in enumerate((("ablate", "detectors\nablated"), ("random", "random latents\nablated"))):
        d = cs[c] - cs["none"]
        mu = d.mean(); se = np.std(d, ddof=1) / np.sqrt(len(d))
        ax.errorbar([xi], [mu], yerr=[1.96 * se], fmt="o", color=DARK, capsize=4, ms=6)
        ax.text(xi + 0.12, mu, f"{mu:+.1f}", va="center", fontsize=9)
    ax.axhline(0, color=DARK, lw=0.8)
    ax.set_xlim(-0.5, 1.5); ax.set_xticks([0, 1])
    ax.set_xticklabels(["detectors\nablated", "random latents\nablated"], fontsize=8)
    ax.set_ylabel("change in relevance vs no ablation")
    ax.set_title("Ablating the detectors changes nothing\n(95% CI, paired by prefix)", fontsize=10)
    save(fig, "fig09_prefill.png")


# -------------------------------------------------------------- 10. judging
def fig_judge() -> None:
    rows = load_rows("gemma-9b_steered+meta_seed0.jsonl")
    if not rows:
        print("  (skipped)")
        return
    raw = sum(1 for r in rows if (r.get("judge") or {}).get("n_attempts_raw", 0) and r["judge"]["n_attempts_raw"] >= 2)
    gated = sum(1 for r in rows if len((r.get("judge") or {}).get("attempts") or []) >= 2)
    regex = sum(1 for r in rows if r.get("response") and restart_clusters(r["response"]))
    n = len(rows)

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    labels = ["judge, taken\nat face value", "judge, gated on text\nactually containing\na restart",
              "regex on the text\n(no judge)"]
    vals = [100 * raw / n, 100 * gated / n, 100 * regex / n]
    ax.bar(np.arange(3), vals, color=[GREY, ACCENT, DARK], edgecolor=DARK, width=0.6)
    for xi, (v, k) in enumerate(zip(vals, [raw, gated, regex])):
        ax.text(xi, v + 0.15, f"{v:.1f}%\n({k}/{n})", ha="center", fontsize=8)
    ax.set_xticks(np.arange(3)); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("% of trials called 'multi-attempt'")
    ax.set_title("A free local judge invents self-corrections\n(Gemma-2-9B, meta-prompted)", fontsize=10)
    save(fig, "fig10_judge.png")


FIGURES = {1: fig_method, 2: fig_boost_sweep, 3: fig_calibration, 4: fig_first_scores,
           5: fig_rates, 6: fig_detectors, 7: fig_trace, 8: fig_trace_summary,
           9: fig_prefill, 10: fig_judge}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=int, nargs="*", default=None)
    args = ap.parse_args()
    for k in sorted(FIGURES):
        if args.only and k not in args.only:
            continue
        print(f"figure {k}: {FIGURES[k].__name__}")
        FIGURES[k]()
    print(f"\nfigures in {OUT}")
