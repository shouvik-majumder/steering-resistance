"""Build the ESR summary deck (data-club style: goal, method, finding).

  python scripts/13_make_deck.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from deckbuild import Deck  # noqa: E402

FIG = ROOT / "data" / "plots" / "deck"
OUT = ROOT / "data" / "deck"


def build() -> Deck:
    d = Deck()

    d.title_slide(
        "Do language models resist being steered off topic?",
        ["Replication of McKenzie et al., Endogenous Resistance to Activation Steering, ICML 2026",
         "Gemma-2-2B and Gemma-2-9B, one RTX 3090, all open-weight and local",
         "Shouvik Majumder  |  data club, September 2026"],
    )

    d.slide(
        "The question",
        bullets=[
            "Activation steering adds a fixed direction to the residual stream, pushing the model toward an unrelated concept while it answers.",
            "Sometimes the model notices mid-answer, says so out loud, and returns to the question while the push is still on.",
            "The authors call this endogenous steering resistance: 3.8% of trials in Llama-3.3-70B, under 1% in everything smaller.",
            "They identify 26 SAE latents by contrastive search and report that ablating them cuts the effect by a quarter.",
            "Two questions here: does it replicate on models that fit one 24 GB card, and are those latents causally responsible?",
        ],
        bullet_size=16,
        notes="Frame it as a mechanism question, not a benchmark question. The interesting claim is the causal one.",
    )

    d.slide("Method: steer every token, then judge the answer", image=FIG / "fig01_method.png",
            notes="Steering stays on for the whole generation, so any recovery happens against a live perturbation.")

    d.slide("Steering has a narrow usable window", image=FIG / "fig02_boost_sweep.png",
            bullets=["Relevance falls from 73 to 9 as strength rises, while repeated word pairs go from 3% to 40%.",
                     "Self-correction can only happen in between: off topic but still coherent.",
                     "The target band is crossed near 0.7, which is where the per-latent calibration lands."],
            notes="Gemma-2-2B, 4 latents x 6 strengths x 3 trials. This is the practical reason "
                  "the experiment needs per-latent calibration rather than one shared setting.")

    d.slide("Every latent needs its own steering strength", image=FIG / "fig03_calibration.png",
            bullets=["Calibrated per latent by probabilistic bisection to a first-attempt score of about 30/100.",
                     "A fixed strength that derails one latent leaves another untouched, so shared settings do not transfer."])

    d.slide("Steering works, and relevance scores are bimodal", image=FIG / "fig04_first_scores.png",
            notes="Bimodality is why single-sample calibration is noisy: answers are usually either fine or wrecked.")

    d.slide("Main result: self-correction only in the larger model", image=FIG / "fig05_rates.png",
            bullets=["Unsteered control: zero restarts, mean relevance 98. The behaviour is caused by steering.",
                     "Gemma-2-9B restarts in a few percent of trials; Gemma-2-2B never did in 60 trials.",
                     "Both are inside the paper's reported range for these two models."])

    d.slide(
        "What a self-correction actually looks like",
        bullets=[
            "Prompt: 'How do you make a perfect omelette?'  Steered toward a 'numbers and ranges' latent.",
            "'About 250 grams of butter ... about 1000 people. This is about the best way to eat it!'",
            "'Seriously, this is ridiculous. Let's get you a recipe that will work!'",
            "Then: a briefly better attempt, a relapse, and 'Let me try again:' before drifting off once more.",
            "The model notices reliably. It almost never succeeds in fixing the answer while the push is still on.",
        ],
        bullet_size=15,
        notes="Read the quotes aloud. This slide is what makes the phenomenon concrete for the audience.",
    )

    d.slide("Finding candidate off-topic detector latents", image=FIG / "fig06_detectors.png",
            bullets=["Unsteered answers are paired with the wrong questions; latents that fire only on mismatches are the candidates.",
                     "No latent met the paper's strict rule, so the 26 largest effect sizes were used for ablation."])

    d.slide("Detector activity through a self-correction", image=FIG / "fig07_trace_example.png",
            bullets=["Dashed lines mark explicit restarts. Activity rises through off-topic stretches and falls when the answer is on topic."])

    d.slide("But the same latents are just as active when the model never restarts",
            image=FIG / "fig08_trace_summary.png",
            bullets=["Elevated 6x over unsteered text, reproducing the paper's 4.4x.",
                     "The control the paper does not report: steered text with no restart at all is elevated just as much.",
                     "So these latents track off-topic content, not the decision to correct."])

    d.slide("Causal test with enough power: prefill, then ablate", image=FIG / "fig09_prefill.png",
            bullets=["Restarts are too rare to test directly, so feed the model its own off-topic text and let it continue unsteered.",
                     "It recovers in about two thirds of cases, almost always silently: 0 of 600 continuations contain a restart phrase.",
                     "Ablating the 26 detector latents changes recovery by -1.5 points (95% CI -5.0 to +2.0), the same as ablating random latents."])

    d.slide("Methods caution: a free local judge invents self-corrections", image=FIG / "fig10_judge.png",
            bullets=["Taken at face value the judge reports several times more restarts than the text contains.",
                     "Fix: accept an attempt boundary only where a restart phrase actually occurs in the model's own words.",
                     "Same lesson as any probe: a number needs a baseline that shares its access to the data."])

    d.slide(
        "Summary",
        bullets=[
            "The pipeline replicates every qualitative claim testable at this scale: no self-correction without steering, a few percent with it in the 9B model, and more insistent but no more successful correction under meta-prompting.",
            "The correlational mechanistic claim replicates: candidate detector latents are strongly elevated in off-topic text.",
            "The causal claim does not transfer. Ablation is indistinguishable from a random-latent control, in a test powered to detect a five-point effect.",
            "Main caveats: one layer, a 16k SAE explaining 61% of variance, and a local judge rather than the paper's Claude.",
            "Next: the same geometry question asked where ground truth exists, which is the second project.",
        ],
        bullet_size=15,
    )
    return d


if __name__ == "__main__":
    deck = build()
    path = deck.save(OUT / "ESR_replication.pptx")
    previews = deck.preview(OUT / "preview")
    print(f"deck  -> {path}")
    print(f"slides: {len(deck.spec)}   previews -> {previews[0].parent}")
