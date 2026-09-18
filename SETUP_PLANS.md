# Setup plans: projects 2 and 3, plus a learning-ranked backlog

Companion to [NEXT_PROJECTS.md](NEXT_PROJECTS.md). Project 1 (belief-state geometry) is built and
running in `D:\dev\belief-geometry`. This file is the concrete setup for the next two, and a
backlog ordered by **how much you learn per hour**, which you said matters most.

Everything below is free. No paid API is required anywhere; where a paper used one, the
substitute is written out.

Disk budget if you do all of it: about 40 GB of new model weights on top of what you have.
You have 3.7 TB free, so disk is not a constraint. VRAM is: 24 GB is the real limit.

---

## Project 2: circular / multi-dimensional features

Reproduce the finding that days of the week and months of the year live on **circles** inside a
language model, and that the model uses those circles to do modular arithmetic.

Paper: [arXiv:2405.14860](https://arxiv.org/abs/2405.14860) ·
Code: [JoshEngels/MultiDimensionalFeatures](https://github.com/JoshEngels/MultiDimensionalFeatures)

### Environment

```powershell
conda create -n circfeat python=3.11 -y
conda activate circfeat
pip install torch --index-url https://download.pytorch.org/whl/cu128
git clone https://github.com/JoshEngels/MultiDimensionalFeatures D:\dev\circular-features
cd D:\dev\circular-features
pip install -r requirements.txt          # pinned for CUDA 12.1; see the fallback below
```

Python 3.11 rather than 3.12 because the pinned dependencies predate 3.12 wheels for some
packages. If `requirements.txt` fights you (likely: it pins older `transformer_lens` and
`sae_lens` against CUDA 12.1), do not fight back. Install current versions instead and fix the
few call sites that moved:

```powershell
pip install transformer_lens sae_lens transformers datasets adjustText circuitsvis ipython matplotlib scikit-learn
```

The analysis is simple enough that a broken pin is never worth an afternoon.

### Downloads

| What | Size | Gated? |
|---|---|---|
| GPT-2 small | ~0.5 GB | no |
| GPT-2 SAEs (`gpt2-small-res-jb` via SAELens) | ~2 GB | no |
| Mistral-7B-v0.1 | ~14 GB | no (Apache 2.0) |
| Mistral SAEs | ~3 GB | no |
| Llama-3-8B | ~16 GB | yes, free acceptance |

**Start with GPT-2 only.** The circles for days and months are already visible there, it costs
30 minutes and 0.5 GB, and you learn the whole method before spending 14 GB. Skip Llama-3
entirely unless you want the third replication.

### First milestones

1. `python reducibility_demo.py` — the toy demonstration of what "irreducibly multi-dimensional"
   means. Read the code before the paper; it is short and it defines the central concept.
2. `python clustering.py` then the GPT-2 script — cluster SAE features by co-occurrence and find
   the day-of-week circle. Your output should be a ring of seven points labelled Monday to Sunday.
3. The intervention experiments (`circle_probe_interventions.py`) — this is the part closest to
   your ESR work: patch the circular subspace and watch the arithmetic break.

### Your extension (the reason to do this project)

You already have Gemma-2-2B/9B and Gemma Scope locally, and a working hook/SAE stack.

- **Do the circles exist in Gemma-2?** Nobody has published this. Same method, model you already
  have, no new downloads.
- **Is the circle an attractor?** Push the residual stream *off* the circle with your ESR steering
  hook, then measure whether later tokens fall back onto it. That is your ESR question asked where
  the ground-truth geometry is known, and it connects directly to project 3.
- **Does ablating the circle's SAE latents destroy the arithmetic?** You have written exactly this
  ablation code once already.

---

## Project 3: manifold steering

Steer *along a fitted manifold* instead of along a straight line, and show that the behaviour
moves smoothly instead of degrading into gibberish.

Paper: [arXiv:2605.05115](https://arxiv.org/abs/2605.05115) ·
Code: [goodfire-ai/causalab, branch `manifold_steering`](https://github.com/goodfire-ai/causalab/tree/manifold_steering)

### Environment

```powershell
conda create -n manifoldsteer python=3.12 -y
conda activate manifoldsteer
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install transformers accelerate scikit-learn scipy matplotlib pandas tqdm einops huggingface_hub python-dotenv
git clone -b manifold_steering https://github.com/goodfire-ai/causalab D:\dev\manifold-steering
```

### Downloads

The paper uses Llama-3.1-8B (gated, free acceptance, ~16 GB, fits in 24 GB at bf16).
**Recommended instead: use Gemma-2-9B, which is already on your disk.** Running the method on a
model the authors did not use is a contribution rather than a copy, and it costs nothing.

### Method, in plain terms (you can implement this yourself in an afternoon)

1. Pick a concept family with known structure, e.g. the seven weekdays.
2. Collect residual-stream activations for prompts about each value, average them into a
   **centroid** per value.
3. PCA the centroids down to about 64 dimensions, then fit a smooth curve (cubic spline) through
   them. That curve is the **activation manifold**.
4. Fit a second curve through the *output distributions* in Hellinger space (the map
   `p → sqrt(p)`, which turns probability distributions into points on a sphere where Euclidean
   distance is meaningful). That is the **behaviour manifold**.
5. Steer by moving along the activation curve, and check that the output moves correspondingly
   along the behaviour curve. Compare against straight-line steering of the same length.

The only genuinely new machinery versus your ESR code is spline fitting and the Hellinger map.
Everything else is the hook you already wrote.

### Your extension

The paper's own limitations section asks for two things you are positioned to supply:

- **Abstract concepts.** They only did weekdays, months, letters, ages. Your ESR corpus is an
  abstract-concept dataset: on-topic versus off-topic drift. Does the "off-topic" direction you
  steered along have a manifold structure, and does steering along it degrade output less?
- **Intermediate variables.** They manipulate output behaviour directly; manipulating a mediating
  internal quantity is listed as future work.

---

## Backlog, ranked by learning value

Ordered by how much of the field you absorb per hour, not by how impressive the result would be.
Feasibility on your hardware is noted for each.

### Do these alongside project 1

**1. ARENA Chapter 1: Transformer Interpretability** — *the single highest-value item on this page.*
[learn.arena.education](https://learn.arena.education/chapter1_transformer_interp/), free, self-paced,
exercises with solutions. Covers building a transformer from scratch, TransformerLens hooks,
linear probes, activation steering, and SAEs — every tool you have been using, taught properly
and in order. You have been learning these backwards, from a research replication. A few evenings
here will fill in the gaps and make everything else faster.
*Feasibility: trivial, mostly CPU or small GPU.*

**2. Toy Models of Superposition** — the founding document of "geometry of representations".
Paper: [transformer-circuits.pub/2022/toy_model](https://transformer-circuits.pub/2022/toy_model/index.html) ·
official notebooks [anthropics/toy-models-of-superposition](https://github.com/anthropics/toy-models-of-superposition) ·
also an ARENA exercise set, and clean replications such as
[zroe1/toy-models-of-superposition](https://github.com/zroe1/toy-models-of-superposition).
A two-layer autoencoder with five features in two dimensions, and you watch the features arrange
themselves into **digons, triangles, pentagons and tetrahedra**. This is where the vocabulary of
superposition, feature geometry and polytopes comes from, and it runs in seconds on a CPU.
*Feasibility: trivial. Do it in an evening.*

**3. Grokking modular addition** — reverse-engineer a complete algorithm from weights.
Paper: [arXiv:2301.05217](https://arxiv.org/abs/2301.05217) · Neel Nanda's
[replication walkthrough](https://www.neelnanda.io/mechanistic-interpretability/modular-addition-walkthrough)
with video and code. A one-layer transformer learns modular addition by embedding numbers on a
**circle** and composing rotations, discovered via Fourier analysis of the weights. Teaches you
to read structure out of weight matrices, not just activations.
Geometry follow-up worth reading after: *On the geometry and topology of representations: the
manifolds of modular addition* ([arXiv:2512.25060](https://arxiv.org/abs/2512.25060)).
*Feasibility: easy, minutes of training on your card.*

### Strong next projects

**4. Othello-GPT world models.** A transformer trained only on legal move sequences builds an
internal board state, found with linear probes and confirmed by intervention. The canonical
"emergent world representation" result, and the cleanest example of probe-then-intervene
methodology. *Feasibility: easy, small model.*

**5. Intrinsic dimension profiles across layers.** Measure how many dimensions the representation
really uses at each layer; the profile dips where the most semantic representation lives.
Paper [arXiv:2302.00294](https://arxiv.org/abs/2302.00294), tooling **DADApy** or
`scikit-dimension`. Cheapest possible reuse of your existing Gemma stack: a few hours, one
afternoon, and it gives you a geometry measurement you will use everywhere.
*Feasibility: very easy. Good "first geometry measurement" on real LLMs.*

**6. Categorical and hierarchical concepts as simplices and polytopes.**
[arXiv:2406.01506](https://arxiv.org/abs/2406.01506), code
[KihoPark/LLM_Categorical_Hierarchical_Representations](https://github.com/KihoPark/LLM_Categorical_Hierarchical_Representations),
runs on **Gemma-2B** which you already have. Teaches the causal inner product, which is the
technically deepest idea in the "concepts as directions" literature.
*Feasibility: easy to medium. No new downloads.*

**7. SAE feature geometry: crystals, lobes, galaxies.**
[arXiv:2410.19750](https://arxiv.org/abs/2410.19750), code
[ejmichaud/feature-geometry](https://github.com/ejmichaud/feature-geometry). Uses the Gemma Scope
SAEs on your disk; much of it is analysis of decoder weight matrices rather than generation, so it
is cheap. *Feasibility: easy. No new downloads.*

### Specialised, higher ceiling

**8. Manifold capacity applied to LLM layers.** Code
[schung039/neural_manifolds_replicaMFT](https://github.com/schung039/neural_manifolds_replicaMFT).
Your own field's method, barely applied to modern LLM residual streams. The learning curve is the
replica theory, which is real work, but you are closer to it than almost anyone else doing
interpretability. *Feasibility: medium. Highest originality-per-effort for you specifically.*

**9. Topological data analysis of representations.** `giotto-tda`, curated list
[AwesomeTDA4NLP](https://github.com/AdaUchendu/AwesomeTDA4NLP). Persistent homology detects a
circle or torus without assuming one. Least crowded area on this page. *Feasibility: medium.*

**10. Refusal direction, and whether it is really one direction.**
[andyrdt/refusal_direction](https://github.com/andyrdt/refusal_direction) plus the 2026 counterpoint
[arXiv:2602.02132](https://arxiv.org/abs/2602.02132). Methodologically the closest relative of your
ESR work. *Feasibility: medium; needs a chat model in 24 GB, so Gemma-2-9B or Qwen2.5-7B, both of
which you have.*

**11. Do SAEs capture concept manifolds?** [goodfire-ai/sae-manifold](https://github.com/goodfire-ai/sae-manifold).
A critique of the SAE-as-directions view, with synthetic benchmarks that run without a large
model. Speaks directly to the weakness you hit in the ESR project, where detector latents indexed
off-topicness without being causally load-bearing. *Feasibility: medium.*

### Suggested rhythm

Work through ARENA Chapter 1 in the evenings while running project 1 experiments during the day.
Add Toy Models of Superposition and the grokking replication as single-evening exercises. By the
time those are done, you will have the vocabulary to choose between projects 2, 3 and 8 on your
own judgement rather than on a recommendation.
