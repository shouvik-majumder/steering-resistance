# Where to go next: geometry of representations in transformers

A shortlist of replicable projects on the **geometry and topology of internal representations**,
chosen for: public paper + public code, runnable on one RTX 3090 with free/local models, and
genuine room to extend into your own experiment. Compiled 2026-09-18.

Context: you already have a working stack (HF hooks, Gemma-2 2B/9B, Gemma Scope SAEs, local
judge, steering + ablation, activation capture) from the ESR replication. Projects that reuse it
are marked **[reuses ESR stack]**.

---

## 0. The vocabulary you will keep meeting

| Term | Meaning |
|---|---|
| **Representation / population geometry** | Treat the activations of N units as a point in N-dim space; a set of related inputs traces out a cloud or surface. Study its shape instead of individual units. Same idea as neural population geometry in systems neuroscience. |
| **Manifold** | A curved low-dimensional surface embedded in the high-dim activation space (a circle of weekdays, a ring of head directions, a swiss roll of "temperature"). |
| **Linear representation hypothesis (LRH)** | The claim that concepts are *directions*: `activation ≈ Σ aᵢ dᵢ`. Most steering and SAE work assumes it. The geometry literature is largely about where it breaks. |
| **Superposition** | More features than dimensions, stored as non-orthogonal directions. Explains why features interfere and why SAEs are used to pull them apart. |
| **Intrinsic dimension (ID)** | How many coordinates a data cloud really needs (TwoNN, MLE estimators), vs the ambient dimension (2304 or 3584 for your Gemmas). ID profiles across layers are a standard "geometry of a network" measurement. |
| **Manifold capacity** | From Chung/Sompolinsky: how many object manifolds a linear readout can separate, as a function of their radius, dimension and correlations. A single number summarising "how linearly usable is this geometry". |
| **Geodesic / pullback metric** | Distance *along* a curved manifold rather than through the ambient space; the pullback metric is the ambient metric restricted to the manifold. (Your `pullback` conda env suggests you have met these.) |
| **Persistent homology / TDA** | Counts holes, loops and voids in a point cloud across scales (a "barcode"). Detects circular or toroidal structure without assuming a parametric shape. |
| **Simplex / polytope representation** | A categorical variable with k values represented as k points forming a simplex; hierarchies become orthogonal directions. |
| **Probe** | A small supervised model (usually linear) trained to read a variable out of activations. Geometry version: fit a *curve or surface* instead of a hyperplane. |
| **Causal intervention** | Ablate / patch / steer, then measure behaviour. The difference between "the geometry is there" and "the model uses it". You did this in the ESR project. |
| **RSA / CKA** | Representational similarity: compare two systems by their internal distance matrices rather than unit-by-unit. Standard bridge between brains and models. |

Venues to follow: **NeurReps** (neurreps.org, symmetry and geometry in neural representations)
and **UniReps** (unireps.org, unifying representations across ML/neuro/cogsci). Both publish
short, reproducible workshop papers, which is a realistic target for a first contribution.

---

## Tier 1: the three I would actually set up

### 1. Belief-state geometry — fractals in the residual stream **(best first project)**

- **Paper:** Shai et al., *Transformers Represent Belief State Geometry in their Residual Stream*,
  NeurIPS 2024 — [arXiv:2405.15943](https://arxiv.org/abs/2405.15943)
- **Follow-up with open threads:** *Constrained Belief Updates Explain Geometric Structures in
  Transformer Representations* — [arXiv:2502.01954](https://arxiv.org/abs/2502.01954)
- **Code:** [adamimos/epsilon-transformers](https://github.com/adamimos/epsilon-transformers)
  (original), [danibalcells/belief-state-transformers](https://github.com/danibalcells/belief-state-transformers)
  (clean re-implementation, easier to read), [ilatims-b/extending-constrained-belief](https://github.com/ilatims-b/extending-constrained-belief)
- **What happens:** train a *tiny* transformer (4 layers, 64-dim) from scratch on strings emitted
  by a 3-state hidden Markov model ("Mess3"). Theory says the optimal predictor must track a
  Bayesian posterior over hidden states, i.e. a point in a 2-simplex, and that the set of
  reachable posteriors is a **fractal**. You then find a 2D linear projection of the residual
  stream and the fractal is simply *there*.
- **Why first:** minutes per training run on your 3090, no gated weights, no LLM judge, no API.
  Crucially the ground truth geometry is computable analytically, so you can tell whether your
  analysis is right — the opposite of the ESR situation where every measurement needed a judge.
  It teaches residual stream, probing, and representation geometry in a setting that cannot lie
  to you.
- **Your extension space:** when during training does the fractal appear (geometry formation
  dynamics)? What happens to it under ablation of specific heads? Does steering *within* the
  simplex change predictions the way theory says? Non-ergodic or higher-dimensional processes?
  This is an unusually open area because almost everyone stops at "the fractal is there".

### 2. Non-linear (circular / multi-dimensional) features **[reuses ESR stack]**

- **Paper:** Engels et al., *Not All Language Model Features Are One-Dimensionally Linear*,
  ICLR 2025 — [arXiv:2405.14860](https://arxiv.org/abs/2405.14860)
- **Code:** [JoshEngels/MultiDimensionalFeatures](https://github.com/JoshEngels/MultiDimensionalFeatures)
- **What happens:** using SAEs, they find features that are irreducibly *multi*-dimensional:
  days of the week and months of the year live on **circles** in GPT-2 / Mistral-7B, and the model
  uses those circles to do modular arithmetic ("two days after Monday"). They confirm with
  intervention experiments — exactly your ablation/causality interest, but on a manifold.
- **Hardware:** GPT-2 parts are trivial; the Mistral/Llama parts need ~60 GB of disk for weights
  and artifacts (you have 3.7 TB) and fit in bf16 on 24 GB.
- **Why it fits:** same libraries you already use (`transformer_lens`, `sae_lens`), and it is the
  cleanest demonstration that "concept = direction" is incomplete.
- **Your extension space:** do the circles exist in **Gemma-2** (which you already have locally,
  with Gemma Scope SAEs)? Is the circle an **attractor** — if you steer *off* the circle, does the
  state relax back onto it? (That is your ESR question in a setting with known ground-truth
  geometry.) Do the circular features survive the ablation of the SAE latents that define them?

### 3. Manifold steering — geometry-aware intervention **[reuses ESR stack]**

- **Paper:** *Manifold Steering Reveals the Shared Geometry of Neural Network Representation and
  Behavior*, 2026 — [arXiv:2605.05115](https://arxiv.org/abs/2605.05115)
- **Code:** [goodfire-ai/causalab, branch `manifold_steering`](https://github.com/goodfire-ai/causalab/tree/manifold_steering)
- **What happens:** fit an *activation manifold* (PCA to 64 dims, then splines through concept
  centroids) and a *behaviour manifold* (output distributions mapped to Hellinger space, splines
  again). Steering **along** the fitted activation manifold produces behaviour that moves
  smoothly along the behaviour manifold; straight-line steering produces unnatural outputs. They
  report geodesic-distance correlations and an "intrinsic R²" of the pullback.
- **Why it fits:** it is the direct upgrade of what you did in the ESR project. You steered along
  a single SAE decoder direction and had to calibrate a scalar boost; this reframes steering as
  "find the right geometry" and explains *why* large boosts give gibberish (you leave the manifold).
- **Your extension space:** the paper's own limitations section asks for exactly two things you
  are positioned to do: (a) apply it to **abstract** concepts such as refusal or off-topic drift
  rather than weekdays, and (b) manipulate **intermediate** variables rather than outputs. Your
  ESR corpus is an abstract-concept dataset already.

---

## Tier 2: strong, more specialised

### 4. SAE feature geometry: crystals, lobes, galaxies **[reuses ESR stack]**
Li, Michaud, Baek, Engels, Sun, Tegmark — [arXiv:2410.19750](https://arxiv.org/abs/2410.19750),
code [ejmichaud/feature-geometry](https://github.com/ejmichaud/feature-geometry).
Three scales of structure in SAE decoder space: "atoms" (parallelogram crystals generalising
king−man+woman), "brain" (spatially modular lobes: a math/code lobe), "galaxy" (power-law
eigenvalue spectrum, i.e. a fractal-ish global shape). Uses **Gemma-2 SAEs, which you already
have downloaded**. Cheap: much of it is analysis of decoder weight matrices, not generation.
*Extension:* are your 26 "off-topic detector" latents a lobe? Do the ESR steering latents that
derail the model at low boost sit in a particular region?

### 5. Concepts as simplices and polytopes
Park, Choe, Jiang, Veitch, *The Geometry of Categorical and Hierarchical Concepts in LLMs*,
ICLR 2025 oral — [arXiv:2406.01506](https://arxiv.org/abs/2406.01506), code
[KihoPark/LLM_Categorical_Hierarchical_Representations](https://github.com/KihoPark/LLM_Categorical_Hierarchical_Representations).
900+ WordNet concepts in **Gemma-2B** and Llama-3-8B: categorical concepts form simplices,
hierarchically related concepts are orthogonal, complex concepts are direct sums (polytopes).
Beautiful, very explicit geometry, modest compute, and the causal-inner-product machinery is
worth learning.

### 6. Do SAEs capture concept manifolds? (a critique you can join)
2026 — [arXiv:2604.28119](https://arxiv.org/abs/2604.28119), code
[goodfire-ai/sae-manifold](https://github.com/goodfire-ai/sae-manifold).
Finds that SAEs **dilute** manifolds across many partially redundant "tiling" features rather
than capturing them compactly, using synthetic ground-truth manifolds plus Llama-3.1-8B. Includes
synthetic benchmarks runnable **without** a big model. A good methods-critique project, and it
speaks directly to the weakness you already hit (your detector latents indexed off-topicness but
were not causally load-bearing).
Related: *Probing for Representation Manifolds in Superposition*
([arXiv:2605.18537](https://arxiv.org/abs/2605.18537)) generalises linear probes to manifold
probes and steers along a learned time manifold in Llama-2-7b.

### 7. Refusal as a direction — and the counter-argument **[reuses ESR stack]**
Arditi et al., NeurIPS 2024 — [arXiv:2406.11717](https://arxiv.org/abs/2406.11717), code
[andyrdt/refusal_direction](https://github.com/andyrdt/refusal_direction), plus an independent
from-scratch reproduction [dallen2021/refusal-direction-repro](https://github.com/dallen2021/refusal-direction-repro).
Counterpoint: *There Is More to Refusal in LLMs than a Single Direction*
([arXiv:2602.02132](https://arxiv.org/abs/2602.02132)).
The canonical "one direction, causally verified by ablation" result, methodologically the closest
relative of your ESR work. The geometry question is live and unsettled: is refusal a direction, a
cone, or a manifold?

### 8. Manifold capacity applied to LLM layers *(your home turf, under-explored)*
Chung, Lee, Sompolinsky theory; code [schung039/neural_manifolds_replicaMFT](https://github.com/schung039/neural_manifolds_replicaMFT)
(the `mftma` package computes capacity, radius, dimension, centre correlation from arrays of
shape (features, samples) per manifold) and [sompolinsky-lab/dnn-object-manifolds](https://github.com/sompolinsky-lab/dnn-object-manifolds).
Overview: *Neural population geometry* (Chung & Abbott, Curr Opin Neurobiol 2021).
Very little of this has been done on modern LLM residual streams with SAE-defined concept
manifolds. Concretely: define a manifold per concept (all activations of prompts about "probability"),
run MFTMA layer by layer, and watch capacity/radius/dimension evolve — then check what steering
does to those quantities. This is the most natural bridge from your existing expertise, and the
most open.

### 9. Intrinsic dimension profiles across layers
Valeriani, Doimo, Cuturello, Laio, Ansuini, Cazzaniga, NeurIPS 2023 —
[arXiv:2302.00294](https://arxiv.org/abs/2302.00294). Tooling: **DADApy** (the Laio group's
density/ID library), also `scikit-dimension`. Finding: ID expands early, contracts in the middle,
and the relative minimum of the ID profile marks the layer with the most semantically useful
representation. Cheap, and a good first "measure the geometry" muscle. Recent methodological
caution: *Rethinking Intrinsic Dimension Estimation in Neural Representations*
([arXiv:2604.20276](https://arxiv.org/abs/2604.20276)) and *Estimating Dimensionality of Neural
Representations from Finite Samples* ([arXiv:2509.26560](https://arxiv.org/abs/2509.26560)).

### 10. Topological data analysis of representations
Toolkit: **giotto-tda** ([JMLR 2021](https://www.jmlr.org/papers/volume22/20-325/20-325.pdf)),
curated list [AdaUchendu/AwesomeTDA4NLP](https://github.com/AdaUchendu/AwesomeTDA4NLP).
Persistent homology of attention maps (Kushnareva et al., artificial-text detection) and of
hidden-state point clouds; *Understanding Chain-of-Thought in LLMs via TDA*
([arXiv:2512.19135](https://arxiv.org/abs/2512.19135)). Least crowded of the ten, and the natural
tool if you want to *detect* a circle or torus rather than assume one.

---

## The project you can start today with zero setup

Your ESR data already poses a geometry question you are unusually well placed to answer, and
nobody has answered it in these terms:

> Steering displaces the residual stream off the manifold the model normally occupies.
> **Is the return to on-topic text a relaxation back onto that manifold?**

You have: ~1,000 saved generations across steered / unsteered / meta-prompted / ablated
conditions, the 600-continuation prefill set (off-topic prefix → the model silently recovers),
per-token activation capture (`08_episode_traces.py` already does forward passes with steering),
and calibrated per-latent boosts.

Concrete measurements, all with code you have:

1. Fit the **unsteered manifold** at layer 26 (PCA or a local-neighbourhood estimate over
   unsteered generations). Compute the distance of each steered token's activation from it.
2. Plot that distance **along the token index** for the prefill continuations: does it decay, and
   with what time constant? That is a relaxation-to-attractor measurement in your own language.
3. Compare **intrinsic dimension** and **manifold capacity** of the on-topic vs off-topic
   activation clouds (items 8 and 9 above give you the tools).
4. Causal test you have already built: does detector ablation change the *return trajectory*
   (geometry) even though it did not change the *outcome* (relevance)? Your null result becomes a
   much more interesting paper if the geometry moves while the behaviour does not.

This is a NeurReps/UniReps-workshop-shaped question, it uses data you already paid for in GPU
hours, and it is a real bridge between your neuroscience background and mechanistic
interpretability.

---

## Status

- **Project 1 (belief-state geometry) is built and running** in `D:\develief-geometry`
  (conda env `beliefgeom`). See its README for results.
- Setup plans for projects 2 and 3, plus a learning-ranked backlog, are in
  [SETUP_PLANS.md](SETUP_PLANS.md).

## Suggested order

1. **Belief-state geometry** (1 to 2 days) — build the vocabulary where ground truth exists.
2. **Your own ESR geometry analysis** (2 to 3 days) — no new data needed, directly yours.
3. **Circular features** (3 to 5 days) — the canonical non-linear-geometry replication, on your stack.
4. Then pick a direction: manifold steering (intervention-flavoured) or manifold capacity
   (neuroscience-flavoured) as the basis for an original project.
