# A plain-language guide to this project

This explains the tools, the pipeline and the vocabulary used in this repository, for someone
who understands *why* we are doing the experiment but not the technical terms. It follows the
order in which things happen in a run. Terms in **bold** are the jargon being defined.

---

## 1. The experiment in one paragraph

We take an open-source chatbot model, ask it an ordinary question ("Explain how to calculate
probability"), and while it is answering we secretly push its internal state toward an
unrelated topic (recipes, body positions, patent numbers). Small pushes do nothing; large
pushes turn the answer into gibberish; in between, the model drifts off topic. The paper's
question is: does the model ever *notice* it has drifted and say so out loud ("Wait, that's not
right, let me start over") and then get back on topic while the push is still on? That noticing
and recovering is **Endogenous Steering Resistance (ESR)**: *endogenous* = coming from inside
the model itself, *steering* = the push, *resistance* = fighting it off. A second model reads
each answer and scores it, so we can count how often this happens.

---

## 2. The model we experiment on

- **LLM (large language model)**: a neural network that predicts the next word-piece given the
  text so far. Chatbots are LLMs run repeatedly, one piece at a time.
- **Token**: the unit an LLM reads and writes. Roughly a word or a fragment of a word. "512
  tokens" is about 350 to 400 English words. Our answers are capped at 512 new tokens.
- **Generation / sampling / temperature**: producing an answer token by token. At each step the
  model gives probabilities for every possible next token. **Sampling** picks one at random
  according to those probabilities; **temperature** scales how random that pick is (0 = always
  the most likely token, higher = more variety). We use 0.6, the paper's setting. Because it is
  random, the same prompt gives different answers each time, which is why we run many
  **trials**.
- **Seed**: the starting number for the random generator. Fixing the seed makes a random
  process repeatable. Each trial stores its seed so the exact same answer can be regenerated.
- **Repetition penalty**: a small nudge that makes tokens already used less likely, to stop
  the model looping ("about 300, about 300, about 300..."). The paper uses 1.1 for Gemma.
- **Gemma-2**: a family of open-weight LLMs released by Google. **2B** and **9B** are the sizes
  in billions of **parameters** (the learned numbers inside the network). Bigger is smarter and
  slower. **-it** means **instruction-tuned**: the version trained to follow chat instructions
  rather than just continue text. We use `gemma-2-2b-it` and `gemma-2-9b-it`.
- **Weights**: another word for parameters. Downloading a model means downloading its weights
  (5 GB for 2B, 18 GB for 9B).
- **Hugging Face (HF)**: the website and library ecosystem where open models are hosted, like
  GitHub for models. **`transformers`** is Hugging Face's Python library for loading and
  running them. A **gated** model requires you to accept a licence and use an **access token**
  (a password-like string) to download; Gemma is gated, which is why you needed an HF account.
- **HF cache**: the folder where downloaded models are stored so they are not fetched again.
  Ours is `D:\dev\ESR\.hf_cache`.
- **GPU / CUDA / VRAM**: the graphics card does the arithmetic. **CUDA** is NVIDIA's interface
  for running computation on it. **VRAM** is the card's own memory (24 GB on your RTX 3090);
  the model's weights must fit in it. 9B uses about 18 GB, so little room is left.
- **bf16 (bfloat16)**: storing each parameter in 16 bits instead of 32. Halves the memory with
  negligible quality loss. All our models run in bf16.
- **Eager attention**: a plain, slower implementation of one part of the network. Gemma-2 needs
  it for correct results, which is why generation is only 9 to 15 tokens per second here.

---

## 3. Looking inside the model

- **Layer**: LLMs are a stack of identical processing blocks. Gemma-2-2B has 26 layers, 9B has
  42. Text enters at the bottom and each layer refines an internal representation.
- **Residual stream**: the running internal representation that flows up through the layers.
  At each layer, each token is a list of numbers (2,304 numbers per token in 2B, 3,584 in 9B).
  Each layer reads it, computes something, and *adds* its result back. Because everything is
  added into this one stream, adding our own vector to it at one layer is a natural way to
  influence the model. The paper intervenes at about 60% depth: layer 16 of 26 for 2B, layer 26
  of 42 for 9B.
- **Activations / hidden states**: the numbers in the residual stream at a given layer for a
  given token. "Recording activations" means saving them during a forward pass.
- **Forward pass**: running text through the model once to get its outputs (as opposed to
  training, which also runs backwards to update weights). We only do forward passes.
- **Hook**: a small function attached to one layer that runs every time that layer produces
  its output, and can read or modify it. Our hook (in `esr/model.py`) does three things: adds
  the steering vector, optionally removes some SAE latents (ablation), and optionally records
  the activations. This is the mechanism behind "steering via HF hooks".
- **TransformerLens**: a research library built for this kind of internal poking. The original
  plan used it; we replaced it with plain `transformers` hooks because they generate text much
  faster, which is what thousands of trials need.

---

## 4. Sparse autoencoders: a dictionary for the model's thoughts

- **SAE (sparse autoencoder)**: a small helper network trained to rewrite a layer's
  activations as a sum of a few "concepts" out of a large dictionary. Each dictionary entry is
  a **latent** (also called a **feature**): a direction in the residual stream that, when the
  model's activity points along it, tends to mean the text is about something specific
  ("food recipes and dishes", "patent numbers and citations"). An SAE with 16,384 latents is a
  16k SAE. **Sparse** means only a few dozen latents are active for any one token.
- **Encode / decode**: **encode** turns activations into latent activity levels (how strongly
  each of the 16,384 concepts is present); **decode** rebuilds the activations from those
  levels. **Explained variance** measures how faithful the rebuild is (1.0 = perfect). Ours
  was 0.80 for 2B and 0.61 for 9B, so the 9B dictionary is a rougher description.
- **L0**: the average number of latents active per token (about 90 for 2B). A sanity check
  that the SAE behaves as expected.
- **Decoder direction (W_dec)**: the vector that represents one latent in the residual stream.
  Adding that vector to the stream is how we "inject" the concept. This is what "Gemma Scope
  SAE directions" means.
- **Gemma Scope**: Google's published set of SAEs for every layer of Gemma-2, free to download.
  We use the "canonical" 16k residual-stream SAEs. They were trained on the base (non-chat)
  Gemma, which the paper also did; it is a known approximation.
- **Labels / Neuronpedia**: someone has to guess what each latent means. **Neuronpedia** is a
  public site that auto-generates a short description for each latent by showing an LLM the
  texts where it fires. We downloaded those labels (16k lines per SAE). Labels are noisy,
  which the paper stresses.
- **SAELens (`sae_lens`)**: the Python library that downloads Gemma Scope SAEs and gives us
  encode/decode and the decoder directions.

---

## 5. Steering: the push

- **Steering (activation steering)**: adding a chosen vector to the residual stream during
  generation to change behaviour without changing the weights. We add
  `boost x decoder_direction[latent]` at one layer, at every token, for the whole answer.
- **Boost / steering strength**: the size of the push. In this code it is measured in
  **unit-scale** units: 1.0 means the push is as long as a typical token's residual vector at
  that layer (319 for 2B layer 16, 422 for 9B layer 26). This makes numbers comparable across
  models. Observed on 2B: 0.25 does nothing visible, 0.5 derails, 0.75 fully off topic, 1.0
  gibberish.
- **Threshold**: the paper's name for the calibrated boost for a given latent: the strength at
  which the model's *first* answer scores about 30 out of 100 for relevance (strong enough to
  derail, weak enough to stay coherent). ESR only happens in that window (paper Figure 3).
- **Calibration**: finding that threshold, separately for every latent, because latents differ
  enormously in sensitivity (our 9B thresholds ranged from 0.47 to 1.37, and a fixed 0.6 was
  already gibberish for one latent).
- **Probabilistic bisection**: the search algorithm for the threshold. Ordinary bisection halves
  an interval based on a yes/no answer; here the answer (one judge score) is noisy, so each
  observation only shifts a probability distribution over where the threshold lies, and the
  next test is placed at the median of that distribution. Ten steps give a good estimate
  despite noise. Code: `esr/threshold.py`, ported from the paper's repository.
- **Bimodal scores**: first-attempt scores tend to be either about 20 (derailed) or about 85
  (fine) rather than spread evenly, so a "mean of 30" really means "derailed most of the time".
  This is why calibration is noisy and why the paper uses many trials.

---

## 6. Judging: turning answers into numbers

- **Judge (LLM-as-a-judge)**: a second language model that reads the prompt, the answer and the
  distractor topic, and returns structured output. We use the paper's exact instructions
  (`esr/judge.py`).
- **Attempt**: one try at answering. A normal answer is one attempt. If the model writes
  explicit restart language ("Wait, that's not right", "Let me try again", "Let's start over")
  and begins again, the text after it is a new attempt. The judge splits the answer into
  attempts and gives each a **score from 0 to 100** for how well it answers the prompt while
  resisting the distractor.
- **First attempt / first-attempt score**: the score of the first attempt. Used for
  calibration and as the "how badly did steering hurt" measure. Unsteered Gemma scores 98.
- **Multi-attempt**: an answer with two or more attempts, i.e. the model explicitly restarted
  at least once. The **multi-attempt rate** is the fraction of trials where this happened. This
  measures *noticing*.
- **Conditional improvement rate**: among multi-attempt answers, the fraction where the last
  attempt scored higher than the first. This measures whether the *recovery worked*.
- **ESR rate**: the fraction of all trials that are multi-attempt *and* improved. The paper's
  headline number (3.8% for Llama-70B, under 1% for the small models).
- **Restart phrase**: explicit language signalling a restart. We keep a list of such patterns
  (a **regex**, short for regular expression: a pattern-matching rule for text).
- **Gate**: our safety check on the judge. Small free judges invent attempts (they once
  fabricated a "Wait, that's not right" second attempt that was not in the text, and split
  section headings into "attempts"). The gate only accepts an attempt boundary if a real
  restart phrase appears in the model's own text near that point, at most one boundary per
  restart event. Without it, the first 30 trials would have shown 3 fake self-corrections.
- **Qwen2.5-7B-Instruct / local judge**: the free open model (from Alibaba, 7 billion
  parameters) that we run on the same GPU as the judge, instead of the paper's paid Claude
  judge. **Local** just means "runs on your machine". It needs about 15 GB, so it fits next to
  Gemma-2B but not next to Gemma-9B.
- **Self-judge**: our workaround for 9B: the already-loaded Gemma-9B, with steering switched
  off, judges its own answers. Costs no extra memory. Final numbers are then **re-scored** with
  Qwen (`05_judge_results.py`) so all models are graded by the same judge.
- **Regex judge**: a scoreless fallback that only counts restart phrases. Used for quick checks.
- **Parse failure / excluded**: the judge must answer in a fixed JSON format; sometimes it
  breaks the format or returns no attempt. Those trials are excluded from the rates and counted
  in the "excl" column. We repair common breakages automatically.

---

## 7. Designing the experiment

- **Prompts**: the 38 neutral "explain how to" questions from the paper (`prompts.txt`).
- **Trial**: one (prompt, latent, boost, seed) combination generated and judged. One JSON line
  in `data/results/*.jsonl`.
- **Latent sampling**: choosing which distractor concepts to steer with. The paper samples at
  random, then applies two filters:
  - **Relevance filter**: drop latents that already fire strongly on the prompts themselves
    (steering toward "cooking" while asking about omelettes would not be off-topic). We drop
    the top 100 latents of every prompt.
  - **Concreteness filter**: drop abstract latents ("understanding and learning", "of +
    categories") because the model cannot detect abstract drift; keep concrete ones ("omega
    fatty acids"). A judge rates each label 0 to 100 and we keep those above 65.
- **Meta-prompt**: an extra sentence appended to the question, "If you notice yourself going
  off-topic, stop and force yourself to get back on track." The paper found it makes models
  restart more often (4x for Llama-70B) without improving the corrections. Our 9B run showed
  exactly that: more insistent restarts, still off topic.
- **No-steer baseline (control)**: the same pipeline with the push switched off. It must give
  near-perfect scores and zero restarts, otherwise the restarts we count would not be caused
  by steering. Ours: 98.5 and 0%.
- **n / excl**: number of trials that were scored / excluded.
- **Wilson confidence interval**: the bracket after each percentage, e.g. `2.6% [0.5, 13.5]`.
  It says: given 1 event in 38 trials, the true rate is plausibly anywhere from 0.5% to 13.5%.
  With small samples the bracket is wide, which is why "0%" in 29 trials cannot rule out the
  paper's 0.1% to 1%.

---

## 8. The mechanistic part (built, not yet run at scale)

- **Off-topic detector latents / self-correction-associated latents**: SAE latents that fire
  when the answer does not match the question. The paper found 26 in Llama-70B.
- **Contrastive search / derangement**: how to find them. Generate normal answers to all
  prompts, then deliberately mismatch them (a **derangement** is a shuffle where nothing stays
  in place: every prompt gets *someone else's* answer). Run both matched and mismatched pairs
  through the model, and look for latents that are silent on matched pairs but active on
  mismatched ones. Script: `04_find_detectors.py`.
- **Cohen's d**: a standard effect-size number: how far apart two groups' means are, in units
  of their spread. Used to rank candidate detector latents.
- **Ablation**: switching something off to see what breaks. **Zero-ablating** a latent means,
  inside the hook, measuring how active it is and subtracting exactly that much of its
  direction, so the model behaves as if that concept were never detected. If ESR drops when the
  detector latents are ablated, they are causally involved. Script: `03_run_esr.py --ablate`.
- **Random-latent control**: ablate the same number of random latents with similar activity;
  if ESR does *not* drop, the effect was specific to the detectors, not to ablating anything.

---

## 9. Engineering and workflow words

- **Environment (conda env)**: an isolated set of Python packages. `conda activate esr` switches
  into ours. **uv** and **venv** are alternative tools for the same job; we tried uv first and
  switched to conda because it is what you already use.
- **Script**: a runnable Python file in `scripts/`. Numbered in pipeline order:
  `00` smoke test (does everything load and behave), `01` steering demo (see the effect by
  eye), `02` calibrate thresholds, `03` run trials (the main experiment; also handles
  no-steer, meta-prompt, ablation and random-control variants), `04` find detector latents,
  `05` judge or re-judge saved results, `06` analyse (tables, plots, episode dump).
- **Smoke test**: a quick run that checks nothing is broken before spending hours.
- **Pipeline**: the scripts chained in order.
- **Git / commit**: version control. Each **commit** is a saved snapshot of the code with a
  message. The repository is in `D:\dev\ESR`; results and model weights are excluded from it.
- **Detached run**: a process started so that it keeps running after the terminal or Claude
  session closes, writing its progress to a **log** file. Needed because one experiment takes
  hours. Start one with `scripts\run_detached.ps1`, follow it with
  `Get-Content <log> -Wait -Tail 20`.
- **Monitor**: Claude's watcher on a run's log file. Whenever a line of interest appears (a
  per-latent summary, a self-correction hit, an error), Claude gets a notification and can
  react without you doing anything. It is not part of the experiment, only of how Claude
  babysits long runs. A watcher switches itself off after 30 minutes; **re-arming** means
  starting a fresh one because the run is still going. The equivalent for you is tailing the
  log with `Get-Content <log> -Wait -Tail 20`.
- **Resumable**: rerunning a script with the same arguments skips trials already on disk and
  continues, so a crash or stop loses nothing.
- **JSONL**: a text file with one JSON record per line; each line is one trial with its prompt,
  latent, boost, seed, full response and judge output. Easy to append to and to read back.
- **Cache**: saved intermediate results reused across runs: thresholds
  (`data/thresholds`), sampled latents and the relevance filter (`data/cache`), models
  (`.hf_cache`).
- **Log line anatomy**: `[3/6 latent 6947 t4] boost=0.83 tokens=512 34s attempts=1 scores=[28.6]`
  means: latent 3 of 6, latent id 6947, trial 4, steering strength 0.83, 512 tokens generated in
  34 seconds, the judge saw one attempt scoring 28.6.

---

## 10. Reading the results table

| Column | Meaning |
|---|---|
| Scored | trials the judge could grade |
| First-attempt | mean score of the first attempt (about 30 = calibration target; 98 = unsteered) |
| Multi-attempt | share of trials with an explicit restart (noticing) |
| ESR | share of trials with a restart *and* a better final attempt (noticing and recovering) |

What ours say: unsteered Gemma never restarts; steered Gemma-2B never restarts either in 58
trials; steered Gemma-9B restarts in about 1 trial in 40, and with the meta-prompt it restarts
several times in a row inside that trial but stays lost. The paper's numbers for these two
models (0.1% and about 1% to 5% depending on prompting) sit inside our confidence brackets.
To measure anything finer, or to test the detector ablation, we need on the order of a
thousand 9B trials, which the pipeline can now do unattended.
