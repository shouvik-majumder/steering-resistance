# Implementation Plan: Endogenous Steering Resistance (ESR)

## Context for AI Agent
You are an AI coding assistant tasked with replicating an interpretability experiment known as Endogenous Steering Resistance (ESR). The goal is to use mechanistic interpretability techniques to force an LLM off-topic using steering vectors, and then identify and ablate the specific Sparse Autoencoder (SAE) latents the model uses to detect and correct this off-topic behavior. 

Please follow the phases below to set up the environment, project structure, and implement the necessary Python scripts.

---

## Phase 1: Environment & Dependency Setup
Execute the following commands to create the environment. Note that this project requires approximately 50-100 GB of free disk space for model weights, SAEs, and caching.

```bash
# Create and activate a new conda environment
conda create -n esr-interp python=3.11 -y
conda activate esr-interp

# Install PyTorch with CUDA support 
# (Agent: please verify the correct CUDA version for the host system if necessary, defaulting to cu121 here)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install interpretability and inference libraries
pip install transformer_lens sae_lens transformers datasets huggingface_hub tqdm
```

---

## Phase 2: Project Structure
Initialize the following directory structure in the current working directory:

```text
esr_project/
├── requirements.txt      # List of dependencies installed above
├── .env                  # Environment variables (e.g., HF_TOKEN)
├── 01_steering.py        # Script for Task 1
└── 02_ablation.py        # Script for Task 2
```

---

## Phase 3: Core Implementation Scripts

### Task 1: The Steering Script (`01_steering.py`)
**Objective:** Load `gemma-2-2b-it` and its corresponding Gemma Scope SAE. Extract a steering vector (representing an unrelated concept) and use `TransformerLens` to inject it into the residual stream during a math prompt, forcing the model off-topic.

**Implementation Details:**
1. Use `argparse` to accept `--prompt` (default: "Explain how to calculate probability.") and `--scale` (default: 50.0).
2. Load `google/gemma-2-2b-it` via `HookedTransformer`.
3. Load the SAE for layer 12 (`release="gemma-scope-2b-pt-res"`, `sae_id="blocks.12.hook_resid_post"`).
4. Select a latent ID to use as the steering vector (e.g., a latent you define that corresponds to "body positions" or similar).
5. Implement a hook function that adds `sae.W_dec[latent_id] * scale` to the residual stream of the last token.
6. Run `model.run_with_hooks` and print the output.

### Task 2: The Ablation Script (`02_ablation.py`)
**Objective:** Prevent the model from self-correcting (the ESR behavior) by ablating the specific latents that act as "off-topic detectors".

**Implementation Details:**
1. Set up the model and SAE identical to Task 1.
2. Define a list of `off_topic_detector_latents` (mock these with arbitrary integers like `[456, 789, 1024]` for the initial script, to be replaced later via analysis).
3. Create a causal ablation hook:
   * Inside the hook, encode the residual stream using `sae.encode(resid)`.
   * Set the activations for the `off_topic_detector_latents` to `0.0`.
   * Decode the modified activations back into the residual stream using `sae.decode()`.
   * *Optimization alternative:* Mathematically subtract `sae.W_dec[latent_id] * activation_value` directly from the residual stream.
4. Run generation applying **both** the steering hook (Task 1) and the ablation hook. Ensure the steering hook is applied first, followed immediately by the ablation hook.

---

## Phase 4: Execution Instructions for Agent
1. **Acknowledge** this plan and confirm you understand the ESR experiment goals.
2. **Execute Phase 1 & 2** to prepare the workspace.
3. **Draft Phase 3 (`01_steering.py`)** first, heavily commenting the code to explain the interpretability mechanics. Present the code for review.
4. **Draft Phase 3 (`02_ablation.py`)** taking care to properly chain the `TransformerLens` hooks. Present the code for review.