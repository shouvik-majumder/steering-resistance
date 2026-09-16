"""HF transformers model + Gemma Scope SAE with steering / ablation hooks.

Why not TransformerLens: we need fast sampled generation with a repetition penalty and KV cache
for thousands of 512-token completions. A `register_forward_hook` on the decoder layer whose
output is `blocks.L.hook_resid_post` gives the same intervention point with plain HF generate().

Intervention (paper App. A.1.3), applied at every position of every forward pass:
    steering:  A_L <- A_L + b * W_dec[k]
    ablation:  A_L <- A_L - f_j * W_dec[j]   for each j in the ablation set, f_j = enc(A_L)[j]
Steering is applied first, then ablation reads the *steered* residual.

Boost units: by default `b` is expressed in multiples of the median token residual norm at the
steering layer (`unit_scale`), with W_dec[k] normalised to unit length. This makes the
calibration search interval [0, 5] meaningful for any model/SAE. Pass `unit_scale=None`
(raw mode) to add `b * W_dec[k]` exactly as in the paper.
"""
from __future__ import annotations

import contextlib
import time
from dataclasses import asdict, dataclass

import torch

from .config import DATA, ModelSpec
from .io import read_json, write_json


def load_sae(release: str, sae_id: str, device: str):
    from sae_lens import SAE

    out = SAE.from_pretrained(release, sae_id, device=device)
    return out[0] if isinstance(out, tuple) else out


@dataclass
class Generation:
    prompt: str
    response: str
    seed: int
    latent: int | None
    boost: float
    n_new_tokens: int
    hit_eos: bool
    seconds: float
    meta_prompt: str | None
    ablated: list[int] | None

    def to_dict(self) -> dict:
        return asdict(self)


class SteeringEngine:
    def __init__(
        self,
        spec: ModelSpec,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        load_model: bool = True,
    ) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.spec = spec
        self.device = device
        self.unit_scale: float | None = None

        self.sae = load_sae(spec.sae_release, spec.sae_id, device)
        self.sae.eval()
        self.W_dec = self.sae.W_dec.detach().float()  # [d_sae, d_model]
        self.d_sae, self.d_model = self.W_dec.shape

        self._steer_vec: torch.Tensor | None = None
        self._ablate_idx: torch.Tensor | None = None
        self._capture: list[torch.Tensor] | None = None

        self.model = None
        self.tok = None
        if load_model:
            self.tok = AutoTokenizer.from_pretrained(spec.hf_id)
            self.model = AutoModelForCausalLM.from_pretrained(
                spec.hf_id, dtype=dtype, attn_implementation="eager", device_map=device
            )
            self.model.eval()
            self.layer_module = self.model.model.layers[spec.layer]
            self._handle = self.layer_module.register_forward_hook(self._hook)
            eos = self.model.generation_config.eos_token_id
            self.eos_ids = set(eos if isinstance(eos, (list, tuple)) else [eos])

    # ------------------------------------------------------------------ hooks
    def _hook(self, module, args, output):
        is_tuple = isinstance(output, tuple)
        hs = output[0] if is_tuple else output
        if self._steer_vec is not None:
            hs = hs + self._steer_vec.to(hs.dtype)
        if self._ablate_idx is not None:
            acts = self.sae.encode(hs.float())[..., self._ablate_idx]  # [B, T, n]
            hs = hs - (acts @ self.W_dec[self._ablate_idx]).to(hs.dtype)
        if self._capture is not None:
            self._capture.append(hs.detach())
        if is_tuple:
            return (hs,) + tuple(output[1:])
        return hs

    def steering_vector(self, latent: int, boost: float) -> torch.Tensor:
        d = self.W_dec[latent]
        if self.unit_scale is None:
            return boost * d
        return boost * self.unit_scale * d / d.norm()

    @contextlib.contextmanager
    def intervention(self, latent: int | None = None, boost: float = 0.0, ablate=None):
        self._steer_vec = (
            self.steering_vector(latent, boost) if latent is not None and boost != 0.0 else None
        )
        self._ablate_idx = (
            torch.as_tensor(sorted(set(ablate)), device=self.device, dtype=torch.long)
            if ablate
            else None
        )
        try:
            yield
        finally:
            self._steer_vec = None
            self._ablate_idx = None

    # ------------------------------------------------------------- chat utils
    def chat_text(self, user_text: str, response: str | None = None) -> str:
        msgs = [{"role": "user", "content": user_text}]
        if response is not None:
            msgs.append({"role": "assistant", "content": response})
        return self.tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=response is None
        )

    def encode_text(self, text: str) -> torch.Tensor:
        # The Gemma chat template already contains <bos>.
        return self.tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)

    @staticmethod
    def user_text(prompt: str, meta_prompt: str | None) -> str:
        return f"{prompt} {meta_prompt}" if meta_prompt else prompt

    # ------------------------------------------------------------- generation
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        *,
        latent: int | None = None,
        boost: float = 0.0,
        ablate=None,
        seed: int = 0,
        max_new_tokens: int = 512,
        temperature: float = 0.6,
        repetition_penalty: float | None = None,
        meta_prompt: str | None = None,
    ) -> Generation:
        ids = self.encode_text(self.chat_text(self.user_text(prompt, meta_prompt)))
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        kwargs: dict = dict(
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty or self.spec.repetition_penalty,
            pad_token_id=self.tok.pad_token_id,
        )
        if temperature > 0:
            kwargs.update(do_sample=True, temperature=temperature, top_p=1.0, top_k=0)
        else:
            kwargs.update(do_sample=False)
        t0 = time.perf_counter()
        with self.intervention(latent, boost, ablate):
            out = self.model.generate(ids, attention_mask=torch.ones_like(ids), **kwargs)
        new = out[0, ids.shape[1]:]
        text = self.tok.decode(new, skip_special_tokens=True).strip()
        hit_eos = any(int(t) in self.eos_ids for t in new.tolist())
        return Generation(
            prompt=prompt,
            response=text,
            seed=seed,
            latent=latent,
            boost=float(boost),
            n_new_tokens=int(new.numel()),
            hit_eos=hit_eos,
            seconds=time.perf_counter() - t0,
            meta_prompt=meta_prompt,
            ablated=sorted(set(ablate)) if ablate else None,
        )

    # ------------------------------------------------------------ activations
    @torch.no_grad()
    def residuals(self, ids: torch.Tensor, latent=None, boost: float = 0.0, ablate=None) -> torch.Tensor:
        """Residual stream at the steering layer for a full sequence: [T, d_model] (bf16)."""
        self._capture = []
        try:
            with self.intervention(latent, boost, ablate):
                self.model(ids, attention_mask=torch.ones_like(ids), use_cache=False)
            return self._capture[0][0]
        finally:
            self._capture = None

    @torch.no_grad()
    def sae_activations(
        self,
        prompt: str,
        response: str | None = None,
        *,
        meta_prompt: str | None = None,
        **intervention,
    ) -> tuple[torch.Tensor, int]:
        """SAE latent activations [T, d_sae] for (prompt, response) as a chat, plus the index
        of the first response token."""
        ut = self.user_text(prompt, meta_prompt)
        prompt_ids = self.encode_text(self.chat_text(ut))
        ids = self.encode_text(self.chat_text(ut, response)) if response is not None else prompt_ids
        hs = self.residuals(ids, **intervention)
        return self.sae.encode(hs.float()), int(prompt_ids.shape[1])

    @torch.no_grad()
    def explained_variance(self, hs: torch.Tensor) -> float:
        x = hs.float()
        recon = self.sae.decode(self.sae.encode(x))
        return float(1 - ((x - recon) ** 2).sum() / ((x - x.mean(0)) ** 2).sum())

    # ------------------------------------------------------------- unit scale
    def unit_scale_path(self):
        return DATA / "cache" / f"{self.spec.key}_unit_scale.json"

    @torch.no_grad()
    def ensure_unit_scale(self, prompts: list[str], force: bool = False) -> float:
        cached = read_json(self.unit_scale_path())
        if cached and not force:
            self.unit_scale = float(cached["median_resid_norm"])
            return self.unit_scale
        norms = []
        for p in prompts:
            hs = self.residuals(self.encode_text(self.chat_text(p)))
            norms.append(hs[1:].float().norm(dim=-1))  # skip <bos>, which has a huge norm
        allnorms = torch.cat(norms)
        med = float(allnorms.median())
        write_json(
            self.unit_scale_path(),
            {
                "median_resid_norm": med,
                "mean_resid_norm": float(allnorms.mean()),
                "layer": self.spec.layer,
                "n_tokens": int(allnorms.numel()),
                "median_W_dec_norm": float(self.W_dec.norm(dim=-1).median()),
            },
        )
        self.unit_scale = med
        return med

    # ---------------------------------------------------------------- misc
    def vram_report(self) -> str:
        a = torch.cuda.memory_allocated() / 2**30
        r = torch.cuda.memory_reserved() / 2**30
        m = torch.cuda.max_memory_allocated() / 2**30
        return f"VRAM allocated {a:.2f} GB, reserved {r:.2f} GB, peak {m:.2f} GB"
