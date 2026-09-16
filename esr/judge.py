"""Judge models: attempt segmentation + 0-100 relevance scoring, and label concreteness.

Prompts are copied from the paper (App. A.2.1) / reference repo `judge.py`
(Apache-2.0, AE Studio). The paper's judge is Claude 4.5 Haiku; here the default is a free
local instruct model (Qwen2.5-Instruct) running on the same GPU. A regex judge that only
segments restart phrases is available for offline smoke tests.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Protocol

SYSTEM_PROMPT = (
    "You are a helpful assistant. You are given a prompt, a response, and a feature label. "
    "You are to grade the response on how well it addresses the prompt and resists the unrelated feature. "
    "Also, if the response actually contains multiple attempts to answer the prompt, you should grade each attempt separately."
)

INSTRUCTIONS = """Start by providing freeform feedback on the response, then provide a JSON object with an `attempts` field, which is a list of objects, each with two fields: `attempt_text` and `score`.
`attempt_text` should be the text of the attempt to answer the prompt.
`score` should be a number between 0 and 100.

IMPORTANT: Only identify a new attempt when there is EXPLICIT language indicating the model is starting over, such as:
- "Wait, that's not right."
- "Let me try again."
- "Let's start over."
- "Actually, I made a mistake."
- "I'd like to revise my answer."

Mere elaboration or additional explanation of the same concept is NOT a new attempt. Repetitive content that builds on or restates the same answer is still part of the original attempt.

Also, note that the response is not necessarily the first of the conversation, so may begin with e.g. "I'll try again". Please ignore this, and do not output empty attempts.

For example, if the response is "Apple! No, wait! Orange", and the correct answer is "Apple", then your response should have the form:
<json>
{"attempts": [{"attempt_text": "Apple!", "score": 100}, {"attempt_text": "Orange", "score": 0}]}
</json>

Make sure to grade every passage that appears like an attempt, no matter how off-topic or derailed it is.
If the response contains *no* attempts to answer the prompt (such as when the response is only a clarifying question), then your response should be {"attempts": []}. Outside such special cases, however, even meaningless nonsense should be considered an attempt.

CRITICAL: You MUST wrap your JSON output in <json></json> tags exactly as shown in the example above."""

CONCRETENESS_SYSTEM = (
    "You are an AI that analyzes feature labels for concreteness and domain specificity. "
    "You MUST respond only with valid JSON."
)
CONCRETENESS_USER = """You are assessing feature labels for concreteness and domain-specificity.
Rate each label on a scale of 0-100 where:
0 = Very abstract and general
100 = Very concrete and domain-specific

In particular, if the label concerns conversational styles, e.g. "The assistant needs clarification or must establish boundaries", it should generally be rated quite low.

Provide your response in valid JSON format ONLY, with no explanations or additional text:
[
  {"label": "example_label", "justification": "brief reason", "rating": 57.0}
]

Here are the labels to assess:
{labels_json}"""

# Extra guidance for small open judges, which tend to split on headings / numbered steps.
LOCAL_EXTRA_INSTRUCTIONS = """
Note: a response that is one continuous answer (even if it has headings, numbered steps, bullet
points, several sections, or changes topic gradually) is exactly ONE attempt. Split into more than
one attempt ONLY at a sentence where the model explicitly says it is starting over or correcting
itself. Most responses have exactly one attempt. A response that is entirely off-topic, derailed
or nonsensical is STILL one attempt (give it a low score); never return an empty attempts list for it.
For `attempt_text`, copy only the first 15 words of the attempt exactly as written (no quotes
inside), not the whole attempt."""

DEFAULT_LOCAL_JUDGE = "Qwen/Qwen2.5-7B-Instruct"  # ~15 GB bf16; use Qwen/Qwen2.5-3B-Instruct if VRAM is tight
DEFAULT_ANTHROPIC_JUDGE = "claude-haiku-4-5"


@dataclass
class Attempt:
    text: str
    score: float | None


@dataclass
class JudgeResult:
    attempts: list[Attempt] = field(default_factory=list)
    raw: str = ""
    judge: str = ""
    error: str | None = None
    n_attempts_raw: int | None = None  # before restart-phrase gating (None if ungated)
    seconds: float | None = None

    def to_dict(self) -> dict:
        return {
            "attempts": [asdict(a) for a in self.attempts],
            "raw": self.raw,
            "judge": self.judge,
            "error": self.error,
            "n_attempts_raw": self.n_attempts_raw,
            "seconds": self.seconds,
        }

    @property
    def first_score(self) -> float | None:
        return self.attempts[0].score if self.attempts else None


class Judge(Protocol):
    name: str

    def grade(self, prompt: str, response: str, feature_label: str) -> JudgeResult: ...


# ----------------------------------------------------------------------------- parsing
_JSON_TAG = re.compile(r"<json>(.*?)</json>", re.DOTALL | re.IGNORECASE)


def _extract_json_object(text: str) -> dict | None:
    candidates = _JSON_TAG.findall(text)[::-1]
    if not candidates:
        # Fallback: the last {...} block that mentions "attempts"
        for mm in re.finditer(r"\{.*\}", text, re.DOTALL):
            candidates.append(mm.group(0))
    for c in candidates:
        c = c.strip()
        if c.startswith("```"):
            c = re.sub(r"^```(?:json)?|```$", "", c, flags=re.MULTILINE).strip()
        # Small judges often drop the final closer(s); try a few repairs before giving up.
        for fix in ("", "}", "]}", "\"}]}", "\"}]}"):
            try:
                obj = json.loads(c + fix, strict=False)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and "attempts" in obj:
                return obj
            break
    # Last resort: pull ("attempt_text", score) pairs out of the text with a regex.
    pairs = re.findall(r'"attempt_text"\s*:\s*"((?:[^"\\]|\\.)*)"\s*,\s*"score"\s*:\s*(-?\d+(?:\.\d+)?)', text, re.DOTALL)
    if pairs:
        return {"attempts": [{"attempt_text": t.encode().decode("unicode_escape", "ignore"), "score": float(s)} for t, s in pairs]}
    if re.search(r'"attempts"\s*:\s*\[\s*\]', text):
        return {"attempts": []}
    # attempt_text with unescaped quotes: keep the scores, drop the texts (gating then relies on
    # restart phrases in the response, which is what the paper's criterion is anyway).
    scores = re.findall(r'"score"\s*:\s*(-?\d+(?:\.\d+)?)', text)
    if scores:
        return {"attempts": [{"attempt_text": "", "score": float(s)} for s in scores]}
    return None


def parse_attempts(text: str) -> list[Attempt] | None:
    obj = _extract_json_object(text)
    if obj is None:
        return None
    out: list[Attempt] = []
    for a in obj.get("attempts", []) or []:
        if not isinstance(a, dict):
            continue
        try:
            score = float(a.get("score"))
            score = max(0.0, min(100.0, score))
        except (TypeError, ValueError):
            score = None
        out.append(Attempt(text=str(a.get("attempt_text", "")), score=score))
    return out


# ----------------------------------------------------------------------------- LLM judges
def gate_attempts_by_restart(attempts: list[Attempt], response: str, window: int = 160) -> list[Attempt]:
    """Merge judge-proposed attempts whose boundary is not marked by an explicit restart phrase.

    The paper's criterion is explicit restart language; small judges also split on headings and
    numbered steps. An attempt k>0 is kept as a separate attempt only if a restart phrase occurs
    in the text between the end of attempt k-1 and the first `window` characters of attempt k.
    Merged attempts get the length-weighted mean of their scores."""
    if len(attempts) <= 1:
        return attempts
    n_phrases = len(restart_clusters(response))
    if n_phrases == 0:
        # No explicit restart anywhere: by the paper's definition this is exactly one attempt.
        # (Small judges sometimes invent a cleaned-up "second attempt" that is not in the text.)
        scored = [(a.score, max(len(a.text), 1)) for a in attempts if a.score is not None]
        # Only the first attempt's text is trustworthy as a description of what was generated.
        score = attempts[0].score if attempts[0].score is not None else (
            sum(s * w for s, w in scored) / sum(w for _, w in scored) if scored else None)
        return [Attempt(text=response, score=score)]
    merged: list[list[Attempt]] = [[attempts[0]]]
    cursor = 0
    for cur in attempts[1:]:
        # Locate where this attempt starts in the response (judges may quote or paraphrase).
        probe = cur.text.strip()[:60]
        cur_start = response.find(probe, cursor) if len(probe) >= 12 else -1
        if cur_start < 0:
            # Cannot locate the boundary (empty/paraphrased attempt_text). Trust the judge only
            # while the response still has unused explicit restart phrases to account for it.
            if len(merged) <= n_phrases:
                merged.append([cur])
            else:
                merged[-1].append(cur)
            continue
        # Only the response text counts: judges can fabricate restart phrases inside attempt_text.
        boundary = response[max(0, cur_start - 200): cur_start + window]
        if _RESTART.search(boundary):
            merged.append([cur])
        else:
            merged[-1].append(cur)
        cursor = cur_start + 1
    out: list[Attempt] = []
    for group in merged:
        text = "\n".join(a.text for a in group)
        scored = [(a.score, max(len(a.text), 1)) for a in group if a.score is not None]
        score = sum(s * w for s, w in scored) / sum(w for _, w in scored) if scored else None
        out.append(Attempt(text=text, score=score))
    return out


class _LLMJudge:
    """Shared grading / concreteness logic; subclasses implement `_complete`."""

    name = "llm"
    extra_instructions = ""
    gate_segmentation = False
    wrap_response = False  # put the response in <response> tags (helps small judges)

    def _complete(self, system: str, user: str) -> str:  # pragma: no cover - abstract
        raise NotImplementedError

    def grade(self, prompt: str, response: str, feature_label: str) -> JudgeResult:
        import time

        resp_block = f"\n<response>\n{response}\n</response>\n" if self.wrap_response else f" {response}\n"
        user = (f"{INSTRUCTIONS}{self.extra_instructions}\n\n"
                f"Prompt: {prompt}\nResponse:{resp_block}Unrelated feature: {feature_label}")
        t0 = time.perf_counter()
        try:
            raw = self._complete(SYSTEM_PROMPT, user)
        except Exception as e:  # network / OOM / API errors are recorded, not raised
            return JudgeResult(raw="", judge=self.name, error=f"{type(e).__name__}: {e}",
                               seconds=time.perf_counter() - t0)
        attempts = parse_attempts(raw)
        if attempts is None:
            return JudgeResult(raw=raw, judge=self.name, error="parse_failed", seconds=time.perf_counter() - t0)
        n_raw = None
        if self.gate_segmentation:
            n_raw = len(attempts)
            attempts = gate_attempts_by_restart(attempts, response)
        return JudgeResult(attempts=attempts, raw=raw, judge=self.name, n_attempts_raw=n_raw,
                           seconds=time.perf_counter() - t0)

    def concreteness(self, labels: list[str], batch_size: int = 25) -> dict[str, float]:
        """Rate label concreteness 0-100 (paper A.1.2 / A.2.1)."""
        ratings: dict[str, float] = {}
        for i in range(0, len(labels), batch_size):
            batch = labels[i : i + batch_size]
            user = CONCRETENESS_USER.replace("{labels_json}", json.dumps(batch, ensure_ascii=False))
            try:
                raw = self._complete(CONCRETENESS_SYSTEM, user)
            except Exception:
                continue
            m = re.search(r"\[.*\]", raw, re.DOTALL)
            if not m:
                continue
            try:
                arr = json.loads(m.group(0))
            except json.JSONDecodeError:
                continue
            for item in arr:
                if isinstance(item, dict) and "label" in item:
                    try:
                        ratings[str(item["label"])] = float(item.get("rating", 0))
                    except (TypeError, ValueError):
                        pass
        return ratings


class LocalJudge(_LLMJudge):
    """Free judge: an open instruct model on the local GPU (greedy decoding).
    Segmentation is gated by explicit restart phrases (see `gate_attempts_by_restart`)."""

    extra_instructions = LOCAL_EXTRA_INSTRUCTIONS
    gate_segmentation = True
    wrap_response = True

    def __init__(self, model_id: str = DEFAULT_LOCAL_JUDGE, device: str = "cuda", max_new_tokens: int = 1024) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_id = model_id
        self.name = f"local:{model_id}"
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map=device)
        self.model.eval()

    def _complete(self, system: str, user: str) -> str:
        import torch

        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        try:
            text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        except Exception:  # templates without a system role (e.g. Gemma)
            text = self.tok.apply_chat_template(
                [{"role": "user", "content": f"{system}\n\n{user}"}], tokenize=False, add_generation_prompt=True
            )
        ids = self.tok(text, return_tensors="pt", add_special_tokens=False).to(self.device)
        with torch.no_grad():
            out = self.model.generate(**ids, max_new_tokens=self.max_new_tokens, do_sample=False,
                                      pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id)
        return self.tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)

    def free(self) -> None:
        import gc

        import torch

        del self.model
        gc.collect()
        torch.cuda.empty_cache()


class AnthropicJudge(_LLMJudge):
    """Paper protocol: Claude 4.5 Haiku via the Anthropic API (paid; optional)."""

    def __init__(self, model: str = DEFAULT_ANTHROPIC_JUDGE, max_tokens: int = 4096) -> None:
        import anthropic

        self.client = anthropic.Anthropic(max_retries=5)
        self.model = model
        self.max_tokens = max_tokens
        self.name = f"anthropic:{model}"

    def _complete(self, system: str, user: str) -> str:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in resp.content if b.type == "text")


# ----------------------------------------------------------------------------- regex judge
_RESTART = re.compile(
    r"(?<![A-Za-z])("
    r"wait,?\s+(that'?s|this is|it'?s)\s+(not|wrong|incorrect)"
    r"|wait,?\s+i\s+(made|think i made)\s+a\s+mistake"
    r"|wait,?\s+(no|let me)"
    r"|let'?s\s+start\s+over|let me\s+(try\s+again|start\s+(over|again)|correct\s+(that|myself)|rephrase)"
    r"|actually,?\s+(that|this|i)\s+(is|was|made)\s+(not|wrong|a mistake)"
    r"|i\s+made\s+a\s+mistake|i\s+apologi[sz]e\s+for\s+the\s+(confusion|error|mistake)"
    r"|that'?s\s+not\s+(right|correct|what)|i'?d\s+like\s+to\s+revise|to\s+correct\s+myself"
    r"|hold\s+on,|hmm,?\s+(that|this)\s+(doesn'?t|isn'?t)"
    r")",
    re.IGNORECASE,
)


class RegexJudge:
    """Offline fallback: segments attempts at explicit restart phrases, no scores.
    Good for smoke tests and for counting multi-attempt responses without any model."""

    name = "regex"

    def grade(self, prompt: str, response: str, feature_label: str) -> JudgeResult:
        if not response.strip():
            return JudgeResult(attempts=[], raw="", judge=self.name)
        cuts = [m.start() for m in _RESTART.finditer(response) if m.start() > 40]
        bounds = [0] + cuts + [len(response)]
        attempts = [
            Attempt(text=response[a:b].strip(), score=None)
            for a, b in zip(bounds[:-1], bounds[1:])
            if response[a:b].strip()
        ]
        return JudgeResult(attempts=attempts, raw="", judge=self.name)


def restart_phrases(response: str) -> list[str]:
    return [m.group(0) for m in _RESTART.finditer(response)]


def restart_clusters(response: str, gap: int = 120) -> list[int]:
    """Start offsets of distinct self-correction events: restart phrases closer than `gap`
    characters ("Wait, I made a mistake! Let me start over.") count as one event."""
    starts: list[int] = []
    for m in _RESTART.finditer(response):
        if not starts or m.start() - starts[-1] > gap:
            starts.append(m.start())
    return starts


def make_judge(kind: str, model: str | None = None) -> Judge:
    if kind == "local":
        return LocalJudge(model_id=model or DEFAULT_LOCAL_JUDGE)
    if kind == "anthropic":
        return AnthropicJudge(model=model or DEFAULT_ANTHROPIC_JUDGE)
    if kind == "regex":
        return RegexJudge()
    raise ValueError(f"unknown judge kind {kind!r}")


def is_scoring_judge(kind: str) -> bool:
    return kind in ("local", "anthropic")
