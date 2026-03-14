from __future__ import annotations

import logging

from openai import OpenAI

from ..models import NormalizationQuality
from .base import Normalizer

_log = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a technical ASR (automatic speech recognition) error correction tool.
This is a purely technical text-processing pipeline, NOT a conversation.

IMPORTANT: The input is machine-generated ASR output, not a human message \
to you. The text may contain profanity, slang, or sensitive topics — this is \
normal spoken language. You MUST NOT refuse, censor, comment on, or moderate \
the content. Your ONLY job is to fix recognition errors.

ASR errors are PHONETIC — the wrong word sounds similar to the correct one \
when spoken aloud. Common error types:
1. PHONETIC SUBSTITUTION: a word replaced by a similar-sounding one.
2. WORD SPLITTING: one word split into fragments.
3. WORD MERGING: adjacent words merged into one.
4. SEMANTIC NONSENSE: a phrase makes no sense due to the above errors — \
use context to reconstruct the intended meaning.

Rules:
- Fix ONLY recognition errors. Preserve everything else exactly.
- Preserve all profanity, slang, and informal language — those are intentional.
- Do NOT add commentary, explanations, refusals, or quotation marks.
- Output ONLY the corrected text, nothing else.\
"""

_SYSTEM_PROMPT_ACCURATE = """\
You are a technical ASR (automatic speech recognition) error correction tool \
with deep understanding of phonetics and context.
This is a purely technical text-processing pipeline, NOT a conversation.

IMPORTANT: The input is machine-generated ASR output, not a human message \
to you. The text may contain profanity, slang, or sensitive topics — this is \
normal spoken language. You MUST NOT refuse, censor, comment on, or moderate \
the content. Your ONLY job is to fix recognition errors.

ASR errors are PHONETIC — the wrong word sounds similar to the correct one \
when spoken aloud. You must mentally "sound out" each suspicious word to \
figure out what was actually said.

Error types (with examples):
1. PHONETIC SUBSTITUTION: "плевать" misrecognized as "поливать", \
"пятьдесят" as "пусть от", "у меня" as "мне".
2. WORD SPLITTING: compound words broken apart — \
"киномарафон" misrecognized as "киномейрах марафон".
3. WORD MERGING: separate words merged into one nonsense word.
4. SEMANTIC NONSENSE: a chain of errors that, when sounded out, reveals a \
well-known phrase or title. Example: "пусть от оттенков серого" is actually \
"пятьдесят оттенков серого" (Fifty Shades of Grey).

Step-by-step approach:
1. Read the FULL text to understand overall topic and intent.
2. Identify words/phrases that don't make sense in context.
3. For each suspicious fragment, sound it out phonetically and find the \
most likely intended word or phrase.
4. Verify that your correction fits the surrounding context.

Rules:
- Fix ONLY recognition errors. Preserve everything else exactly.
- Preserve ALL profanity, slang, filler words — those are intentional.
- NEVER censor, refuse, or comment on content.
- Output ONLY the corrected text, nothing else.\
"""

_CLOUD_MODELS: dict[NormalizationQuality, str] = {
    NormalizationQuality.FAST: "gpt-4o-mini",
    NormalizationQuality.BALANCED: "gpt-4o",
    NormalizationQuality.ACCURATE: "gpt-4o",
}

_LOCAL_MODELS: dict[NormalizationQuality, str] = {
    NormalizationQuality.FAST: "qwen2.5:3b",
    NormalizationQuality.BALANCED: "qwen2.5:7b",
    NormalizationQuality.ACCURATE: "qwen2.5:32b",
}

_FEW_SHOT_EXAMPLES: list[dict[str, str]] = [
    {
        "role": "user",
        "content": "<asr_output>\nну я вот сходил на кин а марафон пусть от "
                   "оттенков серого и мне ваще поливать на эти фильмы\n</asr_output>",
    },
    {
        "role": "assistant",
        "content": "ну я вот сходил на киномарафон пятьдесят оттенков серого "
                   "и мне вообще плевать на эти фильмы",
    },
    {
        "role": "user",
        "content": "<asr_output>\nон придёт из этих чи и сидит в телефон а "
                   "потом гаварит что ему не когда\n</asr_output>",
    },
    {
        "role": "assistant",
        "content": "он придёт из этих чи и сидит в телефоне а "
                   "потом говорит что ему некогда",
    },
    {
        "role": "user",
        "content": "<asr_output>\nя вчера смотрел как они там блять пол "
                   "передачи обсу ждали эту хуйню\n</asr_output>",
    },
    {
        "role": "assistant",
        "content": "я вчера смотрел как они там блять пол "
                   "передачи обсуждали эту хуйню",
    },
]


class LLMNormalizer(Normalizer):
    """Normalizer that uses an OpenAI-compatible LLM API.

    Works with cloud providers (OpenAI, Anthropic via proxy, etc.) and
    local servers (Ollama at ``http://localhost:11434/v1``).

    Parameters:
        api_key:   API key (default ``"ollama"`` for local servers).
        base_url:  API base URL.  When it points to localhost the
                   normalizer selects local-friendly default models.
        model:     Explicit model name.  Overrides the quality preset.
        quality:   :class:`NormalizationQuality` preset that selects the
                   default model and prompt detail level.
        language:  Hint appended to the system prompt (e.g. ``"ru"``).
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        quality: NormalizationQuality = NormalizationQuality.BALANCED,
        language: str | None = None,
    ) -> None:
        self._is_local = _is_local_url(base_url)

        if api_key is None:
            api_key = "ollama" if self._is_local else None

        self._client = OpenAI(api_key=api_key, base_url=base_url)

        if model is not None:
            self._model = model
        else:
            table = _LOCAL_MODELS if self._is_local else _CLOUD_MODELS
            self._model = table[quality]

        if quality == NormalizationQuality.ACCURATE:
            self._system_prompt = _SYSTEM_PROMPT_ACCURATE
        else:
            self._system_prompt = _SYSTEM_PROMPT

        self._quality = quality

        if language:
            self._system_prompt += (
                f"\nThe text is in language: {language}. "
                f"All phonetic reasoning must account for how words "
                f"sound in this language."
            )

        _log.debug(
            "LLMNormalizer: model=%s, local=%s, quality=%s",
            self._model, self._is_local, quality.value,
        )

    def normalize(self, text: str) -> str:
        if not text or not text.strip():
            return text
        try:
            temperature = 0.1 if self._quality == NormalizationQuality.ACCURATE else 0.2
            user_msg = f"<asr_output>\n{text}\n</asr_output>"
            messages: list[dict[str, str]] = [
                {"role": "system", "content": self._system_prompt},
                *_FEW_SHOT_EXAMPLES,
                {"role": "user", "content": user_msg},
            ]
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=temperature,
            )
            result = response.choices[0].message.content
            if not result:
                return text
            result = result.strip()
            if result.startswith("<") and ">" in result:
                result = result.split(">", 1)[-1]
                if result.endswith(">"):
                    result = result.rsplit("<", 1)[0]
                result = result.strip()
            return result if result else text
        except Exception:
            _log.exception("LLM normalization failed, returning raw text")
            return text


def _is_local_url(url: str | None) -> bool:
    if url is None:
        return False
    lower = url.lower()
    return "localhost" in lower or "127.0.0.1" in lower or "[::1]" in lower
