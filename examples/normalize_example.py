"""Transcribe a file with optional LLM normalization.

Usage:
    # Cloud (OpenAI) — set OPENAI_API_KEY env var or pass api_key
    python normalize_example.py recording.ogg

    # Local (Ollama at http://localhost:11434)
    python normalize_example.py recording.ogg --local
"""

import sys

from speech_to_text import (
    Engine,
    NormalizationQuality,
    ResultType,
    SpeechToText,
)

if len(sys.argv) < 2:
    print("Использование: python normalize_example.py <файл> [--local]")
    raise SystemExit(1)

path = sys.argv[1]
local = "--local" in sys.argv

if local:
    normalizer_kwargs = dict(
        normalizer="llm",
        normalizer_base_url="http://localhost:11434/v1",
        normalizer_quality=NormalizationQuality.BALANCED,
        normalizer_language="ru",
    )
else:
    normalizer_kwargs = dict(
        normalizer="llm",
        normalizer_quality=NormalizationQuality.BALANCED,
        normalizer_language="ru",
    )

print(f"Транскрипция + нормализация: {path}")
print(f"Режим: {'локальный (Ollama)' if local else 'облако (OpenAI)'}\n")

for result in SpeechToText(
    Engine.WHISPER,
    source=path,
    normalize_scope=ResultType.FINAL,
    **normalizer_kwargs,
):
    if result.type == ResultType.FINAL and result.text:
        print(f"  Текст:     {result.text}")
        if result.raw_text and result.raw_text != result.text:
            print(f"  Оригинал:  {result.raw_text}")
        print()

print("Готово.")
