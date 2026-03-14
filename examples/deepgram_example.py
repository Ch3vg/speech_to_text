"""Deepgram cloud engine example — lowest latency via WebSocket streaming."""

import os

from speech_to_text import SpeechToText

API_KEY = os.environ.get("DEEPGRAM_API_KEY", "")
if not API_KEY:
    print("Установите переменную окружения DEEPGRAM_API_KEY.")
    print("Зарегистрируйтесь на https://deepgram.com для получения бесплатных $200 кредитов.")
    raise SystemExit(1)

print("Говорите... (Ctrl+C для остановки)\n")

try:
    for result in SpeechToText(
        "deepgram",
        api_key=API_KEY,
        language="ru",
        partial_results=True,
    ):
        prefix = ">>>" if result.type.value == "final" else "..."
        print(f"{prefix} {result.text}")
except KeyboardInterrupt:
    print("\nОстановлено.")
