"""Minimal iterator example — continuous speech recognition with Vosk."""

from speech_to_text import SpeechToText

print("Говорите... (Ctrl+C для остановки)\n")

try:
    for result in SpeechToText("vosk", language="ru"):
        prefix = ">>>" if result.type.value == "final" else "..."
        print(f"{prefix} {result.text}")
except KeyboardInterrupt:
    print("\nОстановлено.")
