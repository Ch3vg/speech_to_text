"""Whisper engine example — high-accuracy local recognition."""

from speech_to_text import Engine, SpeechToText

print("Загрузка модели Whisper (может занять время)...")
print("Говорите... (Ctrl+C для остановки)\n")

try:
    for result in SpeechToText(
        Engine.WHISPER,
        model="small",
        language="ru",
        partial_results=True,
    ):
        prefix = ">>>" if result.type.value == "final" else "..."
        print(f"{prefix} {result.text}")
except KeyboardInterrupt:
    print("\nОстановлено.")
