"""Transcribe an audio file instead of the microphone."""

import sys

from speech_to_text import FileSource, SpeechToText

if len(sys.argv) < 2:
    print("Использование: python file_source_example.py <путь_к_файлу.wav>")
    print()
    print("Поддерживаемые форматы:")
    print("  .wav — всегда (встроенная поддержка)")
    print("  .mp3, .flac, .ogg, ... — нужен pip install soundfile")
    raise SystemExit(1)

path = sys.argv[1]

print(f"Транскрипция: {path}\n")

# realtime=False — обработка на максимальной скорости
for result in SpeechToText("vosk", source=FileSource(path, realtime=False)):
    if result.type.value == "final":
        print(f">>> {result.text}")

print("\nГотово.")
