"""Transcribe an audio file instead of the microphone."""

import sys

from speech_to_text import Engine, FileSource, SpeechToText

if len(sys.argv) < 2:
    print("Использование: python file_source_example.py <путь_к_аудиофайлу>")
    print()
    print("Поддерживаемые форматы: WAV, OGG, FLAC, AIFF, MP3 и др.")
    raise SystemExit(1)

path = sys.argv[1]

print(f"Транскрипция: {path}\n")

# realtime=False — обработка на максимальной скорости
for result in SpeechToText(Engine.VOSK, source=FileSource(path, realtime=False)):
    if result.type.value == "final":
        print(f">>> {result.text}")

print("\nГотово.")
