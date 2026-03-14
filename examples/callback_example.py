"""Callback example — non-blocking speech recognition with result handler."""

from speech_to_text import Engine, ResultType, SpeechToText


def on_text(result):
    if result.type == ResultType.FINAL:
        print(f"\n>>> {result.text}")
    else:
        print(f"    ... {result.text}", end="\r")


stt = SpeechToText(Engine.VOSK, language="ru")
stt.on_result(on_text)
stt.start()

print("Говорите... (Enter для остановки)\n")

try:
    input()
except KeyboardInterrupt:
    pass
finally:
    stt.stop()
    print("Остановлено.")
