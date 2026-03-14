from __future__ import annotations

import json
from collections.abc import Callable

import vosk

from ..models import ResultType, TranscriptionResult
from .base import STTEngine


class VoskEngine(STTEngine):
    """Streaming STT engine based on Vosk (Kaldi).

    Provides the lowest latency (~0.3-0.5 s) among local engines.
    Works on CPU with minimal resources.

    Config keys:
        model_path: explicit local path to an unpacked Vosk model directory.
        model_name: Vosk model identifier for auto-download
                    (default ``vosk-model-small-ru-0.22``).
        language:   ISO language code used when neither *model_path* nor
                    *model_name* is supplied (default ``ru``).
        sample_rate: audio sample rate (default 16000).
    """

    def __init__(
        self,
        on_result: Callable[[TranscriptionResult], None],
        **config: object,
    ) -> None:
        super().__init__(on_result, **config)

        model_path: str | None = config.get("model_path")  # type: ignore[assignment]
        model_name: str | None = config.get("model_name")  # type: ignore[assignment]
        language: str = config.get("language", "ru")  # type: ignore[assignment]
        sample_rate: int = config.get("sample_rate", 16_000)  # type: ignore[assignment]

        vosk.SetLogLevel(-1)

        if model_path:
            self._model = vosk.Model(model_path=model_path)
        elif model_name:
            self._model = vosk.Model(model_name=model_name)
        else:
            self._model = vosk.Model(lang=language)

        self._recognizer = vosk.KaldiRecognizer(self._model, sample_rate)
        self._recognizer.SetWords(True)

    def start(self) -> None:
        pass

    def feed_audio(self, chunk: bytes) -> None:
        if self._recognizer.AcceptWaveform(chunk):
            data = json.loads(self._recognizer.Result())
            text = data.get("text", "").strip()
            if text:
                self._on_result(
                    TranscriptionResult(text=text, type=ResultType.FINAL)
                )
        else:
            data = json.loads(self._recognizer.PartialResult())
            text = data.get("partial", "").strip()
            if text:
                self._on_result(
                    TranscriptionResult(text=text, type=ResultType.PARTIAL)
                )

    def stop(self) -> None:
        data = json.loads(self._recognizer.FinalResult())
        text = data.get("text", "").strip()
        if text:
            self._on_result(
                TranscriptionResult(text=text, type=ResultType.FINAL)
            )
