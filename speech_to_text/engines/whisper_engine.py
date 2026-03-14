from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable

import numpy as np
from faster_whisper import WhisperModel

from ..models import ResultType, TranscriptionResult
from .base import STTEngine


class WhisperEngine(STTEngine):
    """Streaming STT engine based on faster-whisper.

    Accumulates audio while speech is detected (energy-based VAD) and
    transcribes the buffer when silence is detected.  Partial results are
    emitted periodically during long utterances.

    Config keys:
        model:            Whisper model size (default ``small``).
        compute_device:   ``"auto"``, ``"cpu"``, or ``"cuda"`` (default ``auto``).
        compute_type:     CTranslate2 compute type (default ``default``).
        language:         ISO language code (default ``ru``).
        sample_rate:      audio sample rate (default 16000).
        energy_threshold: RMS energy threshold to detect speech (default 300).
        silence_duration: seconds of silence to finalise an utterance (default 0.8).
        partial_interval: seconds between partial transcriptions during speech
                          (default 2.0).
    """

    def __init__(
        self,
        on_result: Callable[[TranscriptionResult], None],
        **config: object,
    ) -> None:
        super().__init__(on_result, **config)

        model_size: str = config.get("model", "small")  # type: ignore[assignment]
        device: str = config.get("compute_device", "auto")  # type: ignore[assignment]
        compute_type: str = config.get("compute_type", "default")  # type: ignore[assignment]

        self._language: str = config.get("language", "ru")  # type: ignore[assignment]
        self._sample_rate: int = config.get("sample_rate", 16_000)  # type: ignore[assignment]
        self._energy_threshold: float = config.get("energy_threshold", 300)  # type: ignore[assignment]
        self._silence_duration: float = config.get("silence_duration", 0.8)  # type: ignore[assignment]
        self._partial_interval: float = config.get("partial_interval", 2.0)  # type: ignore[assignment]

        self._model = WhisperModel(model_size, device=device, compute_type=compute_type)

        self._audio_buffer = bytearray()
        self._is_speaking = False
        self._silence_start: float | None = None
        self._last_partial: float = 0.0
        self._lock = threading.Lock()

        self._transcribe_queue: queue.Queue[tuple[bytes, ResultType] | None] = queue.Queue()
        self._worker: threading.Thread | None = None
        self._running = False

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        self._running = True
        self._audio_buffer.clear()
        self._is_speaking = False
        self._silence_start = None
        self._last_partial = 0.0
        self._worker = threading.Thread(target=self._transcription_loop, daemon=True)
        self._worker.start()

    def stop(self) -> None:
        with self._lock:
            if self._audio_buffer:
                self._transcribe_queue.put((bytes(self._audio_buffer), ResultType.FINAL))
                self._audio_buffer.clear()
        self._running = False
        self._transcribe_queue.put(None)
        if self._worker is not None:
            self._worker.join(timeout=30)
            self._worker = None

    # -- audio ingestion ---------------------------------------------------

    def feed_audio(self, chunk: bytes) -> None:
        if not self._running:
            return

        audio = np.frombuffer(chunk, dtype=np.int16)
        energy = float(np.sqrt(np.mean(audio.astype(np.float32) ** 2)))
        now = time.monotonic()

        with self._lock:
            if energy > self._energy_threshold:
                self._is_speaking = True
                self._silence_start = None
                self._audio_buffer.extend(chunk)

                if (
                    now - self._last_partial >= self._partial_interval
                    and len(self._audio_buffer) > self._sample_rate * 2
                ):
                    self._last_partial = now
                    self._transcribe_queue.put(
                        (bytes(self._audio_buffer), ResultType.PARTIAL)
                    )

            elif self._is_speaking:
                self._audio_buffer.extend(chunk)
                if self._silence_start is None:
                    self._silence_start = now
                elif now - self._silence_start >= self._silence_duration:
                    self._transcribe_queue.put(
                        (bytes(self._audio_buffer), ResultType.FINAL)
                    )
                    self._audio_buffer.clear()
                    self._is_speaking = False
                    self._silence_start = None
                    self._last_partial = 0.0

    # -- transcription worker ----------------------------------------------

    def _transcription_loop(self) -> None:
        while True:
            try:
                item = self._transcribe_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if item is None:
                break
            audio_bytes, result_type = item
            self._transcribe(audio_bytes, result_type)

    def _transcribe(self, audio_bytes: bytes, result_type: ResultType) -> None:
        audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        segments, info = self._model.transcribe(
            audio,
            language=self._language,
            beam_size=5,
            vad_filter=True,
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()
        if text:
            self._on_result(
                TranscriptionResult(
                    text=text,
                    type=result_type,
                    language=info.language,
                    confidence=info.language_probability,
                )
            )
