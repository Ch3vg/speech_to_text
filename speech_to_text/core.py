from __future__ import annotations

import queue
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .audio import AudioSource, BytesSource, FileSource, MicrophoneSource
from .engines import create_engine
from .models import Engine, ResultType, TranscriptionResult

_SENTINEL = object()


class SpeechToText:
    """High-level speech-to-text interface with pluggable engines.

    Supports two consumption styles:

    **Iterator** (blocks the calling thread)::

        for result in SpeechToText(Engine.VOSK):
            print(result.text)

    **Callback** (non-blocking)::

        stt = SpeechToText(Engine.WHISPER, model="small", language="ru")
        stt.on_result(lambda r: print(r.text))
        stt.start()

    **Audio source** defaults to the system microphone but can be set to a
    file or any custom :class:`AudioSource`::

        for result in SpeechToText(Engine.VOSK, source="recording.wav"):
            print(result.text)
    """

    def __init__(
        self,
        engine: Engine | str,
        *,
        source: AudioSource | str | Path | bytes | bytearray | None = None,
        partial_results: bool = True,
        device: int | str | None = None,
        sample_rate: int = 16_000,
        block_size: int = 4000,
        **engine_config: Any,
    ) -> None:
        self._partial_results = partial_results
        self._result_queue: queue.Queue[TranscriptionResult | object] = queue.Queue()
        self._callbacks: list[Callable[[TranscriptionResult], None]] = []
        self._running = False
        self._lock = threading.Lock()

        self._engine = create_engine(
            engine, on_result=self._handle_result, sample_rate=sample_rate, **engine_config
        )

        if isinstance(source, AudioSource):
            self._audio: AudioSource = source
        elif isinstance(source, (bytes, bytearray)):
            self._audio = BytesSource(
                source, sample_rate=sample_rate, block_size=block_size
            )
        elif isinstance(source, (str, Path)):
            self._audio = FileSource(
                source, sample_rate=sample_rate, block_size=block_size
            )
        else:
            self._audio = MicrophoneSource(
                sample_rate=sample_rate, block_size=block_size, device=device
            )

        self._audio.on_audio = self._engine.feed_audio
        self._audio.on_finished = self._on_source_finished

    # -- result handling ---------------------------------------------------

    def _handle_result(self, result: TranscriptionResult) -> None:
        if not self._partial_results and result.type == ResultType.PARTIAL:
            return
        for cb in self._callbacks:
            cb(result)
        self._result_queue.put(result)

    def on_result(self, callback: Callable[[TranscriptionResult], None]) -> None:
        """Register a callback invoked for every transcription result."""
        self._callbacks.append(callback)

    # -- convenience -------------------------------------------------------

    def transcribe(self) -> str:
        """Run the full pipeline and return the complete transcription as a single string.

        Collects all FINAL results, joins them, and returns when the source
        is exhausted.  Most useful with file / bytes sources::

            text = SpeechToText(Engine.WHISPER, source="recording.ogg").transcribe()
        """
        parts: list[str] = []
        for result in self:
            if result.type == ResultType.FINAL and result.text:
                parts.append(result.text)
        return " ".join(parts)

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        """Start capturing audio and recognising speech."""
        with self._lock:
            if self._running:
                return
            self._running = True
        self._engine.start()
        self._audio.start()

    def stop(self) -> None:
        """Stop capturing and flush remaining results."""
        with self._lock:
            if not self._running:
                return
            self._running = False
        self._audio.stop()
        self._engine.stop()
        self._result_queue.put(_SENTINEL)

    @property
    def is_running(self) -> bool:
        return self._running

    def _on_source_finished(self) -> None:
        """Called (from the source thread) when a finite source is exhausted."""
        threading.Thread(target=self.stop, daemon=True).start()

    # -- iterator protocol -------------------------------------------------

    def __iter__(self) -> SpeechToText:
        if not self._running:
            self.start()
        return self

    def __next__(self) -> TranscriptionResult:
        while True:
            try:
                item = self._result_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            if item is _SENTINEL:
                raise StopIteration
            return item  # type: ignore[return-value]

    # -- context manager ---------------------------------------------------

    def __enter__(self) -> SpeechToText:
        self.start()
        return self

    def __exit__(self, *exc: object) -> bool:
        self.stop()
        return False

    def __del__(self) -> None:
        if self._running:
            self.stop()
