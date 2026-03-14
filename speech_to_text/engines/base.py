from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

from ..models import ResultType, TranscriptionResult


class STTEngine(ABC):
    """Base class for all speech-to-text engines.

    Each engine receives an ``on_result`` callback and invokes it whenever a
    transcription result (partial or final) becomes available.  This unifies
    synchronous engines (Vosk) with asynchronous ones (Deepgram WebSocket).

    Subclass this directly for **local** engines.  For cloud-based engines
    prefer :class:`CloudSTTEngine` which adds common boilerplate (API key,
    URL, ``_emit`` helper).
    """

    def __init__(self, on_result: Callable[[TranscriptionResult], None], **config: object) -> None:
        self._on_result = on_result
        self._config = config

    @abstractmethod
    def start(self) -> None:
        """Prepare the engine (load models, open connections, etc.)."""

    @abstractmethod
    def feed_audio(self, chunk: bytes) -> None:
        """Feed a raw PCM audio chunk (16 kHz, mono, int16 LE)."""

    @abstractmethod
    def stop(self) -> None:
        """Flush pending results and release resources."""


class CloudSTTEngine(STTEngine, ABC):
    """Convenience base for cloud / remote STT engines.

    Handles the most common configuration shared by cloud providers and
    exposes a one-liner :meth:`_emit` helper so subclasses don't need to
    construct :class:`TranscriptionResult` manually.

    Pre-parsed attributes available in subclasses:

    * ``self._api_key``     — API key (validated as non-empty).
    * ``self._base_url``    — Optional custom endpoint URL.
    * ``self._language``    — ISO language code (default ``ru``).
    * ``self._sample_rate`` — Audio sample rate (default 16 000).
    * ``self._encoding``    — Audio encoding string (default ``linear16``).

    Minimal custom cloud engine example::

        from speech_to_text import register_engine
        from speech_to_text.engines import CloudSTTEngine

        class MyCloudEngine(CloudSTTEngine):
            def start(self):
                self._ws = my_connect(self._base_url, key=self._api_key)

            def feed_audio(self, chunk: bytes):
                resp = self._ws.send_and_recv(chunk)
                self._emit(resp["text"], is_final=resp["done"])

            def stop(self):
                self._ws.close()

        register_engine("my_cloud", MyCloudEngine)
    """

    def __init__(
        self,
        on_result: Callable[[TranscriptionResult], None],
        **config: object,
    ) -> None:
        super().__init__(on_result, **config)

        self._api_key: str = config.get("api_key", "")  # type: ignore[assignment]
        if not self._api_key:
            raise ValueError(
                f"{type(self).__name__} requires a non-empty 'api_key' parameter."
            )

        self._base_url: str | None = config.get("base_url")  # type: ignore[assignment]
        self._language: str = config.get("language", "ru")  # type: ignore[assignment]
        self._sample_rate: int = config.get("sample_rate", 16_000)  # type: ignore[assignment]
        self._encoding: str = config.get("encoding", "linear16")  # type: ignore[assignment]

    def _emit(
        self,
        text: str,
        *,
        is_final: bool = True,
        confidence: float | None = None,
        language: str | None = None,
    ) -> None:
        """Create a :class:`TranscriptionResult` and pass it to the callback.

        Empty/whitespace-only *text* is silently ignored.
        """
        text = text.strip()
        if not text:
            return
        self._on_result(
            TranscriptionResult(
                text=text,
                type=ResultType.FINAL if is_final else ResultType.PARTIAL,
                language=language or self._language,
                confidence=confidence,
            )
        )
