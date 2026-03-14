from __future__ import annotations

import threading
from collections.abc import Callable

from deepgram import DeepgramClient
from deepgram.core.events import EventType
from deepgram.listen.v1.types.listen_v1results import ListenV1Results

from ..models import TranscriptionResult
from .base import CloudSTTEngine


class DeepgramEngine(CloudSTTEngine):
    """Streaming STT engine using the Deepgram cloud API (WebSocket).

    Offers very low latency (~0.3-1 s) and high accuracy via the Nova model
    family.  Requires a Deepgram API key (free $200 credit with sign-up).

    Config keys (in addition to :class:`CloudSTTEngine` keys):
        model:        Deepgram model name (default ``nova-3``).
        smart_format: enable smart formatting / punctuation (default True).
    """

    def __init__(
        self,
        on_result: Callable[[TranscriptionResult], None],
        **config: object,
    ) -> None:
        super().__init__(on_result, **config)

        self._model_name: str = config.get("model", "nova-3")  # type: ignore[assignment]
        self._smart_format: bool = config.get("smart_format", True)  # type: ignore[assignment]

        self._client = DeepgramClient(api_key=self._api_key)
        self._connection = None
        self._ctx = None
        self._listener_thread: threading.Thread | None = None
        self._running = False

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        self._ctx = self._client.listen.v1.connect(
            model=self._model_name,
            language=self._language,
            encoding=self._encoding,
            sample_rate=str(self._sample_rate),
            channels="1",
            interim_results="true",
            punctuate="true",
            smart_format=str(self._smart_format).lower(),
        )
        self._connection = self._ctx.__enter__()
        self._connection.on(EventType.MESSAGE, self._on_message)
        self._listener_thread = threading.Thread(
            target=self._connection.start_listening, daemon=True
        )
        self._listener_thread.start()
        self._running = True

    def feed_audio(self, chunk: bytes) -> None:
        if self._running and self._connection is not None:
            self._connection.send_media(chunk)

    def stop(self) -> None:
        self._running = False
        if self._connection is not None:
            try:
                self._connection.send_finalize()
                self._connection.send_close_stream()
            except Exception:
                pass
        if self._ctx is not None:
            try:
                self._ctx.__exit__(None, None, None)
            except Exception:
                pass
            self._ctx = None
            self._connection = None
        if self._listener_thread is not None:
            self._listener_thread.join(timeout=5)
            self._listener_thread = None

    # -- Deepgram event handler --------------------------------------------

    def _on_message(self, message: object) -> None:
        if not isinstance(message, ListenV1Results):
            return
        try:
            alternative = message.channel.alternatives[0]
            transcript: str = alternative.transcript
            is_final = bool(message.is_final)
            confidence = alternative.confidence

            self._emit(transcript, is_final=is_final, confidence=confidence)
        except (AttributeError, IndexError, TypeError):
            pass
