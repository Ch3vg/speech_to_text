from __future__ import annotations

from collections.abc import Callable

from deepgram import DeepgramClient, LiveOptions, LiveTranscriptionEvents

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

        self._client = DeepgramClient(self._api_key)
        self._connection = None
        self._running = False

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        self._connection = self._client.listen.live.v("1")
        self._connection.on(LiveTranscriptionEvents.Transcript, self._on_transcript)

        options = LiveOptions(
            model=self._model_name,
            language=self._language,
            encoding=self._encoding,
            sample_rate=self._sample_rate,
            channels=1,
            interim_results=True,
            punctuate=True,
            smart_format=self._smart_format,
        )
        self._connection.start(options)
        self._running = True

    def feed_audio(self, chunk: bytes) -> None:
        if self._running and self._connection is not None:
            self._connection.send(chunk)

    def stop(self) -> None:
        self._running = False
        if self._connection is not None:
            try:
                self._connection.finish()
            except Exception:
                pass
            self._connection = None

    # -- Deepgram event handler --------------------------------------------

    def _on_transcript(self, _client: object, result: object, **kwargs: object) -> None:
        try:
            channel = result.channel  # type: ignore[attr-defined]
            alternative = channel.alternatives[0]
            transcript: str = alternative.transcript

            self._emit(
                transcript,
                is_final=result.is_final,  # type: ignore[attr-defined]
                confidence=alternative.confidence,
            )
        except (AttributeError, IndexError, TypeError):
            pass
