"""Example: registering and using a custom cloud STT engine.

Demonstrates how to create your own engine by subclassing CloudSTTEngine
(for cloud APIs) or STTEngine (for local engines) and plugging it into
the SpeechToText pipeline.
"""

from speech_to_text import (
    CloudSTTEngine,
    ResultType,
    SpeechToText,
    STTEngine,
    TranscriptionResult,
    register_engine,
)


# ---------------------------------------------------------------------------
# Example 1 — Custom cloud engine (inherits API key handling + _emit helper)
# ---------------------------------------------------------------------------
class MyCloudEngine(CloudSTTEngine):
    """Minimal cloud engine skeleton.

    Replace the bodies of start / feed_audio / stop with your real
    cloud-provider logic (WebSocket, REST, gRPC, etc.).
    """

    def start(self) -> None:
        print(f"[MyCloudEngine] connecting to {self._base_url} ...")
        # self._ws = websocket.connect(self._base_url, api_key=self._api_key)

    def feed_audio(self, chunk: bytes) -> None:
        # self._ws.send(chunk)
        # response = self._ws.recv()
        # self._emit(response["text"], is_final=response["is_final"])
        pass

    def stop(self) -> None:
        print("[MyCloudEngine] disconnected")
        # self._ws.close()


# ---------------------------------------------------------------------------
# Example 2 — Custom local engine (full control, no cloud boilerplate)
# ---------------------------------------------------------------------------
class EchoEngine(STTEngine):
    """Trivial engine that 'transcribes' audio chunk sizes (for testing)."""

    def start(self) -> None:
        self._counter = 0

    def feed_audio(self, chunk: bytes) -> None:
        self._counter += 1
        if self._counter % 20 == 0:
            self._on_result(
                TranscriptionResult(
                    text=f"[echo] received {self._counter} chunks, last {len(chunk)} bytes",
                    type=ResultType.FINAL,
                )
            )

    def stop(self) -> None:
        self._on_result(
            TranscriptionResult(
                text=f"[echo] total chunks: {self._counter}",
                type=ResultType.FINAL,
            )
        )


# Register under custom names — now usable via SpeechToText("my_cloud", ...)
register_engine("my_cloud", MyCloudEngine)
register_engine("echo", EchoEngine)

# --- quick demo with the echo engine (no real STT, just proves the wiring) -
if __name__ == "__main__":
    print("Echo engine demo — prints a message every ~5 seconds.\n"
          "Press Ctrl+C to stop.\n")
    try:
        for result in SpeechToText("echo"):
            print(f">>> {result.text}")
    except KeyboardInterrupt:
        print("\nОстановлено.")
