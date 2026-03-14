from __future__ import annotations

import logging
import os
import queue
import sys
import threading
from collections.abc import Callable
from pathlib import Path

import numpy as np
from faster_whisper import WhisperModel

from ..models import Device, ResultType, TranscriptionResult
from .base import STTEngine

_log = logging.getLogger(__name__)
_nvidia_dlls_registered = False


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
        device: str = config.get("compute_device", Device.AUTO)  # type: ignore[assignment]
        compute_type: str = config.get("compute_type", "default")  # type: ignore[assignment]

        self._language: str = config.get("language", "ru")  # type: ignore[assignment]
        self._sample_rate: int = config.get("sample_rate", 16_000)  # type: ignore[assignment]
        self._energy_threshold: float = config.get("energy_threshold", 300)  # type: ignore[assignment]
        self._silence_duration: float = config.get("silence_duration", 0.8)  # type: ignore[assignment]
        self._partial_interval: float = config.get("partial_interval", 2.0)  # type: ignore[assignment]

        self._model = _load_model(model_size, device, compute_type)

        self._audio_buffer = bytearray()
        self._is_speaking = False
        self._silence_samples: int = 0
        self._silence_threshold_samples = int(self._silence_duration * self._sample_rate)
        self._samples_since_partial: int = 0
        self._partial_threshold_samples = int(self._partial_interval * self._sample_rate)
        self._lock = threading.Lock()

        self._transcribe_queue: queue.Queue[tuple[bytes, ResultType] | None] = queue.Queue()
        self._worker: threading.Thread | None = None
        self._running = False

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        self._running = True
        self._audio_buffer.clear()
        self._is_speaking = False
        self._silence_samples = 0
        self._samples_since_partial = 0
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
        n_samples = len(audio)
        energy = float(np.sqrt(np.mean(audio.astype(np.float32) ** 2)))

        with self._lock:
            if energy > self._energy_threshold:
                self._is_speaking = True
                self._silence_samples = 0
                self._audio_buffer.extend(chunk)
                self._samples_since_partial += n_samples

                if (
                    self._samples_since_partial >= self._partial_threshold_samples
                    and len(self._audio_buffer) > self._sample_rate * 2
                ):
                    self._samples_since_partial = 0
                    self._transcribe_queue.put(
                        (bytes(self._audio_buffer), ResultType.PARTIAL)
                    )

            elif self._is_speaking:
                self._audio_buffer.extend(chunk)
                self._silence_samples += n_samples
                if self._silence_samples >= self._silence_threshold_samples:
                    self._transcribe_queue.put(
                        (bytes(self._audio_buffer), ResultType.FINAL)
                    )
                    self._audio_buffer.clear()
                    self._is_speaking = False
                    self._silence_samples = 0
                    self._samples_since_partial = 0

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
            try:
                self._transcribe(audio_bytes, result_type)
            except Exception:
                _log.exception("Transcription failed")

    def _transcribe(self, audio_bytes: bytes, result_type: ResultType) -> None:
        audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        try:
            segments, info = self._model.transcribe(
                audio,
                language=self._language,
                beam_size=5,
                vad_filter=True,
            )
            text = " ".join(seg.text.strip() for seg in segments).strip()
        except RuntimeError as exc:
            if _is_cuda_error(exc):
                raise RuntimeError(_CUDA_ERROR_MSG) from exc
            raise
        if text:
            self._on_result(
                TranscriptionResult(
                    text=text,
                    type=result_type,
                    language=info.language,
                    confidence=info.language_probability,
                )
            )


_CUDA_ERROR_MSG = (
    "CUDA libraries not found. Install them with:\n"
    "  pip install speech-to-text[whisper-gpu]\n"
    "Or force CPU mode:\n"
    "  SpeechToText(Engine.WHISPER, compute_device=Device.CPU)"
)


def _is_cuda_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "cublas" in msg or "cudnn" in msg or "cuda" in msg


def _register_nvidia_dll_dirs() -> None:
    """On Windows, add pip-installed NVIDIA DLL dirs to search path.

    ``nvidia-cublas-cu12`` and similar packages place DLLs under
    ``site-packages/nvidia/<lib>/bin/``.  ctranslate2 loads them via
    ``LoadLibrary`` which checks ``PATH`` — so we prepend those dirs.
    ``os.add_dll_directory`` is also called as a belt-and-suspenders
    measure for libraries that use the newer Windows API.
    """
    global _nvidia_dlls_registered
    if _nvidia_dlls_registered or sys.platform != "win32":
        return
    _nvidia_dlls_registered = True

    added: list[str] = []
    for site_dir in _site_packages_dirs():
        nvidia_dir = Path(site_dir) / "nvidia"
        if not nvidia_dir.is_dir():
            continue
        for bin_dir in nvidia_dir.glob("*/bin"):
            if bin_dir.is_dir():
                dir_str = str(bin_dir)
                os.add_dll_directory(dir_str)
                added.append(dir_str)

    if added:
        os.environ["PATH"] = os.pathsep.join(added) + os.pathsep + os.environ.get("PATH", "")
        _log.debug("Registered %d NVIDIA DLL dirs: %s", len(added), added)


def _site_packages_dirs() -> list[str]:
    """Return site-packages directories from sys.path."""
    return [p for p in sys.path if p and Path(p).name == "site-packages" and Path(p).is_dir()]


def _load_model(
    model_size: str, device: str, compute_type: str
) -> WhisperModel:
    """Load WhisperModel with automatic CUDA -> CPU fallback for device='auto'."""
    _register_nvidia_dll_dirs()
    try:
        return WhisperModel(model_size, device=device, compute_type=compute_type)
    except RuntimeError as exc:
        if not _is_cuda_error(exc):
            raise
        if device != Device.AUTO:
            raise RuntimeError(_CUDA_ERROR_MSG) from exc
        _log.warning(
            "CUDA unavailable (%s), falling back to CPU. "
            "For GPU support: pip install speech-to-text[whisper-gpu]",
            exc,
        )
        return WhisperModel(model_size, device="cpu", compute_type="default")
