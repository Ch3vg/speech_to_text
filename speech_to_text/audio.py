from __future__ import annotations

import io
import queue
import threading
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

SAMPLE_RATE = 16_000
CHANNELS = 1
DTYPE = "int16"
BLOCK_SIZE = 4000  # 250 ms at 16 kHz


# ---------------------------------------------------------------------------
# Abstract source
# ---------------------------------------------------------------------------

class AudioSource(ABC):
    """Base class for audio sources.

    Before :meth:`start` is called, the caller (normally :class:`SpeechToText`)
    sets two attributes:

    * ``on_audio``    — callback that receives raw PCM chunks (``bytes``).
    * ``on_finished`` — optional callback invoked when the source has no more
      data (e.g. end of file).  Microphone sources never call it.
    """

    def __init__(self) -> None:
        self.on_audio: Callable[[bytes], None] = lambda _chunk: None
        self.on_finished: Callable[[], None] | None = None

    @abstractmethod
    def start(self) -> None:
        """Begin producing audio."""

    @abstractmethod
    def stop(self) -> None:
        """Stop producing audio and release resources."""

    @property
    @abstractmethod
    def is_running(self) -> bool: ...


# ---------------------------------------------------------------------------
# Microphone
# ---------------------------------------------------------------------------

class MicrophoneSource(AudioSource):
    """Captures audio from a system microphone.

    The ``sounddevice`` callback only enqueues raw bytes (fast, lock-free).
    A dedicated processing thread dequeues and delivers them via
    :attr:`on_audio`, so the engine's ``feed_audio`` never blocks the audio
    driver.

    Parameters:
        sample_rate: Sample rate in Hz (default 16 000).
        block_size:  Samples per chunk (default 4 000 = 250 ms).
        device:      Device index (``int``) or substring of the device name
                     (``str``).  *None* means the system default input device.
                     Use :func:`list_devices` to discover available devices.
    """

    def __init__(
        self,
        *,
        sample_rate: int = SAMPLE_RATE,
        block_size: int = BLOCK_SIZE,
        device: int | str | None = None,
    ) -> None:
        super().__init__()
        self._sample_rate = sample_rate
        self._block_size = block_size
        self._device = device

        self._queue: queue.Queue[bytes | None] = queue.Queue()
        self._stream: sd.InputStream | None = None
        self._thread: threading.Thread | None = None
        self._running = False

    def start(self) -> None:
        self._running = True
        self._stream = sd.InputStream(
            samplerate=self._sample_rate,
            channels=CHANNELS,
            dtype=DTYPE,
            blocksize=self._block_size,
            device=self._device,
            callback=self._sd_callback,
        )
        self._thread = threading.Thread(target=self._process_loop, daemon=True)
        self._stream.start()
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        self._queue.put(None)
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=2)
        self._thread = None

    @property
    def is_running(self) -> bool:
        return self._running

    def _sd_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: object,
        status: sd.CallbackFlags,
    ) -> None:
        self._queue.put_nowait(bytes(indata))

    def _process_loop(self) -> None:
        while self._running:
            try:
                chunk = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if chunk is None:
                break
            self.on_audio(chunk)


# ---------------------------------------------------------------------------
# Bytes
# ---------------------------------------------------------------------------

class BytesSource(AudioSource):
    """Feeds pre-loaded audio bytes.

    Accepts ``bytes``, ``bytearray``, or ``memoryview``.  By default the data
    is assumed to already be in the engine format (16 kHz, mono, int16 LE).
    Set *raw=False* if the data is an encoded audio byte string (WAV, OGG,
    FLAC, etc.) — it will be decoded and converted automatically.

    Parameters:
        data:        Raw PCM bytes or encoded audio bytes.
        raw:         If *True* (default), *data* is raw PCM.  If *False*,
                     *data* is decoded via ``soundfile`` (WAV, OGG, FLAC, …).
        sample_rate: Target sample rate (default 16 000).
        block_size:  Samples per chunk (default 4 000 = 250 ms).
        realtime:    If *True*, chunks are fed at real-time pace.
                     *False* (default) = as fast as possible.
    """

    def __init__(
        self,
        data: bytes | bytearray | memoryview,
        *,
        raw: bool = True,
        sample_rate: int = SAMPLE_RATE,
        block_size: int = BLOCK_SIZE,
        realtime: bool = False,
    ) -> None:
        super().__init__()
        self._sample_rate = sample_rate
        self._block_size = block_size
        self._realtime = realtime

        if raw:
            self._pcm = bytes(data)
        else:
            self._pcm = _decode_audio_bytes(data, sample_rate)

        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._running = False

    def start(self) -> None:
        self._running = True
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._feed_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        self._stop_event.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=10)
        self._thread = None

    @property
    def is_running(self) -> bool:
        return self._running

    def _feed_loop(self) -> None:
        chunk_bytes = self._block_size * 2
        chunk_duration = self._block_size / self._sample_rate

        offset = 0
        while offset < len(self._pcm) and not self._stop_event.is_set():
            self.on_audio(self._pcm[offset : offset + chunk_bytes])
            offset += chunk_bytes
            if self._realtime:
                self._stop_event.wait(timeout=chunk_duration)

        self._running = False
        if self.on_finished is not None:
            self.on_finished()


# ---------------------------------------------------------------------------
# File
# ---------------------------------------------------------------------------

class FileSource(AudioSource):
    """Reads audio from a file and feeds it as PCM chunks.

    Supports WAV, OGG, FLAC, AIFF, and other formats handled by ``libsndfile``
    (via the ``soundfile`` package).  Automatically converts sample rate,
    channels, and bit depth to the engine format (16 kHz, mono, int16).

    Parameters:
        path:        Path to the audio file.
        sample_rate: Target sample rate (default 16 000).
        block_size:  Samples per chunk (default 4 000 = 250 ms).
        realtime:    If *True* (default), chunks are fed at real-time pace.
                     Set to *False* to process as fast as possible.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        sample_rate: int = SAMPLE_RATE,
        block_size: int = BLOCK_SIZE,
        realtime: bool = True,
    ) -> None:
        super().__init__()
        self._path = Path(path)
        if not self._path.is_file():
            raise FileNotFoundError(f"Audio file not found: {self._path}")

        self._sample_rate = sample_rate
        self._block_size = block_size
        self._realtime = realtime

        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._running = False

    def start(self) -> None:
        self._running = True
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._feed_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        self._stop_event.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=10)
        self._thread = None

    @property
    def is_running(self) -> bool:
        return self._running

    def _feed_loop(self) -> None:
        pcm = self._load_audio()
        chunk_bytes = self._block_size * 2  # int16 = 2 bytes per sample
        chunk_duration = self._block_size / self._sample_rate

        offset = 0
        while offset < len(pcm) and not self._stop_event.is_set():
            self.on_audio(pcm[offset : offset + chunk_bytes])
            offset += chunk_bytes
            if self._realtime:
                self._stop_event.wait(timeout=chunk_duration)

        self._running = False
        if self.on_finished is not None:
            self.on_finished()

    def _load_audio(self) -> bytes:
        """Load audio file -> 16 kHz mono int16 PCM bytes."""
        audio, sr = sf.read(str(self._path), dtype="float32")
        return _to_pcm(audio, sr, self._sample_rate)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _to_pcm(audio: np.ndarray, sr: int, target_sr: int) -> bytes:
    """Convert a float32 numpy array to 16 kHz mono int16 PCM bytes."""
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        audio = _resample(audio, sr, target_sr)
    return (audio * 32767).astype(np.int16).tobytes()


def _decode_audio_bytes(data: bytes | bytearray | memoryview, target_sr: int) -> bytes:
    """Decode in-memory audio bytes (WAV, OGG, FLAC, …) to PCM."""
    audio, sr = sf.read(io.BytesIO(bytes(data)), dtype="float32")
    return _to_pcm(audio, sr, target_sr)


def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample via linear interpolation (good enough for speech)."""
    if orig_sr == target_sr:
        return audio
    n_samples = int(len(audio) * target_sr / orig_sr)
    indices = np.linspace(0, len(audio) - 1, n_samples)
    return np.interp(indices, np.arange(len(audio)), audio.astype(np.float64)).astype(
        audio.dtype
    )


def list_devices() -> list[dict]:
    """Return a list of available audio **input** devices.

    Each entry is a dict with keys:

    * ``index`` — device index (pass to ``SpeechToText(device=…)``).
    * ``name``  — human-readable device name.
    * ``channels`` — max input channels.
    * ``default_sample_rate`` — native sample rate.
    * ``is_default`` — *True* if this is the system default input device.
    """
    devices = sd.query_devices()
    default_input = sd.default.device[0]
    result = []
    for i, dev in enumerate(devices):
        if dev["max_input_channels"] > 0:
            result.append(
                {
                    "index": i,
                    "name": dev["name"],
                    "channels": dev["max_input_channels"],
                    "default_sample_rate": dev["default_samplerate"],
                    "is_default": i == default_input,
                }
            )
    return result
