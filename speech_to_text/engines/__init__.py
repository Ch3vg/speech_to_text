from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .base import STTEngine
from ..models import TranscriptionResult

_ENGINE_REGISTRY: dict[str, type[STTEngine]] = {}


def register_engine(name: str, engine_class: type[STTEngine]) -> None:
    _ENGINE_REGISTRY[name] = engine_class


def create_engine(
    name: str,
    on_result: Callable[[TranscriptionResult], None],
    **config: Any,
) -> STTEngine:
    if name not in _ENGINE_REGISTRY:
        _try_load_engine(name)
    if name not in _ENGINE_REGISTRY:
        available = ", ".join(sorted(_ENGINE_REGISTRY)) or "(none)"
        raise ValueError(
            f"Unknown engine '{name}'. Available: {available}. "
            f"Install the required extra, e.g.: pip install speech-to-text[{name}]"
        )
    return _ENGINE_REGISTRY[name](on_result=on_result, **config)


def _try_load_engine(name: str) -> None:
    loaders = {
        "vosk": _load_vosk,
        "whisper": _load_whisper,
        "deepgram": _load_deepgram,
    }
    loader = loaders.get(name)
    if loader is not None:
        loader()


def _load_vosk() -> None:
    try:
        import vosk as _vosk  # noqa: F401
    except ImportError:
        raise ImportError(
            "Vosk is not installed. Run: pip install speech-to-text[vosk]"
        ) from None
    from .vosk_engine import VoskEngine
    register_engine("vosk", VoskEngine)


def _load_whisper() -> None:
    try:
        import faster_whisper as _fw  # noqa: F401
    except ImportError:
        raise ImportError(
            "faster-whisper is not installed. Run: pip install speech-to-text[whisper]"
        ) from None
    from .whisper_engine import WhisperEngine
    register_engine("whisper", WhisperEngine)


def _load_deepgram() -> None:
    try:
        import deepgram as _dg  # noqa: F401
    except ImportError:
        raise ImportError(
            "Deepgram SDK is not installed. Run: pip install speech-to-text[deepgram]"
        ) from None
    from .deepgram_engine import DeepgramEngine
    register_engine("deepgram", DeepgramEngine)
