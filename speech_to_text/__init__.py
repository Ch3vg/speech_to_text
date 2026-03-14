"""Continuous speech-to-text library with pluggable engines."""

from .audio import AudioSource, BytesSource, FileSource, MicrophoneSource, list_devices
from .core import SpeechToText
from .engines import register_engine
from .engines.base import CloudSTTEngine, STTEngine
from .models import ResultType, TranscriptionResult

__all__ = [
    "SpeechToText",
    "TranscriptionResult",
    "ResultType",
    "STTEngine",
    "CloudSTTEngine",
    "AudioSource",
    "MicrophoneSource",
    "FileSource",
    "BytesSource",
    "list_devices",
    "register_engine",
]
