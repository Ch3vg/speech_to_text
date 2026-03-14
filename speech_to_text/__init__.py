"""Continuous speech-to-text library with pluggable engines."""

from .audio import AudioSource, BytesSource, FileSource, MicrophoneSource, list_devices
from .core import SpeechToText
from .engines import register_engine
from .engines.base import CloudSTTEngine, STTEngine
from .models import Device, Engine, NormalizationQuality, ResultType, TranscriptionResult
from .normalizers import register_normalizer
from .normalizers.base import Normalizer

__all__ = [
    "SpeechToText",
    "Engine",
    "Device",
    "TranscriptionResult",
    "ResultType",
    "NormalizationQuality",
    "Normalizer",
    "STTEngine",
    "CloudSTTEngine",
    "AudioSource",
    "MicrophoneSource",
    "FileSource",
    "BytesSource",
    "list_devices",
    "register_engine",
    "register_normalizer",
]
