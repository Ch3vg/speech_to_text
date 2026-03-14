from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from time import time


class ResultType(Enum):
    PARTIAL = "partial"
    FINAL = "final"


@dataclass
class TranscriptionResult:
    text: str
    type: ResultType
    language: str | None = None
    confidence: float | None = None
    timestamp: float = field(default_factory=time)
