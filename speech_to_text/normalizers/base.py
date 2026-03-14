from __future__ import annotations

from abc import ABC, abstractmethod


class Normalizer(ABC):
    """Abstract base class for text normalizers.

    Subclass this to implement custom post-processing of transcription
    results (spell-checking, LLM-based correction, domain-specific
    replacements, etc.).

    The only required method is :meth:`normalize` which receives the raw
    transcribed text and must return the corrected version.
    """

    @abstractmethod
    def normalize(self, text: str) -> str:
        """Fix recognition errors in *text* and return the corrected version."""
