from __future__ import annotations

from typing import Any

from .base import Normalizer

_NORMALIZER_REGISTRY: dict[str, type[Normalizer]] = {}


def register_normalizer(name: str, normalizer_class: type[Normalizer]) -> None:
    """Register a custom normalizer class under *name*."""
    _NORMALIZER_REGISTRY[name] = normalizer_class


def create_normalizer(name: str, **config: Any) -> Normalizer:
    """Create a normalizer by registered name."""
    if name not in _NORMALIZER_REGISTRY:
        _try_load_normalizer(name)
    if name not in _NORMALIZER_REGISTRY:
        available = ", ".join(sorted(_NORMALIZER_REGISTRY)) or "(none)"
        raise ValueError(
            f"Unknown normalizer '{name}'. Available: {available}. "
            f"Install the required extra: pip install speech-to-text[normalize]"
        )
    return _NORMALIZER_REGISTRY[name](**config)


def _try_load_normalizer(name: str) -> None:
    loaders = {
        "llm": _load_llm,
    }
    loader = loaders.get(name)
    if loader is not None:
        loader()


def _load_llm() -> None:
    try:
        import openai as _openai  # noqa: F401
    except ImportError:
        raise ImportError(
            "openai is not installed. Run: pip install speech-to-text[normalize]"
        ) from None
    from .llm_normalizer import LLMNormalizer
    register_normalizer("llm", LLMNormalizer)
