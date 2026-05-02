"""Token counting utilities."""

from __future__ import annotations

from functools import lru_cache

try:
    import tiktoken
except ImportError:  # pragma: no cover - optional in tests
    tiktoken = None


class TokenCounter:
    """Count tokens with `cl100k_base`, or use a stable text fallback."""

    def __init__(self) -> None:
        self._encoding = tiktoken.get_encoding("cl100k_base") if tiktoken else None

    @lru_cache(maxsize=16384)
    def count(self, text: str) -> int:
        if not text:
            return 0
        if self._encoding is not None:
            return len(self._encoding.encode(text))
        return max(1, len(text.split()) + len(text) // 16)

