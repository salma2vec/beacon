"""Exact rendered-context token counts using the documented cl100k_base codec."""

from __future__ import annotations

import tiktoken


class TokenCounter:
    """Count literal source text, including strings resembling special tokens."""

    def __init__(self) -> None:
        self._encoding = tiktoken.get_encoding("cl100k_base")
        self._cache: dict[str, int] = {}

    def count(self, text: str) -> int:
        if text not in self._cache:
            if len(self._cache) >= 16384:
                self._cache.clear()
            self._cache[text] = len(self._encoding.encode(text, disallowed_special=()))
        return self._cache[text]
