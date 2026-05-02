"""Optional native acceleration with a Python fallback."""

from __future__ import annotations

from collections import deque


def bfs_neighbors(adjacency: dict[str, list[str]], start: str, depth: int) -> list[tuple[str, int]]:
    queue: deque[tuple[str, int]] = deque([(start, 0)])
    visited = {start}
    ordered: list[tuple[str, int]] = []
    while queue:
        current, hop_count = queue.popleft()
        if hop_count >= depth:
            continue
        for neighbor in adjacency.get(current, []):
            if neighbor in visited:
                continue
            visited.add(neighbor)
            next_depth = hop_count + 1
            ordered.append((neighbor, next_depth))
            queue.append((neighbor, next_depth))
    return ordered


def has_native_backend() -> bool:
    return False
