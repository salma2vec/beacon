from __future__ import annotations

from pathlib import Path

from cegraph.mutations import discover_mutations_in_file


def test_discover_mutations_finds_valid_changes(tmp_path: Path) -> None:
    file_path = tmp_path / "sample.py"
    file_path.write_text(
        """
def greet() -> str:
    value: int = 1
    return "hi" if True else "bye"
""".strip(),
        encoding="utf-8",
    )
    candidates = discover_mutations_in_file(file_path, tmp_path)
    assert candidates
    assert any(candidate.mutation_type == "type_change" for candidate in candidates)

