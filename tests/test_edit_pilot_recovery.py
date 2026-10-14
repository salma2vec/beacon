import json
from pathlib import Path

import pytest

from examples.complete_edit_pilot import digest_bytes, missing_cell, verify_parent_evidence


def matrix():
    return [
        {
            "task_id": task,
            "method": method,
            "budget": budget,
            "format": fmt,
            "status": "error" if budget == 256 else "completed",
        }
        for task in ("a", "b", "c")
        for method in ("bm25", "bca")
        for budget in (256, 1024)
        for fmt in ("diff", "exact_edit")
    ]


def test_recovery_never_retries_recorded_errors():
    assert missing_cell(matrix()[:-1], ["a", "b", "c"]) == ("c", "bca", 1024, "exact_edit")


@pytest.mark.parametrize("change", ["duplicate", "unknown", "running", "short", "complete"])
def test_ambiguous_or_out_of_protocol_recovery_is_rejected(change):
    rows = matrix()[:-1]
    if change == "duplicate":
        rows[-1] = rows[0]
    elif change == "unknown":
        rows[-1]["method"] = "oracle"
    elif change == "running":
        rows[-1]["status"] = "running"
    elif change == "short":
        rows.pop()
    else:
        rows = matrix()
    with pytest.raises(ValueError):
        missing_cell(rows, ["a", "b", "c"])


def parent_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "executor.py").write_text("value = 1\n")
    parent = tmp_path / "parent"
    parent.mkdir()
    protocol = {
        "source_sha256": {
            "executor.py": digest_bytes((source_root / "executor.py").read_bytes())
        }
    }
    artifacts = {
        "protocol.json": json.dumps(protocol).encode(),
        "results.jsonl": b'{"status":"completed"}\n',
        "controls.jsonl": b'{"baseline":true}\n',
    }
    for name, data in artifacts.items():
        (parent / name).write_bytes(data)
    manifest = tmp_path / "parent-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "beacon-parent-evidence-v1",
                "parent_directory": str(parent.resolve()),
                "artifacts": {name: digest_bytes(data) for name, data in artifacts.items()},
            }
        )
    )
    return source_root, parent, manifest


def test_parent_evidence_requires_independent_manifest_and_contained_regular_files(
    tmp_path: Path,
) -> None:
    source_root, parent, manifest = parent_fixture(tmp_path)
    verified = verify_parent_evidence(parent, manifest, source_root)
    assert set(verified) == {
        "parent_manifest",
        "protocol.json",
        "results.jsonl",
        "controls.jsonl",
    }
    inside_manifest = parent / "parent-manifest.json"
    inside_manifest.write_bytes(manifest.read_bytes())
    with pytest.raises(ValueError, match="outside the parent artifact directory"):
        verify_parent_evidence(parent, inside_manifest, source_root)

    outside = tmp_path / "outside"
    outside.write_text("forged\n")
    (parent / "results.jsonl").unlink()
    (parent / "results.jsonl").symlink_to(outside)
    with pytest.raises(ValueError, match="regular file"):
        verify_parent_evidence(parent, manifest, source_root)


def test_parent_source_paths_cannot_escape_source_root(tmp_path: Path) -> None:
    source_root, parent, manifest = parent_fixture(tmp_path)
    protocol = json.loads((parent / "protocol.json").read_text())
    protocol["source_sha256"] = {"../outside.py": "0" * 64}
    (parent / "protocol.json").write_text(json.dumps(protocol))
    payload = json.loads(manifest.read_text())
    payload["artifacts"]["protocol.json"] = digest_bytes((parent / "protocol.json").read_bytes())
    manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="contained relative path"):
        verify_parent_evidence(parent, manifest, source_root)
