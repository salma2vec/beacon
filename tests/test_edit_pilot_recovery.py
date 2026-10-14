import pytest

from examples.complete_edit_pilot import missing_cell


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
