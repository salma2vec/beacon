"""Report publication must not follow output links or result-supplied paths."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock

import pytest

from paper.experiments import generate_figures, merge_and_report, posthoc_analysis, router_ab
from paper.experiments.safe_output import atomic_output, figure_filename, write_text_atomic


def make_run(tmp_path: Path, query_type: str = "vague") -> Path:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    row = {
        "task_id": "fixture",
        "method": "bca",
        "query_type": query_type,
        "budget": 64,
        "pass_at_1": True,
    }
    (run_dir / "results.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    return run_dir


@pytest.mark.parametrize(
    ("module", "filenames"),
    [
        (generate_figures, ["figures.tsv"]),
        (merge_and_report, ["report.md", "report.tex"]),
        (posthoc_analysis, ["posthoc_analysis.json"]),
        (router_ab, ["router_ab.json"]),
    ],
)
def test_report_scripts_replace_preplanted_links_without_touching_targets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, module: ModuleType, filenames: list[str]
) -> None:
    run_dir = make_run(tmp_path)
    monkeypatch.setattr(generate_figures, "plt", None)
    for filename in filenames:
        outside = tmp_path / filename
        outside.write_bytes(b"outside marker")
        (run_dir / filename).symlink_to(outside)
    module.main(["--run-dir", str(run_dir)])
    for filename in filenames:
        assert (tmp_path / filename).read_bytes() == b"outside marker"
        assert not (run_dir / filename).is_symlink()
        assert (run_dir / filename).read_bytes() != b"outside marker"
    assert not list(run_dir.glob(".*.tmp"))


@pytest.mark.parametrize("link_kind", ["symlink", "broken_symlink", "hardlink"])
def test_atomic_writer_does_not_modify_linked_inodes(tmp_path: Path, link_kind: str) -> None:
    outside = tmp_path / "outside"
    destination = tmp_path / "report"
    if link_kind != "broken_symlink":
        outside.write_bytes(b"original")
    if link_kind == "hardlink":
        os.link(outside, destination)
    else:
        destination.symlink_to(outside)
    write_text_atomic(destination, "report\n")
    assert destination.read_bytes() == b"report\n"
    if link_kind == "broken_symlink":
        assert not outside.exists()
    else:
        assert outside.read_bytes() == b"original"


def test_failed_report_write_leaves_previous_output_and_removes_temporary(tmp_path: Path) -> None:
    destination = tmp_path / "report"
    destination.write_bytes(b"previous")
    with pytest.raises(RuntimeError, match="renderer failed"), atomic_output(destination) as handle:
        handle.write(b"partial")
        raise RuntimeError("renderer failed")
    assert destination.read_bytes() == b"previous"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["report"]


def test_figure_renderer_uses_owned_file_handle_and_retains_legitimate_label(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    query_type = "dev report-é.v2"
    run_dir = make_run(tmp_path, query_type)
    output = run_dir / f"figure_{query_type}.png"
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"original")
    output.symlink_to(outside)
    figure, axis = MagicMock(), MagicMock()
    figure.savefig.side_effect = lambda handle, **kwargs: handle.write(b"PNG fixture")
    plot = MagicMock()
    plot.subplots.return_value = figure, axis
    monkeypatch.setattr(generate_figures, "plt", plot)
    assert generate_figures.main(["--run-dir", str(run_dir)]) == [output]
    assert output.read_bytes() == b"PNG fixture" and not output.is_symlink()
    assert outside.read_bytes() == b"original"
    assert figure.savefig.call_args.kwargs == {"format": "png"}
    plot.close.assert_called_once_with(figure)


@pytest.mark.parametrize(
    "label",
    [
        "",
        ".",
        "..",
        "x/../../outside",
        "../outside",
        "x\\..\\outside",
        "bad\x00",
        "bad\n",
        "a" * 181,
    ],
)
@pytest.mark.parametrize("fallback", [False, True])
def test_unsafe_query_labels_fail_before_any_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, label: str, fallback: bool
) -> None:
    run_dir = make_run(tmp_path, label)
    plot = None if fallback else MagicMock()
    monkeypatch.setattr(generate_figures, "plt", plot)
    with pytest.raises(ValueError, match="query_type"):
        generate_figures.main(["--run-dir", str(run_dir)])
    assert sorted(path.name for path in run_dir.iterdir()) == ["results.jsonl"]
    if plot is not None:
        plot.subplots.assert_not_called()


@pytest.mark.parametrize("label", ["vague", "exact", "dev_report", "custom-label", "évidence 2"])
def test_safe_query_labels_keep_existing_filenames(label: str) -> None:
    assert figure_filename(label) == f"figure_{label}.png"


def test_report_tables_escape_untrusted_labels() -> None:
    aggregates = {("<script>|m_#%", 64, "q&$\\\nnext"): 1.0}
    markdown = merge_and_report.build_markdown_table(aggregates)
    latex = merge_and_report.build_latex_table(aggregates)
    assert "<script>" not in markdown
    assert "&lt;script&gt;\\|m_#%" in markdown
    assert "q&amp;$\\\\&#10;next" in markdown
    assert r"<script>|m\_\#\%" in latex
    assert r"q\&\$\textbackslash{} next" in latex


def test_report_serialization_is_unchanged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_dir = make_run(tmp_path)
    monkeypatch.setattr(generate_figures, "plt", None)
    rows = merge_and_report.load_results(run_dir)
    aggregated = merge_and_report.aggregate_pass_rates(rows)
    merge_and_report.main(["--run-dir", str(run_dir)])
    posthoc_analysis.main(["--run-dir", str(run_dir)])
    router_ab.main(["--run-dir", str(run_dir)])
    generate_figures.main(["--run-dir", str(run_dir)])
    assert (run_dir / "report.md").read_bytes() == (
        merge_and_report.build_markdown_table(aggregated) + "\n"
    ).encode()
    assert (run_dir / "report.tex").read_bytes() == (
        merge_and_report.build_latex_table(aggregated) + "\n"
    ).encode()
    assert (run_dir / "posthoc_analysis.json").read_bytes() == (
        json.dumps(posthoc_analysis.summarize_router_study(rows), indent=2) + "\n"
    ).encode()
    assert (run_dir / "router_ab.json").read_bytes() == (
        json.dumps(router_ab.compare_methods(rows, "adaptive", "bca"), indent=2) + "\n"
    ).encode()
    assert (run_dir / "figures.tsv").read_bytes() == (
        b"method\tquery_type\tbudget\tpass_at_1\nbca\tvague\t64\t1.000\n"
    )


@pytest.mark.parametrize(
    "module", [generate_figures, merge_and_report, posthoc_analysis, router_ab]
)
def test_report_scripts_still_support_direct_invocation(module: ModuleType, tmp_path: Path) -> None:
    completed = subprocess.run(
        [sys.executable, str(module.__file__), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
