"""Public package entrypoints for beacon."""

from .edits import (
    ExactEditContract,
    LineSpanEditContract,
    grounded_edit_contract,
    grounded_line_span_contract,
)
from .graph import CodebaseGraph
from .methods import assemble_context, available_methods
from .models import AssembledContext, BenchmarkResult, ContextChunk, TaskRecord

__all__ = [
    "__version__",
    "AssembledContext",
    "BenchmarkResult",
    "CodebaseGraph",
    "ContextChunk",
    "ExactEditContract",
    "LineSpanEditContract",
    "TaskRecord",
    "assemble_context",
    "available_methods",
    "grounded_edit_contract",
    "grounded_line_span_contract",
]

__version__ = "0.1.0"
