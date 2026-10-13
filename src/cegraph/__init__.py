"""Public package entrypoints for beacon."""

from .graph import CodebaseGraph
from .methods import assemble_context, available_methods
from .models import AssembledContext, BenchmarkResult, ContextChunk, TaskRecord

__all__ = [
    "__version__",
    "AssembledContext",
    "BenchmarkResult",
    "CodebaseGraph",
    "ContextChunk",
    "TaskRecord",
    "assemble_context",
    "available_methods",
]

__version__ = "0.1.0"
