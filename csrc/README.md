# Optional native path

`beacon` keeps its runtime usable without any compiled extension. The optional
native path can accelerate graph traversal and neighbor expansion, but the
Python fallback in `src/cegraph/native.py` remains the supported interface for
all callers.

## Expected shape

A future native module should expose the same surface as `cegraph.native`:

- `neighbors(graph, node)` -> iterable of neighbor node IDs
- `reachable(graph, seeds, depth)` -> iterable of reachable node IDs

When the compiled module is missing, importers should continue to use the Python
fallback without special handling.
