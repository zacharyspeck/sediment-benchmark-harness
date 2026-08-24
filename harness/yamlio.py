"""YAML loading with a documented fallback.

Prefers PyYAML (pinned). Falls back to harness._minyaml so a bare Python can
still run the harness. `WHICH` records which parser was used, and the report
prints it -- if a run was graded with the fallback parser you should be able to
see that without digging.
"""

from __future__ import annotations

import io
import os
from typing import Any

try:  # pragma: no cover - trivial import branch
    import yaml as _pyyaml

    WHICH = "pyyaml"
except Exception:  # pragma: no cover
    _pyyaml = None
    WHICH = "minyaml"

from . import _minyaml


def loads(src: str) -> Any:
    if _pyyaml is not None:
        return _pyyaml.safe_load(src)
    return _minyaml.safe_load(src)


def load_file(path: str | os.PathLike) -> Any:
    with io.open(path, "r", encoding="utf-8") as fh:
        src = fh.read()
    try:
        return loads(src)
    except Exception as exc:  # fail loud, with the file name attached
        raise ValueError(f"failed to parse YAML: {path}: {type(exc).__name__}: {exc}") from exc


def loads_both(src: str):
    """Parse with both parsers. Used by tests to assert they agree."""
    if _pyyaml is None:
        raise RuntimeError("PyYAML not installed; cannot cross-check")
    return _pyyaml.safe_load(src), _minyaml.safe_load(src)
