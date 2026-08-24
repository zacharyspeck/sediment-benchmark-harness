r"""Write containment.

The corpus at `../sediment-corpus` is READ-ONLY to this repo. Before this
module existed, eight CLI flags reached an unvalidated write and

    python -m harness.cli report --run ../sediment-corpus/ledger-v4

deposited `report.md` and `summary.json` into the corpus and exited 0.

Every filesystem-mutating call site in `harness/` now routes its target
through `assert_writable()` first. The allowed roots are deliberately narrow:
the repo's own `results/` tree, plus whatever the caller opts into explicitly
via `allow_roots=` (the test suite uses this for `tmp_path`).

WHY REALPATH AND NOT ABSPATH
----------------------------
`os.path.abspath` does lexical normalisation only: it collapses `..` textually
without touching the filesystem. A directory junction or symlink under
`results/` pointing at the corpus survives `abspath` untouched and is caught
only by `realpath`, which resolves links. On Windows this also normalises 8.3
short names (`SEDIME~1`) and the `\\?\` long-path prefix, both of which are
otherwise distinct strings naming the same directory.

WHAT THIS DOES NOT DO
---------------------
It does not make a universal negative true. `--out` accepts any string, and a
future write site added without calling `assert_writable` would bypass it.
`tests/test_paths.py::test_no_mutating_site_bypasses_the_guard` is what closes
that hole: it AST-scans `harness/` and fails when a mutating call appears in a
function that never consults the guard.
"""

from __future__ import annotations

import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: Roots a write is always permitted under. Relative to REPO.
DEFAULT_ALLOWED = ("results", "logs", "data")

#: Named artifacts the harness may write at the repo root. An explicit list of
#: filenames, not a blanket permission on the root: a stray write still fails,
#: and adding a new artifact is a deliberate one-line edit rather than a silent
#: consequence of widening a directory.
ALLOWED_REPO_FILES = (
    "CAPACITY.md",
    "capacity.json",
    "BUILD-REPORT.md",
    "implementation-notes.md",
    # v2 build artifacts. Named individually, deliberately: the repo root is not
    # a writable directory and must not become one.
    "PRE-REGISTRATION.md",
    "phase1-input-check.md",
    "leak-check.md",
    "capacity-delta-v6.md",
    "degeneracy-v2.md",
    "build-report-v2.md",
    "phase7-verify.txt",
)

#: Roots that are refused even if some other rule would allow them. This is a
#: belt-and-braces second opinion, not the primary mechanism: the primary
#: mechanism is that everything outside DEFAULT_ALLOWED is refused by default.
ALWAYS_FORBIDDEN = ("sediment-corpus",)


class RefusedWrite(PermissionError):
    """Raised when a write would land outside the allowed roots."""


#: Process-wide opt-in roots. The test suite registers pytest's tmp_path base
#: here via an autouse fixture (tests/conftest.py) so hermetic tests can write
#: without every production signature growing an `allow_roots` parameter.
#: Nothing in shipped code calls `permit_root`.
_PERMITTED: list[str] = []


def permit_root(path: str) -> None:
    """Register an additional writable root for this process."""
    r = os.path.normcase(os.path.realpath(os.path.abspath(path)))
    if r not in _PERMITTED:
        _PERMITTED.append(r)


def clear_permitted() -> None:
    _PERMITTED.clear()


def _real(path: str) -> str:
    """Fully resolved, case-folded-on-Windows absolute path.

    `realpath` resolves symlinks and junctions; `normcase` folds case and
    separators on Windows so `C:/x` and `c:\\X` compare equal.
    """
    return os.path.normcase(os.path.realpath(os.path.abspath(path)))


def _is_within(child: str, parent: str) -> bool:
    """True when `child` is `parent` or sits underneath it.

    Uses `os.path.relpath` rather than string prefixing, so `/a/bc` is not
    treated as living inside `/a/b`.
    """
    try:
        rel = os.path.relpath(child, parent)
    except ValueError:
        # Different drives on Windows -- cannot be contained.
        return False
    return rel == os.curdir or not rel.startswith(os.pardir + os.sep) and rel != os.pardir


def allowed_roots(extra: tuple[str, ...] | list[str] | None = None) -> list[str]:
    roots = [_real(os.path.join(REPO, r)) for r in DEFAULT_ALLOWED]
    roots.extend(_PERMITTED)
    for r in extra or ():
        roots.append(_real(r))
    return roots


def is_writable(path: str, allow_roots=None) -> bool:
    """Whether `assert_writable` would permit this path."""
    target = _real(path)
    for bad in ALWAYS_FORBIDDEN:
        # Match a path COMPONENT, never a substring: a legitimate results dir
        # named `sediment-corpus-notes` must not be refused.
        if bad in target.replace("/", os.sep).split(os.sep):
            return False
    if any(_is_within(target, root) for root in allowed_roots(allow_roots)):
        return True
    for name in ALLOWED_REPO_FILES:
        if target == _real(os.path.join(REPO, name)):
            return True
    return False


def assert_writable(path: str, allow_roots=None, what: str = "write") -> str:
    """Raise `RefusedWrite` unless `path` is inside an allowed root.

    Returns the path unchanged so it can be used inline:

        with io.open(assert_writable(p), "w") as fh: ...
    """
    if not is_writable(path, allow_roots):
        raise RefusedWrite(
            "refused to {} {!r}\n"
            "  resolves to: {}\n"
            "  allowed roots: {}\n"
            "The corpus is read-only to this repo. If this path is legitimate, "
            "pass allow_roots= explicitly rather than widening DEFAULT_ALLOWED.".format(
                what, path, _real(path), ", ".join(allowed_roots(allow_roots))
            )
        )
    return path
