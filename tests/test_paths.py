"""The corpus is read-only. These tests are what makes that claim checkable.

Before harness/paths.py existed this was true:

    python -m harness.cli report --run ../sediment-corpus/ledger-v4
    -> wrote report.md + summary.json into the corpus, exit 0

The honest framing, stated here so nobody over-reads these tests: a test cannot
assert a universal negative over unbounded runtime input. `--out` takes any
string. What these tests assert is narrower and checkable:

  1. the guard refuses a table of adversarial paths that all name the corpus
  2. no mutating call site in harness/ bypasses the guard (AST scan)
  3. the specific command that used to write into the corpus now raises
"""

from __future__ import annotations

import ast
import io
import os

import pytest

from harness import paths as P
from harness.paths import RefusedWrite, assert_writable, is_writable

REPO = P.REPO
CORPUS = os.path.join(os.path.dirname(REPO), "sediment-corpus")


# ---------------------------------------------------------------- adversarial
# Every entry names the corpus by a different spelling. All must be refused.
ADVERSARIAL = [
    "../sediment-corpus/ledger-v4",
    "../sediment-corpus/ledger-v4/index.json",
    os.path.join(REPO, "..", "sediment-corpus", "ledger-v4"),
    CORPUS,
    os.path.join(CORPUS, "corpus-v4", "westbrook"),
    # relative traversal that climbs back out of an allowed root
    os.path.join(REPO, "results", "..", "..", "sediment-corpus", "x"),
    "results/../../sediment-corpus/x",
    # forward slashes on Windows
    REPO.replace("\\", "/") + "/../sediment-corpus/x",
    # trailing separator and redundant components
    os.path.join(CORPUS, ""),
    os.path.join(CORPUS, ".", "ledger-v4"),
]


@pytest.mark.parametrize("bad", ADVERSARIAL)
def test_guard_refuses_every_spelling_of_the_corpus(bad):
    assert is_writable(bad) is False
    with pytest.raises(RefusedWrite):
        assert_writable(bad)


def test_guard_refuses_the_windows_long_path_prefix():
    """\\\\?\\C:\\... names the same directory and must not slip through."""
    if os.name != "nt":
        pytest.skip("Windows-only path form")
    assert is_writable("\\\\?\\" + os.path.abspath(CORPUS)) is False


def test_guard_refuses_a_unc_path():
    """A UNC path is outside every allowed root, so it is refused by default."""
    assert is_writable(r"\\server\share\sediment-corpus\ledger-v4") is False
    assert is_writable(r"\\server\share\anything") is False


def test_guard_refuses_the_8_dot_3_short_name():
    """SEDIME~1 resolves to the corpus; realpath is what catches it."""
    if os.name != "nt" or not os.path.isdir(CORPUS):
        pytest.skip("needs the real corpus on Windows")
    short = os.path.join(os.path.dirname(REPO), "SEDIME~1")
    # Only meaningful if the short name actually resolves to the corpus.
    if os.path.normcase(os.path.realpath(short)) != os.path.normcase(os.path.realpath(CORPUS)):
        pytest.skip("8.3 short names are disabled on this volume")
    assert is_writable(short) is False


def test_guard_refuses_a_junction_pointing_at_the_corpus(tmp_path):
    """A link under an ALLOWED root that targets the corpus.

    This is the case abspath cannot see and realpath can. If the OS refuses to
    create the link (no privilege), skip rather than pass vacuously.
    """
    if not os.path.isdir(CORPUS):
        pytest.skip("corpus not present")
    link = os.path.join(REPO, "results", "_junction_probe")
    if os.path.exists(link):
        pytest.skip("probe path already exists")
    try:
        os.symlink(CORPUS, link, target_is_directory=True)
    except (OSError, NotImplementedError, AttributeError):
        pytest.skip("cannot create a directory link in this environment")
    try:
        # `results/` IS an allowed root, so a lexical check would allow this.
        assert is_writable(os.path.join(link, "x.md")) is False
    finally:
        try:
            os.remove(link)
        except OSError:
            os.rmdir(link)


# ------------------------------------------------------------------- positive
def test_guard_allows_the_results_tree():
    assert is_writable(os.path.join(REPO, "results", "run-x", "report.md")) is True


def test_guard_allows_an_explicitly_permitted_root(tmp_path):
    outside = str(tmp_path / "elsewhere" / "x.md")
    assert is_writable(outside, allow_roots=[str(tmp_path)]) is True
    assert assert_writable(outside, allow_roots=[str(tmp_path)]) == outside


def test_a_results_dir_whose_name_merely_contains_the_word_is_allowed():
    """ALWAYS_FORBIDDEN matches a path component, not a substring."""
    ok = os.path.join(REPO, "results", "sediment-corpus-notes", "report.md")
    assert is_writable(ok) is True


def test_repo_root_itself_is_not_writable():
    """Only results/ and logs/ are open. A stray write to the repo root fails."""
    assert is_writable(os.path.join(REPO, "stray.md")) is False


# --------------------------------------------------- the regression that bit
def test_report_command_refuses_to_write_into_the_corpus(tmp_path):
    """The exact invocation that used to deposit bytes in the corpus."""
    from harness.report import write_report, write_summary_json

    target = os.path.join(CORPUS, "ledger-v4", "report.md")
    with pytest.raises(RefusedWrite):
        write_report(target, "# nope\n")
    with pytest.raises(RefusedWrite):
        write_summary_json(os.path.join(CORPUS, "ledger-v4", "summary.json"), {}, [])
    assert not os.path.exists(target)


def test_cli_report_into_the_corpus_exits_2_without_a_traceback(capsys):
    """The guard surfaces as a clean exit code, not an unhandled exception."""
    from harness.cli import main

    rc = main(["report", "--run", "../sediment-corpus/ledger-v4"])
    assert rc == 2
    assert "REFUSED" in capsys.readouterr().err


def test_runner_refuses_an_out_dir_inside_the_corpus(items, ledger, grading_config):
    from harness.arms import load_arms
    from harness.runner import Runner

    arms = load_arms(os.path.join(REPO, "config", "arms.yaml"), ["null_arm"])
    with pytest.raises(RefusedWrite):
        Runner(items=items[:1], arms=arms,
               out_dir=os.path.join(CORPUS, "ledger-v4", "_run"),
               ledger=ledger, config=grading_config)


# ------------------------------------------------------------------- AST scan
# Names that mutate ONLY when called on os / shutil. `replace`, `remove` and
# `copy` are also ordinary str/list methods -- `t.replace("`", "'")` is not a
# filesystem write, and treating it as one made this scan fire on four
# innocent functions the first time it ran.
MODULE_MUTATORS = {
    "makedirs", "mkdir", "remove", "unlink", "rmdir", "rename", "replace",
    "rmtree", "copy", "copy2", "copyfile", "copytree", "move",
}
MUTATOR_RECEIVERS = {"os", "shutil"}
# pathlib.Path methods. These names have no string/list counterpart, so the
# receiver does not need checking.
PATH_MUTATORS = {"write_text", "write_bytes", "touch"}
WRITE_MODES = ("w", "a", "x", "+")


def _harness_sources():
    d = os.path.join(REPO, "harness")
    for name in sorted(os.listdir(d)):
        if name.endswith(".py") and name != "paths.py":
            yield os.path.join(d, name)


def _opens_for_write(node: ast.Call) -> bool:
    fn = node.func
    name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
    if name != "open":
        return False
    mode = None
    if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
        mode = node.args[1].value
    for kw in node.keywords:
        if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
            mode = kw.value.value
    if not isinstance(mode, str):
        return False
    return any(m in mode for m in WRITE_MODES)


def _receiver(fn: ast.Attribute) -> str:
    v = fn.value
    if isinstance(v, ast.Name):
        return v.id
    if isinstance(v, ast.Attribute):          # os.path.<x>
        return _receiver(v) or v.attr
    return ""


def _is_mutating(node: ast.Call) -> bool:
    fn = node.func
    if _opens_for_write(node):
        return True
    if not isinstance(fn, ast.Attribute):
        return False
    if fn.attr in PATH_MUTATORS:
        return True
    return fn.attr in MODULE_MUTATORS and _receiver(fn) in MUTATOR_RECEIVERS


def test_no_mutating_site_bypasses_the_guard():
    """Every function in harness/ that mutates the filesystem consults the guard.

    This is the test that keeps the other tests honest: it fails when someone
    adds a new write site without routing it through assert_writable, which is
    the only realistic way the guard gets holed.
    """
    offenders = []
    for src in _harness_sources():
        tree = ast.parse(io.open(src, encoding="utf-8").read(), filename=src)
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call)]
            if not any(_is_mutating(c) for c in calls):
                continue
            guarded = any(
                (isinstance(c.func, ast.Name) and c.func.id == "assert_writable")
                or (isinstance(c.func, ast.Attribute) and c.func.attr == "assert_writable")
                for c in calls
            )
            if not guarded:
                offenders.append(f"{os.path.basename(src)}::{fn.name}")
    assert offenders == [], (
        "these functions mutate the filesystem without calling assert_writable: "
        + ", ".join(offenders)
    )


def test_the_scan_would_actually_catch_a_bypass(tmp_path):
    """Positive control -- proves the AST scan is not vacuously passing."""
    bad = tmp_path / "bypass.py"
    bad.write_text(
        "import io, os, shutil\n"
        "def sneaky(p):\n"
        "    os.makedirs(p)\n"
        "    io.open(p, 'w').write('x')\n"
        "def innocent(t):\n"
        "    return t.replace('a', 'b')\n",
        encoding="utf-8",
    )
    tree = ast.parse(bad.read_text(encoding="utf-8"))
    sneaky, innocent = tree.body[1], tree.body[2]

    sneaky_calls = [n for n in ast.walk(sneaky) if isinstance(n, ast.Call)]
    assert any(_is_mutating(c) for c in sneaky_calls), "scan missed a real write"
    assert not any(
        isinstance(c.func, ast.Name) and c.func.id == "assert_writable"
        for c in sneaky_calls
    )

    # and the other half of the control: str.replace must NOT read as a write
    innocent_calls = [n for n in ast.walk(innocent) if isinstance(n, ast.Call)]
    assert not any(_is_mutating(c) for c in innocent_calls), "scan over-fires on str.replace"


def test_named_repo_artifacts_are_writable_but_the_root_is_not():
    """CAPACITY.md is allowed by name; a neighbouring file is not.

    The allowance is a list of filenames rather than permission on the repo
    root, so adding an artifact stays a deliberate edit.
    """
    assert is_writable(os.path.join(REPO, "CAPACITY.md")) is True
    assert is_writable(os.path.join(REPO, "capacity.json")) is True
    assert is_writable(os.path.join(REPO, "BUILD-REPORT.md")) is True
    assert is_writable(os.path.join(REPO, "CAPACITY.md.bak")) is False
    assert is_writable(os.path.join(REPO, "harness", "cli.py")) is False
    assert is_writable(os.path.join(REPO, "items", "single_fact_recall.yaml")) is False


def test_the_data_dir_is_writable_and_still_cannot_tunnel_out():
    assert is_writable(os.path.join(REPO, "data", "holdout-targets-v2.json")) is True
    assert is_writable(os.path.join(REPO, "data", "..", "..", "sediment-corpus", "x")) is False
