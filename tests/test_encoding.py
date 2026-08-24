"""Every committed text file must be clean UTF-8, no BOM, no mojibake.

This is a Windows hazard, not a theoretical one. PowerShell 5.1's
`Set-Content -Encoding utf8` writes a BOM, and a read/write round-trip through
the wrong codepage smears every non-ASCII character. Both happened during this
build:

  - a BOM on items/implicit_rule_application.yaml broke the fallback YAML
    parser, which read it as the first character of the first key;
  - a bulk rewrite of README.md re-encoded every em-dash into a three-character
    smear.

Neither is obvious in a diff, and both survive into anything the harness
renders. So they are asserted.

Fix, if this ever fails: rewrite the file with the editing tooling, or
`[System.IO.File]::WriteAllText($p, $text, (New-Object System.Text.UTF8Encoding($false)))`
-- never `Set-Content -Encoding utf8` on a file containing non-ASCII.
"""

from __future__ import annotations

import os
import subprocess

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEXT_EXT = (".md", ".yaml", ".yml", ".py", ".txt", ".ps1", ".json", ".gitignore")


def _smear(*byte_values: int) -> str:
    """Build a mojibake marker from raw byte values.

    Constructed rather than written literally so THIS FILE STAYS PURE ASCII.
    The scanner below runs over every tracked file including its own source; a
    literal marker here would make the mojibake test fail on itself, and an
    escape sequence is not reliable either because tooling that rewrites the
    file may normalise it back into the character it denotes. Byte values
    survive any of that.
    """
    return bytes(byte_values).decode("latin-1")


MOJIBAKE_MARKERS = (
    _smear(0xE2, 0x80),        # em-dash / smart quotes round-tripped
    _smear(0xC3, 0xA9),        # e-acute round-tripped
    _smear(0xEF, 0xBB, 0xBF),  # a BOM decoded as text
    _smear(0xC2, 0xB1),        # plus-minus round-tripped
    _smear(0xC2, 0xA0),        # non-breaking space round-tripped
)


def _tracked_text_files() -> list[str]:
    out = subprocess.run(["git", "-C", REPO, "ls-files"],
                         capture_output=True, text=True, timeout=60)
    files = [f for f in out.stdout.split("\n") if f.strip()]
    return [f for f in files if f.endswith(TEXT_EXT) or f == ".gitignore"]


TRACKED = _tracked_text_files()


def test_there_are_tracked_text_files():
    assert len(TRACKED) > 30, f"expected the repo's text files, found {len(TRACKED)}"


def test_this_file_is_pure_ascii():
    """The scanner must not be able to trip on its own marker list."""
    raw = open(os.path.abspath(__file__), "rb").read()
    bad = [(i, b) for i, b in enumerate(raw) if b > 0x7F]
    assert not bad, f"test_encoding.py must stay ASCII; non-ASCII at {bad[:5]}"


@pytest.mark.parametrize("rel", TRACKED, ids=lambda r: r)
def test_file_is_clean_utf8(rel):
    path = os.path.join(REPO, rel)
    raw = open(path, "rb").read()

    assert not raw.startswith(b"\xef\xbb\xbf"), (
        f"{rel} starts with a UTF-8 BOM; harness/_minyaml.py strips it defensively "
        f"but the file should not carry one")

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        pytest.fail(f"{rel} is not valid UTF-8: {exc}")

    found = [repr(m) for m in MOJIBAKE_MARKERS if m in text]
    assert not found, (
        f"{rel} contains mojibake {found} -- it was round-tripped through the "
        f"wrong codepage. Rewrite it as UTF-8 without a BOM.")


@pytest.mark.parametrize("rel", [f for f in TRACKED if f.endswith(".py")], ids=lambda r: r)
def test_python_files_have_no_tab_indentation(rel):
    path = os.path.join(REPO, rel)
    with open(path, "r", encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            stripped = line.lstrip(" ")
            assert not stripped.startswith("\t"), f"{rel}:{n} indents with a tab"
