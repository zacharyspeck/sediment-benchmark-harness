"""The fallback YAML parser must agree with PyYAML on every file in the repo.

harness/_minyaml.py exists so a bare Python can still load config and
templates. That promise is only worth anything if the two parsers actually
agree -- otherwise the fallback is a trap that changes what a template means.
"""

import glob
import io
import os
import re

import pytest

from harness import _minyaml, yamlio

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

YAML_FILES = sorted(
    glob.glob(os.path.join(REPO, "config", "*.yaml"))
    + glob.glob(os.path.join(REPO, "items", "*.yaml"))
    + glob.glob(os.path.join(REPO, "fixtures", "*.yaml"))
)


def test_there_are_yaml_files_to_check():
    assert len(YAML_FILES) >= 9, f"expected the repo's YAML files, found {YAML_FILES}"


@pytest.mark.parametrize("path", YAML_FILES, ids=[os.path.basename(p) for p in YAML_FILES])
def test_minyaml_agrees_with_pyyaml(path):
    pytest.importorskip("yaml")
    with io.open(path, "r", encoding="utf-8") as fh:
        src = fh.read()
    expected, got = yamlio.loads_both(src)
    assert got == expected, f"{os.path.basename(path)}: fallback parser disagrees with PyYAML"


# An anchor is `&name` with no space after the ampersand. This must NOT match
# ordinary prose like "consumer products & services", which is a real sector.
_ANCHOR = re.compile(r"(?:^|[\s:\[{,])&[A-Za-z_][\w.-]*(?:\s|$)")
_ALIAS = re.compile(r"(?:^|[\s:\[{,])\*[A-Za-z_][\w.-]*(?:\s|$)")


@pytest.mark.parametrize("path", YAML_FILES, ids=[os.path.basename(p) for p in YAML_FILES])
def test_no_anchors_or_aliases_in_repo_yaml(path):
    """Anchors are unsupported by the fallback, so they are banned outright."""
    with io.open(path, "r", encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            code = line.split("#", 1)[0]
            assert not _ANCHOR.search(code), (
                f"{os.path.basename(path)}:{n} defines a YAML anchor; "
                f"the fallback parser does not support them -- repeat yourself instead")
            assert not _ALIAS.search(code), (
                f"{os.path.basename(path)}:{n} uses a YAML alias; unsupported by the fallback")


# -------------------------------------------------------------- unit coverage
def test_scalars():
    d = _minyaml.safe_load("a: 1\nb: 2.5\nc: true\nd: null\ne: hello\nf: 'x'\ng: \"y\"\n")
    assert d == {"a": 1, "b": 2.5, "c": True, "d": None, "e": "hello", "f": "x", "g": "y"}


def test_nested_mappings_and_sequences():
    src = """
root:
  list:
    - one
    - two
  inner:
    k: v
    nums: [1, 2, 3]
    map: {a: 1, b: 2}
"""
    assert _minyaml.safe_load(src) == {
        "root": {"list": ["one", "two"],
                 "inner": {"k": "v", "nums": [1, 2, 3], "map": {"a": 1, "b": 2}}}
    }


def test_sequence_of_mappings():
    src = """
items:
  - id: a
    n: 1
  - id: b
    n: 2
    deep:
      x: 1
"""
    assert _minyaml.safe_load(src) == {
        "items": [{"id": "a", "n": 1}, {"id": "b", "n": 2, "deep": {"x": 1}}]
    }


def test_block_scalars():
    src = "lit: |\n  line1\n  line2\nfold: >-\n  a\n  b\n"
    d = _minyaml.safe_load(src)
    assert d["lit"] == "line1\nline2\n"
    assert d["fold"] == "a b"


def test_comments_are_stripped_but_not_inside_quotes():
    d = _minyaml.safe_load('a: 1  # trailing\n# whole line\nb: "x # y"\n')
    assert d == {"a": 1, "b": "x # y"}


def test_sequence_at_same_indent_as_key():
    src = "key:\n- a\n- b\n"
    assert _minyaml.safe_load(src) == {"key": ["a", "b"]}


def test_tab_indentation_is_rejected_loudly():
    with pytest.raises(_minyaml.MinYamlError):
        _minyaml.safe_load("a:\n\tb: 1\n")


def test_which_parser_is_recorded():
    assert yamlio.WHICH in ("pyyaml", "minyaml")
