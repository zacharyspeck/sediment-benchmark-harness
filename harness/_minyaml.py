"""A dependency-free parser for the YAML subset this repo actually uses.

WHY THIS EXISTS
---------------
Templates, arm config and fixtures are YAML because they are meant to be
hand-edited. PyYAML is the real dependency (pinned in requirements.txt), but a
benchmark harness that cannot even load its own config on a bare Python install
is a bad harness. This is the fallback.

It is NOT a general YAML implementation. It supports exactly what the repo's
files use:

  - block mappings            key: value
  - block sequences           - item   /   - key: value
  - arbitrary nesting by indentation
  - scalars                   int, float, bool, null, plain and quoted strings
  - block scalars             |  |-  >  >-
  - flow collections          [a, b]   {a: b, c: d}
  - comments                  full-line and trailing (quote-aware)

It deliberately does NOT support anchors, aliases, tags, multi-document
streams, or complex keys. If you reach for one of those in a template, install
PyYAML -- and tests/test_yamlio.py will tell you, because it asserts this
parser and PyYAML agree on every YAML file in the repo.
"""

from __future__ import annotations

import re

__all__ = ["safe_load", "MinYamlError"]


class MinYamlError(ValueError):
    """Raised when the subset parser meets something it will not guess at."""


_INT_RE = re.compile(r"^[+-]?\d+$")
_FLOAT_RE = re.compile(r"^[+-]?(\d+\.\d*|\.\d+|\d+)([eE][+-]?\d+)?$")
_NULLS = {"", "null", "~", "Null", "NULL"}
_TRUE = {"true", "True", "TRUE", "yes", "Yes", "YES", "on", "On"}
_FALSE = {"false", "False", "FALSE", "no", "No", "NO", "off", "Off"}


def _strip_comment(line: str) -> str:
    """Remove a trailing comment, respecting quotes.

    A '#' only starts a comment when it is at the start of the line or
    preceded by whitespace -- so `url: http://x#y` and `a: "b # c"` survive.
    """
    out = []
    quote = None
    prev = ""
    for i, ch in enumerate(line):
        if quote:
            out.append(ch)
            if ch == quote and prev != "\\":
                quote = None
        else:
            if ch in ("'", '"'):
                quote = ch
                out.append(ch)
            elif ch == "#" and (i == 0 or prev in (" ", "\t")):
                break
            else:
                out.append(ch)
        prev = ch
    return "".join(out).rstrip()


def _unescape_double(s: str) -> str:
    out = []
    i = 0
    while i < len(s):
        c = s[i]
        if c == "\\" and i + 1 < len(s):
            n = s[i + 1]
            mapping = {"n": "\n", "t": "\t", "r": "\r", '"': '"', "\\": "\\", "/": "/", "0": "\0"}
            if n in mapping:
                out.append(mapping[n])
                i += 2
                continue
            if n == "u" and i + 5 < len(s) + 1:
                try:
                    out.append(chr(int(s[i + 2 : i + 6], 16)))
                    i += 6
                    continue
                except ValueError:
                    pass
        out.append(c)
        i += 1
    return "".join(out)


def _scalar(tok: str):
    """Convert a scalar token to a Python value."""
    t = tok.strip()
    if len(t) >= 2 and t[0] == t[-1] == '"':
        return _unescape_double(t[1:-1])
    if len(t) >= 2 and t[0] == t[-1] == "'":
        return t[1:-1].replace("''", "'")
    if t in _NULLS:
        return None
    if t in _TRUE:
        return True
    if t in _FALSE:
        return False
    if _INT_RE.match(t):
        try:
            return int(t)
        except ValueError:
            pass
    if _FLOAT_RE.match(t) and not _INT_RE.match(t):
        try:
            return float(t)
        except ValueError:
            pass
    return t


def _split_flow(body: str):
    """Split a flow collection body on top-level commas."""
    parts, cur, depth, quote, prev = [], [], 0, None, ""
    for ch in body:
        if quote:
            cur.append(ch)
            if ch == quote and prev != "\\":
                quote = None
        elif ch in ("'", '"'):
            quote = ch
            cur.append(ch)
        elif ch in "[{":
            depth += 1
            cur.append(ch)
        elif ch in "]}":
            depth -= 1
            cur.append(ch)
        elif ch == "," and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
        prev = ch
    if "".join(cur).strip():
        parts.append("".join(cur))
    return [p.strip() for p in parts]


def _parse_flow(tok: str):
    t = tok.strip()
    if t.startswith("[") and t.endswith("]"):
        body = t[1:-1].strip()
        if not body:
            return []
        return [_parse_value_token(p) for p in _split_flow(body)]
    if t.startswith("{") and t.endswith("}"):
        body = t[1:-1].strip()
        if not body:
            return {}
        out = {}
        for p in _split_flow(body):
            k, sep, v = _split_key(p)
            if not sep:
                raise MinYamlError(f"flow mapping entry without ':' -> {p!r}")
            out[str(_scalar(k))] = _parse_value_token(v)
        return out
    raise MinYamlError(f"not a flow collection: {tok!r}")


def _parse_value_token(tok: str):
    t = tok.strip()
    if t.startswith("[") or t.startswith("{"):
        return _parse_flow(t)
    return _scalar(t)


def _split_key(line: str):
    """Split `key: value` at the first top-level ': ' (or trailing ':').

    Returns (key, found_sep, value). Quote- and bracket-aware so that
    `{a: b}` and `"x: y": z` behave.
    """
    quote, depth, prev = None, 0, ""
    for i, ch in enumerate(line):
        if quote:
            if ch == quote and prev != "\\":
                quote = None
        elif ch in ("'", '"'):
            quote = ch
        elif ch in "[{":
            depth += 1
        elif ch in "]}":
            depth -= 1
        elif ch == ":" and depth == 0:
            rest = line[i + 1 :]
            if rest == "" or rest[0] in (" ", "\t"):
                return line[:i].strip(), True, rest.strip()
        prev = ch
    return line.strip(), False, ""


class _Line:
    # `block` holds the expanded body of a block scalar (| or >). It must be in
    # __slots__ -- without it, assigning .block raises AttributeError and every
    # block scalar in the repo silently fails to parse.
    __slots__ = ("indent", "text", "no", "block")

    def __init__(self, indent, text, no):
        self.indent, self.text, self.no = indent, text, no
        self.block = None


def _lex(src: str):
    """Produce logical lines, expanding block scalars inline."""
    # A UTF-8 BOM survives io.open(encoding="utf-8") and would otherwise be
    # read as the first character of the first key. PyYAML strips it; so do we.
    if src.startswith("﻿"):
        src = src[1:]
    raw = src.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    out = []
    i = 0
    while i < len(raw):
        line = raw[i]
        if "\t" in line[: len(line) - len(line.lstrip("\t "))]:
            raise MinYamlError(f"line {i+1}: tab used for indentation")
        stripped = _strip_comment(line)
        if not stripped.strip():
            i += 1
            continue
        indent = len(stripped) - len(stripped.lstrip(" "))
        text = stripped.strip()
        if text == "---":
            i += 1
            continue
        # block scalar?
        m = re.search(r"(\|[+-]?|>[+-]?)$", text)
        if m:
            style = m.group(1)
            head = text[: m.start()].rstrip()
            i += 1
            body, base = [], None
            while i < len(raw):
                nxt = raw[i]
                if not nxt.strip():
                    body.append("")
                    i += 1
                    continue
                ind = len(nxt) - len(nxt.lstrip(" "))
                if ind <= indent:
                    break
                if base is None:
                    base = ind
                body.append(nxt[base:] if len(nxt) >= base else nxt.lstrip(" "))
                i += 1
            while body and body[-1] == "":
                body.pop()
            if style.startswith("|"):
                val = "\n".join(body)
            else:
                folded, buf = [], []
                for b in body:
                    if b.strip() == "":
                        folded.append(" ".join(buf))
                        buf = []
                        folded.append("")
                    else:
                        buf.append(b.strip())
                if buf:
                    folded.append(" ".join(buf))
                val = "\n".join(folded)
            if not style.endswith("-"):
                val += "\n"
            out.append(_Line(indent, head + " \x00BLOCK\x00", len(out) + 1))
            out[-1].block = val  # type: ignore[attr-defined]
            continue
        out.append(_Line(indent, text, i + 1))
        i += 1
    return out


def _block_value(ln: _Line):
    return ln.block


def _parse_block(lines, pos, indent):
    """Parse a block node starting at lines[pos] with the given indent."""
    if pos >= len(lines):
        return None, pos
    if lines[pos].text.startswith("- "):
        return _parse_seq(lines, pos, indent)
    if lines[pos].text == "-":
        return _parse_seq(lines, pos, indent)
    return _parse_map(lines, pos, indent)


def _parse_seq(lines, pos, indent):
    items = []
    while pos < len(lines) and lines[pos].indent == indent and (
        lines[pos].text == "-" or lines[pos].text.startswith("- ")
    ):
        ln = lines[pos]
        rest = ln.text[1:].strip() if ln.text != "-" else ""
        blk = _block_value(ln)
        if rest == "" and blk is None:
            pos += 1
            if pos < len(lines) and lines[pos].indent > indent:
                val, pos = _parse_block(lines, pos, lines[pos].indent)
                items.append(val)
            else:
                items.append(None)
            continue
        key, sep, val_tok = _split_key(rest)
        if sep:
            # sequence entry that is itself a mapping; synthesise the inline
            # first pair then continue at the entry's own indent
            child_indent = ln.indent + 2
            synth = [_Line(child_indent, rest, ln.no)]
            if blk is not None:
                synth[0].block = blk  # type: ignore[attr-defined]
            j = pos + 1
            while j < len(lines) and lines[j].indent > indent:
                synth.append(lines[j])
                j += 1
            base = min(l.indent for l in synth)
            norm = []
            for l in synth:
                nl = _Line(l.indent - base + child_indent, l.text, l.no)
                b = _block_value(l)
                if b is not None:
                    nl.block = b  # type: ignore[attr-defined]
                norm.append(nl)
            val, _ = _parse_map(norm, 0, child_indent)
            items.append(val)
            pos = j
        else:
            items.append(blk if blk is not None else _parse_value_token(rest))
            pos += 1
    return items, pos


def _parse_map(lines, pos, indent):
    out = {}
    while pos < len(lines) and lines[pos].indent == indent:
        ln = lines[pos]
        if ln.text.startswith("- ") or ln.text == "-":
            break
        key, sep, val_tok = _split_key(ln.text)
        if not sep:
            raise MinYamlError(f"line {ln.no}: expected 'key: value', got {ln.text!r}")
        k = _scalar(key)
        if isinstance(k, str):
            k = k.strip()
        blk = _block_value(ln)
        if blk is not None:
            out[k] = blk
            pos += 1
            continue
        if val_tok != "":
            out[k] = _parse_value_token(val_tok)
            pos += 1
            continue
        # value lives on following, more-indented lines
        pos += 1
        if pos < len(lines) and lines[pos].indent > indent:
            child, pos = _parse_block(lines, pos, lines[pos].indent)
            out[k] = child
        elif pos < len(lines) and lines[pos].indent == indent and (
            lines[pos].text.startswith("- ") or lines[pos].text == "-"
        ):
            # sequence at the same indent as its key (valid YAML)
            child, pos = _parse_seq(lines, pos, indent)
            out[k] = child
        else:
            out[k] = None
    return out, pos


def safe_load(src: str):
    """Parse a YAML document from a string. Returns dict/list/scalar/None."""
    lines = _lex(src)
    if not lines:
        return None
    base = lines[0].indent
    val, pos = _parse_block(lines, 0, base)
    if pos != len(lines):
        raise MinYamlError(
            f"line {lines[pos].no}: unexpected indentation {lines[pos].indent} "
            f"(document base is {base}) at {lines[pos].text!r}"
        )
    return val
