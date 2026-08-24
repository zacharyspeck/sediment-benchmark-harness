"""An index over the corpus document tree.

The ledger is the GOLD source; the corpus is the EVIDENCE source. Capacity
cannot be measured from the ledger alone -- a question whose answer sits in no
document is not answerable by any arm, and a question the documents contradict
has the wrong gold. Both gates read this index.

Everything here is a substring or regex operation over document text. There is
no model call and no judgement: a gate either matches a pattern or it does not.

COST
----
One pass over ~10k files / ~155 MB, roughly 3-6 seconds and ~400 MB resident.
Built once per audit and reused by every gate, rather than once per template.

DOCUMENT NAMING
---------------
Files are `YYYY-MM-DD_<doctype>_<firmprefix>-<doctype>-NNNN.md` and the first
heading is `# <Doc Type> Project <Codename>`. Both are parsed; the heading is
authoritative for the subject and the filename for the date and doc type,
because the heading does not carry a date and the filename does not reliably
carry the codename.
"""

from __future__ import annotations

import io
import os
import re
from collections import Counter, defaultdict

# `# Screening Memo Project Ashrstead` -> Ashrstead
#
# Captures the REST of the title line, not one word: at least one codename in
# this ledger is two words ("Halifax Loomleighleigh") and a single-token capture silently
# indexed it as "Halifax", which is not a deal. `known_codenames` resolves the
# capture by longest match so multi-word names survive.
_TITLE_RE = re.compile(r"^#\s+.*?\bProject\s+(.+?)\s*$", re.MULTILINE)
_FIRST_TOKEN_RE = re.compile(r"^([A-Z][A-Za-z0-9'\-]*)")
# 2024-04-06_ioi_loi_termsheet_we-ioi_loi_termsheet-0098.md
_FNAME_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})_(.+?)_(?:we|ha)-", re.IGNORECASE)


class Doc:
    __slots__ = ("path", "rel", "firm_dir", "date", "doctype", "subject", "text")

    def __init__(self, path, rel, firm_dir, date, doctype, subject, text):
        self.path = path
        self.rel = rel
        self.firm_dir = firm_dir
        self.date = date
        self.doctype = doctype
        self.subject = subject
        self.text = text

    def __repr__(self):
        return f"<Doc {self.rel} subject={self.subject} date={self.date}>"


class CorpusIndex:
    """Loaded once, queried by every gate."""

    def __init__(self, root: str, known_codenames=None):
        self.root = os.path.abspath(root)
        self.known = set(known_codenames or ())
        # Longest first, so "Halifax Loomleighleigh" wins over a hypothetical "Halifax".
        self._known_sorted = sorted(self.known, key=len, reverse=True)
        self.docs: list[Doc] = []
        self.by_subject: dict[str, list[Doc]] = defaultdict(list)
        self.by_doctype: dict[str, list[Doc]] = defaultdict(list)
        self.unresolved_subjects: Counter = Counter()
        self._lower_all: str | None = None
        self._load()

    def _resolve_subject(self, tail: str) -> str | None:
        """Turn the text after 'Project ' into a ledger codename."""
        tail = tail.strip()
        if not tail:
            return None
        if self.known:
            for cn in self._known_sorted:
                if tail == cn or tail.startswith(cn + " ") or tail.startswith(cn + ","):
                    return cn
        m = _FIRST_TOKEN_RE.match(tail)
        return m.group(1) if m else None

    # ------------------------------------------------------------------ build
    def _load(self) -> None:
        for dirpath, _dirs, files in os.walk(self.root):
            for name in sorted(files):
                if not name.endswith(".md"):
                    continue
                full = os.path.join(dirpath, name)
                rel = os.path.relpath(full, self.root).replace("\\", "/")
                firm_dir = rel.split("/")[0] if "/" in rel else ""
                m = _FNAME_RE.match(name)
                date = m.group(1) if m else ""
                doctype = m.group(2).lower() if m else ""
                try:
                    text = io.open(full, encoding="utf-8").read()
                except (OSError, UnicodeDecodeError):
                    continue
                tm = _TITLE_RE.search(text)
                subject = self._resolve_subject(tm.group(1)) if tm else None
                if subject is not None and self.known and subject not in self.known:
                    self.unresolved_subjects[subject] += 1
                d = Doc(full, rel, firm_dir, date, doctype, subject, text)
                self.docs.append(d)
                if subject:
                    self.by_subject[subject].append(d)
                if doctype:
                    self.by_doctype[doctype].append(d)

    # ------------------------------------------------------------------ query
    def docs_for(self, codename: str | None) -> list[Doc]:
        """Every document whose title names this deal."""
        if not codename:
            return []
        return self.by_subject.get(codename, [])

    def doctypes_for(self, codename: str | None) -> set[str]:
        return {d.doctype for d in self.docs_for(codename) if d.doctype}

    def dates_for(self, codename: str | None) -> set[str]:
        return {d.date for d in self.docs_for(codename) if d.date}

    def has_doc(self, codename: str | None) -> bool:
        return bool(self.docs_for(codename))

    def matches_any(self, codename: str | None, patterns) -> bool:
        """True when any of this deal's documents matches any pattern."""
        if not patterns:
            return False
        rx = [p if hasattr(p, "search") else re.compile(p) for p in patterns]
        for d in self.docs_for(codename):
            for r in rx:
                if r.search(d.text):
                    return True
        return False

    def matching_docs(self, codename: str | None, patterns) -> list[Doc]:
        rx = [p if hasattr(p, "search") else re.compile(p) for p in patterns or ()]
        out = []
        for d in self.docs_for(codename):
            if any(r.search(d.text) for r in rx):
                out.append(d)
        return out

    def has_doctype(self, codename: str | None, doctypes) -> bool:
        want = {t.lower() for t in (doctypes or ())}
        return bool(want & self.doctypes_for(codename))

    # ------------------------------------------- corpus-wide collision screen
    def contains_anywhere(self, needle: str) -> bool:
        """Case-insensitive substring search over the whole corpus.

        Backs the holdout collision check. The concatenated lower-cased blob is
        built once on first use -- ~155 MB, which is worth it because the
        alternative is 10,261 `in` tests per candidate.
        """
        if self._lower_all is None:
            self._lower_all = "\n".join(d.text for d in self.docs).lower()
        return needle.lower() in self._lower_all

    def files_containing(self, needle: str, limit: int = 5) -> list[str]:
        n = needle.lower()
        out = []
        for d in self.docs:
            if n in d.text.lower():
                out.append(d.rel)
                if len(out) >= limit:
                    break
        return out

    # ------------------------------------------------------------------ stats
    def summary(self) -> dict:
        by_dir: dict[str, int] = {}
        for d in self.docs:
            by_dir[d.firm_dir] = by_dir.get(d.firm_dir, 0) + 1
        return {
            "root": self.root,
            "documents": len(self.docs),
            "documents_by_dir": dict(sorted(by_dir.items())),
            "titled": sum(1 for d in self.docs if d.subject),
            "distinct_subjects": len(self.by_subject),
            "distinct_doctypes": len(self.by_doctype),
            "distinct_dates": len({d.date for d in self.docs if d.date}),
            "subjects_not_in_ledger": dict(self.unresolved_subjects.most_common(10)),
        }


_CACHE: dict[str, CorpusIndex] = {}


def load_corpus_index(root: str, known_codenames=None) -> CorpusIndex:
    """Memoised per path -- an audit builds this once, not once per family."""
    key = os.path.abspath(root)
    if key not in _CACHE:
        _CACHE[key] = CorpusIndex(key, known_codenames)
    return _CACHE[key]
