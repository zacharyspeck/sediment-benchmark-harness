"""The run loop: generate (or replay), grade, fabrication-score, write, resume.

DURABILITY
----------
Each item's result is written to disk the moment it completes -- appended to
`grades.jsonl` and dropped as an individual JSON file under `items/` -- and
each write is flushed and fsync'd. A run that dies halfway leaves every
completed item on disk and resumes from exactly there. This is not a
convenience; a benchmark you cannot interrupt is a benchmark you will not
re-run.

FAILURE ISOLATION
-----------------
Every item is wrapped in its own try/except. A grader crash, a malformed
fixture, an arm that raises -- all become an `error` record with the full
traceback preserved, and the run continues. Nothing is swallowed silently:
errors are counted, listed in the report, and never counted as model failures.
"""

from __future__ import annotations

import io
import json
import os
import platform
import subprocess
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable

from graders import GradeContext
from graders.base import grade_item as _grade_item
from graders.fabrication import FabricationScorer
from .arms import Arm, ArmNotWired
from .paths import assert_writable
from .yamlio import WHICH as YAML_PARSER, load_file

__all__ = ["Runner", "RunPaths", "load_items", "run_key"]


def run_key(item_id: str, arm: str) -> str:
    return f"{item_id}::{arm}"


def _safe_name(s: str) -> str:
    return "".join(c if (c.isalnum() or c in "-_.") else "_" for c in str(s))[:120]


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _git_sha(repo: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", repo, "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=15,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


# --------------------------------------------------------------------------
def load_items(path: str) -> list[dict]:
    """Load an items/fixtures file. Fails loud on shape problems."""
    doc = load_file(path)
    if isinstance(doc, list):
        items = doc
    elif isinstance(doc, dict):
        items = doc.get("items")
        if items is None:
            raise ValueError(f"{path}: expected a top-level `items:` list")
    else:
        raise ValueError(f"{path}: expected a mapping or a list, got {type(doc).__name__}")

    seen: set[str] = set()
    for i, it in enumerate(items):
        if not isinstance(it, dict):
            raise ValueError(f"{path}: item #{i} is not a mapping")
        iid = it.get("item_id")
        if not iid:
            raise ValueError(f"{path}: item #{i} has no `item_id`")
        if iid in seen:
            raise ValueError(f"{path}: duplicate item_id {iid!r}")
        seen.add(iid)
        if not it.get("family"):
            raise ValueError(f"{path}: item {iid!r} has no `family`")
    return list(items)


@dataclass
class RunPaths:
    root: str
    grades: str = field(init=False)
    items_dir: str = field(init=False)
    run_json: str = field(init=False)
    log: str = field(init=False)

    def __post_init__(self):
        self.grades = os.path.join(self.root, "grades.jsonl")
        self.items_dir = os.path.join(self.root, "items")
        self.run_json = os.path.join(self.root, "run.json")
        self.log = os.path.join(self.root, "run.log")

    def ensure(self) -> None:
        assert_writable(self.root, what="create run directory")
        os.makedirs(self.items_dir, exist_ok=True)


class Runner:
    def __init__(
        self,
        items: list[dict],
        arms: dict[str, Arm],
        out_dir: str,
        ledger: Any = None,
        config: dict | None = None,
        fabrication_config: dict | None = None,
        repo_root: str | None = None,
    ):
        self.items = items
        self.arms = arms
        self.ledger = ledger
        self.config = dict(config or {})
        self.paths = RunPaths(out_dir)
        self.paths.ensure()
        self.repo_root = repo_root or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.scorer = FabricationScorer(ledger, fabrication_config)
        self.errors: list[dict] = []
        self._log_fh = io.open(assert_writable(self.paths.log, what="open run log"), "a", encoding="utf-8")

    # ------------------------------------------------------------------
    def log(self, msg: str) -> None:
        line = f"[{_utc_now()}] {msg}"
        self._log_fh.write(line + "\n")
        self._log_fh.flush()
        try:
            os.fsync(self._log_fh.fileno())
        except OSError:
            pass
        print(line, flush=True)

    def completed_keys(self) -> set[str]:
        """Keys already on disk, so a resumed run skips them."""
        done: set[str] = set()
        if not os.path.exists(self.paths.grades):
            return done
        with io.open(self.paths.grades, "r", encoding="utf-8") as fh:
            for n, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    # A torn final line from a killed run. Everything before it
                    # is still valid; note it and carry on.
                    self.log(f"WARN grades.jsonl line {n} is unparseable; ignoring it")
                    continue
                k = rec.get("key") or run_key(rec.get("item_id", ""), rec.get("arm", ""))
                done.add(k)
        return done

    def _append(self, record: dict) -> None:
        assert_writable(self.paths.grades, what="append grades")
        with io.open(self.paths.grades, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            fh.flush()
            try:
                os.fsync(fh.fileno())
            except OSError:
                pass
        ipath = os.path.join(self.paths.items_dir, _safe_name(record["key"]) + ".json")
        assert_writable(ipath, what="write item result")
        with io.open(ipath, "w", encoding="utf-8") as fh:
            json.dump(record, fh, ensure_ascii=False, indent=2, default=str)
            fh.flush()
            try:
                os.fsync(fh.fileno())
            except OSError:
                pass

    def _write_run_json(self, resumed: int, total: int) -> None:
        payload = {
            "run_started": _utc_now(),
            "harness_git_sha": _git_sha(self.repo_root),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "yaml_parser": YAML_PARSER,
            "n_items": len(self.items),
            "n_arms": len(self.arms),
            "n_planned": total,
            "n_resumed": resumed,
            "arms": {n: a.describe() for n, a in self.arms.items()},
            "ledger": self.ledger.summary() if self.ledger is not None else None,
            "fabrication_allowlist": self.scorer.allowlist_summary(),
            "config": self.config,
            # Keyed off what CAN actually generate, not off the type name. A
            # disabled `sediment` arm makes no call, and reporting
            # zero_model_calls: false for a run that could not have made one
            # turns a provenance guarantee into noise.
            "zero_model_calls": not any(
                a.runnable() and a.type not in ("fixture", "null", "constant")
                for a in self.arms.values()
            ),
            # The identifier each arm actually resolved to, recorded per run so
            # a result can never be attributed to the wrong weights.
            "resolved_models": {
                n: {
                    "model": a.model,
                    "adapter": (a.params or {}).get("lora_adapter"),
                    "cartridge": a.cartridge,
                    "endpoint": a.endpoint,
                    "tools": list(a.tools),
                    "corpus_dir": (a.params or {}).get("corpus_dir"),
                }
                for n, a in self.arms.items()
            },
        }
        assert_writable(self.paths.run_json, what="write run.json")
        with io.open(self.paths.run_json, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2, default=str)

    # ------------------------------------------------------------------
    def run(self, resume: bool = True) -> dict:
        done = self.completed_keys() if resume else set()
        total = len(self.items) * len(self.arms)
        self._write_run_json(len(done), total)
        self.log(
            f"run start: {len(self.items)} items x {len(self.arms)} arms = {total} gradings; "
            f"{len(done)} already on disk; out={self.paths.root}"
        )
        for name, arm in self.arms.items():
            self.log(f"  arm {name}: type={arm.type} runnable={arm.runnable()} model={arm.model}")

        counts = {"graded": 0, "skipped": 0, "errors": 0}
        t0 = time.time()

        for item in self.items:
            iid = item.get("item_id", "<no-id>")
            for arm_name, arm in self.arms.items():
                key = run_key(iid, arm_name)
                if key in done:
                    counts["skipped"] += 1
                    continue
                rec = self._grade_one(item, arm, arm_name, key)
                self._append(rec)
                counts["graded"] += 1
                if rec.get("verdict") == "error":
                    counts["errors"] += 1
                    self.errors.append({"key": key, "error": rec.get("error")})

        counts["elapsed_s"] = round(time.time() - t0, 2)
        self.log(
            f"run complete: graded={counts['graded']} skipped={counts['skipped']} "
            f"errors={counts['errors']} in {counts['elapsed_s']}s"
        )
        if self.errors:
            self.log(f"ERRORS ({len(self.errors)}): " +
                     "; ".join(e['key'] for e in self.errors[:20]))
        self._log_fh.flush()
        return counts

    # ------------------------------------------------------------------
    def _grade_one(self, item: dict, arm: Arm, arm_name: str, key: str) -> dict:
        question = item.get("question", "")
        ctx = GradeContext(ledger=self.ledger, config=self.config, arm=arm_name)

        # --- generation / replay ---
        try:
            response = arm.generate(question, item)
        except ArmNotWired as exc:
            return {
                "key": key, "item_id": item.get("item_id"), "arm": arm_name,
                "family": item.get("family"), "verdict": "error", "score": 0.0,
                "binary": True, "error": f"ArmNotWired: {exc}",
                "comparison": "arm refused to generate (not wired for network calls)",
                "flags": ["arm_not_wired"], "graded_at": _utc_now(),
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "key": key, "item_id": item.get("item_id"), "arm": arm_name,
                "family": item.get("family"), "verdict": "error", "score": 0.0,
                "binary": True,
                "error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
                "comparison": "arm raised while generating",
                "flags": ["arm_error"], "graded_at": _utc_now(),
            }

        # --- grading (already exception-safe) ---
        grade = _grade_item(item, response, ctx)

        # --- fabrication, independent of correctness ---
        try:
            extra_allow = list((item.get("grading") or {}).get("allow_entities") or [])
            fab = self.scorer.score(response, question=question, extra_allow=extra_allow)
            grade.fabrication = fab.to_dict()
            if fab.fabricated:
                grade.flags.append("fabrication_tier_a")
        except Exception as exc:  # noqa: BLE001
            grade.fabrication = {
                "error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
                "fabricated": None,
            }
            grade.flags.append("fabrication_scorer_error")

        rec = grade.to_dict()
        rec["key"] = key
        rec["graded_at"] = _utc_now()
        rec["question"] = question
        rec["difficulty"] = (item.get("meta") or {}).get("difficulty")
        # What a responder that reads nothing would score on this item's
        # slice. Carried onto the grade record so the REPORT can print the
        # floor beside the score without needing the item file: a family-4
        # number read without its floor is the single most likely way to
        # misread this benchmark.
        rec["chance_floor"] = (item.get("meta") or {}).get("chance_floor")
        rec["gold_class"] = (item.get("meta") or {}).get("gold_class")
        # Whether this item counts toward its family's SCORE. Family 4's controls
        # are graded and reported on their own line but are excluded from the
        # family score by the pre-registration, and the aggregate cannot honour
        # that unless the grade record says so.
        rec["scored"] = (item.get("meta") or {}).get("scored", True)
        return rec
