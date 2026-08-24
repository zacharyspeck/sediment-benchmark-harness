"""Arms are configuration, not code.

Adding a cartridge arm, changing Claude's setup, or running an ablation is an
edit to config/arms.yaml. Nothing in this file needs to change to add an arm --
only to add a new *kind* of arm (a new transport).

TONIGHT: ZERO MODEL CALLS
-------------------------
Every network-backed arm type is declared so the config schema is real and
reviewable, and every one of them refuses to run. A generating arm requires
BOTH:

    config/arms.yaml       enabled: true
    environment            BENCHMARK_ALLOW_NETWORK=1

Both default off. There is no code path in this repository that reaches a model
endpoint without those two switches being thrown deliberately, and no API key
is read, stored or defaulted anywhere.

The only arm type that runs tonight is `fixture`, which replays hand-written
responses from disk.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from .yamlio import load_file

__all__ = ["Arm", "ArmNotWired", "load_arms", "FixtureArm", "NullArm", "build_arm"]

NETWORK_ENV = "BENCHMARK_ALLOW_NETWORK"


class ArmNotWired(RuntimeError):
    """Raised when something asks a non-runnable arm to generate."""


@dataclass
class Arm:
    """One system under test."""

    name: str
    type: str
    enabled: bool = False
    model: str | None = None
    endpoint: str | None = None
    tools: list = field(default_factory=list)
    cartridge: Any = None
    params: dict = field(default_factory=dict)
    notes: str = ""
    raw: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    def describe(self) -> dict:
        return {
            "name": self.name, "type": self.type, "enabled": self.enabled,
            "model": self.model, "endpoint": self.endpoint,
            "tools": list(self.tools), "cartridge": self.cartridge,
            "params": dict(self.params), "notes": self.notes,
            "runnable": self.runnable(),
        }

    def runnable(self) -> bool:
        return False

    def generate(self, question: str, item: dict) -> str:
        raise ArmNotWired(
            f"arm {self.name!r} (type {self.type!r}) cannot generate.\n"
            f"Network arms require BOTH `enabled: true` in config/arms.yaml AND "
            f"the environment variable {NETWORK_ENV}=1. Both are off by design: "
            f"this harness was built under a zero-model-call constraint."
        )


class NullArm(Arm):
    """Returns nothing. Used to exercise the pipeline's empty-response path."""

    def runnable(self) -> bool:
        return True

    def generate(self, question: str, item: dict) -> str:
        return ""


class FixtureArm(Arm):
    """Replays hand-written responses keyed by item_id. No model involved."""

    responses: dict

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.responses = {}

    def load_responses(self, mapping: dict) -> None:
        self.responses = dict(mapping or {})

    def runnable(self) -> bool:
        return True

    def generate(self, question: str, item: dict) -> str:
        # Fixture items carry their responses inline, one per arm, so the
        # question and every arm's answer to it sit in the same reviewable
        # block of YAML.
        inline = (item.get("responses") or {})
        if self.name in inline:
            return inline[self.name]
        key = item.get("item_id")
        if key in self.responses:
            return self.responses[key]
        raise KeyError(
            f"fixture arm {self.name!r} has no response for item {key!r}. "
            f"Every fixture item must carry a `responses:` entry for every fixture arm "
            f"(found: {sorted(inline)})."
        )


class ConstantArm(Arm):
    """Answers the same string to every item, regardless of the question.

    WHY THIS EXISTS
    ---------------
    Some families can be scored highly without reading anything. Measured on
    this corpus: every admissible discriminating holdout target carries gold
    `pass`, so a responder that ignores the facts and always says "pass" scores
    100% on `implicit_rule_application`'s scored slice. A family score that a
    constant beats is not evidence about the model.

    `null_arm` does not cover this -- it returns an empty string and scores 0,
    which is the opposite failure. Without a constant baseline the degenerate
    ceiling is invisible, and the number gets read as if a model earned it.

    It is not a model and makes no call, so it runs even under the zero-network
    constraint and its score is printed beside every family score.
    """

    def runnable(self) -> bool:
        return True

    def generate(self, question: str, item: dict) -> str:
        answer = self.params.get("answer", "")
        fmt = self.params.get("format", "answer_line")
        if fmt == "raw":
            return str(answer)
        return (f"ANSWER: {answer}\n\n"
                "Constant baseline: this response is identical for every item.")


class NetworkArm(Arm):
    """Declared transport for a real system. Refuses to run tonight.

    Kept concrete rather than hypothetical so the config schema below is the
    real one -- when the arms are wired, the fields are already named and the
    report already prints them.
    """

    def runnable(self) -> bool:
        return bool(self.enabled) and os.environ.get(NETWORK_ENV) == "1"

    def generate(self, question: str, item: dict) -> str:
        if not self.runnable():
            raise ArmNotWired(
                f"arm {self.name!r} (type {self.type!r}) is not wired.\n"
                f"  enabled in config: {self.enabled}\n"
                f"  {NETWORK_ENV}: {os.environ.get(NETWORK_ENV) or '<unset>'}\n"
                f"Both must be on. No transport is implemented in this repository yet -- "
                f"implement it in harness/arms.py when the arms are ready to run."
            )
        raise NotImplementedError(
            f"arm type {self.type!r} has no transport implementation. "
            f"Add it here; the runner needs only `generate(question, item) -> str`."
        )


_TYPES = {
    "fixture": FixtureArm,
    "null": NullArm,
    "constant": ConstantArm,          # degenerate baseline; not a model
    "local_cli_subprocess": NetworkArm,   # Arm B: Claude Code headless
    "sediment": NetworkArm,           # Arm A: Qwen 8B + LoRA + corpus search
    "anthropic_api": NetworkArm,      # Arm B: Claude + searchable corpus dir
    "openai_compatible": NetworkArm,
    "local_cli": NetworkArm,
    "http": NetworkArm,
}


def build_arm(name: str, spec: dict) -> Arm:
    spec = dict(spec or {})
    atype = str(spec.get("type") or "").strip()
    cls = _TYPES.get(atype)
    if cls is None:
        raise ValueError(
            f"arm {name!r} has unknown type {atype!r}; known types: {sorted(_TYPES)}"
        )
    return cls(
        name=name,
        type=atype,
        enabled=bool(spec.get("enabled", False)),
        model=spec.get("model"),
        endpoint=spec.get("endpoint"),
        tools=list(spec.get("tools") or []),
        cartridge=spec.get("cartridge"),
        params=dict(spec.get("params") or {}),
        notes=str(spec.get("notes") or ""),
        raw=spec,
    )


def load_arms(path: str, only: list[str] | None = None) -> dict[str, Arm]:
    """Load config/arms.yaml into Arm objects."""
    cfg = load_file(path) or {}
    arms_cfg = cfg.get("arms") or {}
    if not arms_cfg:
        raise ValueError(f"no `arms:` block found in {path}")
    out: dict[str, Arm] = {}
    for name, spec in arms_cfg.items():
        if only and name not in only:
            continue
        out[name] = build_arm(name, spec)
    if only:
        missing = [n for n in only if n not in out]
        if missing:
            raise ValueError(f"requested arm(s) not in {path}: {missing}")
    return out
