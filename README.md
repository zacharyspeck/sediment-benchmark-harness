# sediment-benchmark-harness

An evaluation harness for measuring whether a language model that has **studied**
a body of private institutional history answers questions about it better than
the same model **searching** that history at query time.

This repository contains the harness: the graders, the statistics, the item
generator, the question templates, and the test suite. It does not contain the
corpus, the generated items, or the answer keys. That is deliberate, and
[why](#what-is-withheld-and-why) is the most important section of this README.

---

## The claim being tested

The benchmark is built around a three-arm comparison over a synthetic private
equity corpus — Westbrook Capital Partners, a fictional PE firm, and
Halloran & Co., a fictional sell-side bank. Neither firm exists; no real company,
filing, or client material appears anywhere in the corpus or in this repository.

| arm | what it is |
|---|---|
| **A — sediment** | An open-weights base model with a LoRA adapter fine-tuned on the corpus, plus a search tool over that same corpus. |
| **B — reference** | A strong general model driven headless with the corpus as its working directory. Reported as a reference point, not a control. |
| **C — base** | The same base model with **no adapter**, plus the same search tool. |

**A minus C is the only comparison in which everything except studied knowledge
is held fixed.** The two arms are the same weights, tools, prompts, and decoding
parameters, and differ at exactly one configuration path. `tests/test_arms_abc.py`
asserts that structural identity, so configuration drift fails the test suite
rather than quietly becoming part of the result.

Both arms can search the same corpus. That is the point: questions whose answers
sit in retrievable text should tie, and questions requiring knowledge that exists
in no document are where a fine-tune can show.

## Six families of equal standing

They are reported side by side and **never averaged** — different graders,
different item counts, and source rows that overlap heavily.

| family | what it measures |
|---|---|
| `single_fact_recall` | one document holds the answer |
| `multi_doc_aggregation` | more documents than one search returns |
| `corpus_wide_statistic` | a property of the whole corpus |
| `implicit_rule_application` | institutional behaviour that appears in no document |
| `convention_conformance` | house drafting conventions, checked deterministically |
| `absence_and_abstention` | a thin record, where the wrong answer is an invention |

A `recency` slice is reported separately and is not one of the six.

## Three things that are load-bearing

1. **No LLM judge, anywhere.** Every check is a number comparison, a string
   comparison, or a regex. A convention that cannot be written as one is listed
   under `deliberately_not_checked` rather than approximated.
2. **No score without an interval.** When two arms' intervals overlap the report
   prints "no significant difference" and does not name a winner. It states the
   Wilson half-width at the item count before it shows you any score.
3. **A constant-answer baseline beside every family score.** A family that a
   fixed string can beat is not measuring the model. Two degenerate arms are
   reported and the higher of the two is the honest floor.

---

## Running it

Requires Python 3.11+. The only runtime dependency is PyYAML, and the harness
ships a fallback parser for environments without it.

```bash
git clone https://github.com/zacharyspeck/sediment-benchmark-harness.git
cd sediment-benchmark-harness
python -m venv .venv
. .venv/bin/activate            # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
pytest -q
```

One command runs everything — the test suite, template resolution against the
fixture ledger, and an end-to-end fixture run into a report:

```bash
./scripts/verify.sh             # macOS / Linux
.\scripts\verify.ps1            # Windows
```

It exits non-zero on any failure, makes no model calls, and touches no network.

### What passes on a clean clone

The suite runs green with no corpus and no private data present. A block of
tests covering family 4's holdout selection **skips** rather than fails, and
announces why — those paths need the private pool. That is the intended
experience for an external reader: everything that can be verified without the
corpus is verified, and everything that cannot says so out loud.

### Network arms are off twice over

Every network arm is `enabled: false` in the committed config **and** additionally
gated on `BENCHMARK_ALLOW_NETWORK=1`. Both switches are off and no transport is
implemented for any of them. Nothing in this repository can make a model call.

---

## What is withheld, and why

This repository is one half of a deliberately split pair. The other half — the
corpus, the generated items with their gold answers, the family-4 holdout
targets, and the documents describing what each implicit rule does — is private
and stays private.

**The reason is benchmark validity, not secrecy for its own sake.**

The headline question this benchmark asks is whether a model has absorbed
institutional behaviour that is written down nowhere. That question is only
answerable while the behaviour is genuinely unavailable to the system under test.
A control arm whose operator can read the rule in a public repository is not a
control; it is an arm that has been handed the answer, and the difference between
it and the fine-tuned arm stops meaning anything. Publishing the keys would not
weaken this benchmark by some margin — it would end it, permanently and for
everyone, including anyone who later wanted to reproduce a result against it.

So the following are absent here by design:

| withheld | what it is |
|---|---|
| the corpus | the document set both arms search |
| generated item sets | every item carries its gold answer |
| family-4 holdout targets and pool | every target carries its ground-truth verdict |
| implicit rule definitions | which fields each rule reads, and the values it turns on |
| the pre-registration and build reports | these describe rule behaviour in prose |

Two harness components read private definitions at runtime and are stubbed here:
`harness.holdout_pool.BOUNDARY_GAPS` and the family-4 lever map. Both load from
side-files supplied alongside the private pool. When those files are absent the
harness does not silently proceed with an empty configuration — it refuses, or it
measures zero, because a gate that quietly does not run would change the
population while reporting it as unchanged. The mechanism is visible; the values
are not.

### What this means if you want to use it

You can read, run, and adapt the harness. You can point it at your own ledger and
your own corpus, write your own templates, and get scored, interval-bounded
results out of it. What you cannot do is score against *this* corpus, because
this corpus is not here.

If you want to evaluate a system against the private benchmark, that is a
conversation rather than a download — open an issue.

---

## Layout

```
harness/     ledger adapter, corpus index, extractor, runner, stats, report,
             capacity engine, item generator, CLI
graders/     one module per family, plus a two-tier fabrication scorer
items/       question templates, one YAML per family (data, not code)
config/      arms.yaml, ledger.yaml, grading.yaml, fabrication.yaml
fixtures/    hand-written test items with per-response expected scores,
             an adversarial edge-case suite, and a hermetic mini ledger
tests/       the suite (503 tests: 463 pass, 40 skip without the private pool)
scripts/     verify.sh, verify.ps1, venv bootstrap
```

`fixtures/mini_ledger.json` is a small self-contained ledger used only to exercise
the graders and resolve templates. Its deals, figures, and rules are invented for
that purpose and are not drawn from the private corpus.

## Answer format

Every question instructs an arm to answer in two parts:

```
ANSWER: <single value, verdict, or comma-separated list>
<then normal prose reasoning>
```

Graders parse the `ANSWER` line. If it is missing or malformed a fallback
extractor reads the prose and flags the item `extraction_uncertain`. Every run
reports that rate; above 5% the fix is a stronger format instruction in the
prompt, not a cleverer fallback.

## Extending it

**Add an arm:** edit `config/arms.yaml`. No code change. An ablation with search
disabled, a different model, or a cartridge variant are all config entries. Only
a genuinely new *kind* of transport needs Python.

**Add a question template:** edit the relevant file in `items/`. Each file opens
with a worked example of the template shape and the full list of filter and
aggregation operators. Then:

```bash
python -m harness.cli validate-templates
```

which reports how many ledger rows each template would draw from and whether its
gold answer is computable — and generates nothing.

**Add a regression case:** when you find an output that fools a grader, add it to
`fixtures/edge_cases.yaml` with the score it should get.

## Commands

```bash
python -m harness.cli validate-templates     # resolve templates, generate nothing
python -m harness.cli validate-items --items <file>
python -m harness.cli fixtures               # end-to-end fixture run + report
python -m harness.cli run --items <file> --arms a,b --report
python -m harness.cli report --run results/<run_id>
python -m harness.cli allowlist --probe "some text"
```

Runs write each item's result to disk as it completes and resume from partial
results. Every run produces `report.md` and a machine-readable `summary.json`
carrying the same numbers.

## Licence

MIT. See [LICENSE](LICENSE).
