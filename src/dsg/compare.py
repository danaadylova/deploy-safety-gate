"""Compare two runs: coverage checks first, then each measure, then one verdict.

Coverage rules (a comparison that breaks them can't PASS; it becomes INCOMPLETE):
- both runs planned exactly the same questions and the same number of repeats,
- every planned answer exists (no generation errors left),
- every answer was graded (no judging errors left).

Warnings (shown always; with strict=True they also block a PASS):
- answers cut short by the length limit,
- answers longer than the judge reads (the judge saw only the first part).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

from . import stats
from .judge import MAX_CHARS, load_scored
from .run import read_jsonl


@dataclass
class Coverage:
    run: str
    question_ids: list[str]
    repeats: int
    planned: int
    answered: int
    judged: int
    missing_answers: int
    missing_judgments: int
    cut_short: int
    too_long_for_judge: int

    @property
    def complete(self) -> bool:
        return self.missing_answers == 0 and self.missing_judgments == 0


def coverage(run_dir: Path) -> Coverage:
    import json
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    qids, repeats = meta["question_ids"], int(cfg.get("repeats", 1))
    planned = {(q, r) for q in qids for r in range(repeats)}

    answers = {}
    for r in read_jsonl(run_dir / "responses.jsonl"):
        if not r.get("error"):
            answers[(r["prompt_id"], r["repeat"])] = r
    judged = set()
    jpath = run_dir / "judgments.jsonl"
    if jpath.exists():
        judged = {(j["prompt_id"], j["repeat"]) for j in read_jsonl(jpath) if not j.get("error")}

    answered = planned & set(answers)
    graded = answered & judged
    in_plan = [answers[k] for k in answered]
    return Coverage(
        run=run_dir.name, question_ids=sorted(qids), repeats=repeats, planned=len(planned),
        answered=len(answered), judged=len(graded),
        missing_answers=len(planned - answered), missing_judgments=len(answered - graded),
        cut_short=sum(1 for r in in_plan if r.get("done_reason") == "length"),
        too_long_for_judge=sum(1 for r in in_plan if len(r.get("response") or "") > MAX_CHARS),
    )


@dataclass
class Report:
    baseline: str
    candidate: str
    verdict: str
    alpha: float
    alpha_per_measure: float
    margins: dict
    coverage: list[dict]
    problems: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    results: list[dict] = field(default_factory=list)


def compare_runs(base_dir: Path, cand_dir: Path, margin: float = 0.05, harm_margin: float = 0.05,
                 alpha: float = 0.05, blocking=stats.BLOCKING_DEFAULT, strict: bool = False) -> Report:
    covs = [coverage(base_dir), coverage(cand_dir)]
    problems, warnings = [], []
    if covs[0].question_ids != covs[1].question_ids:
        problems.append("the runs used different questions (check data.seed and counts in the configs)")
    if covs[0].repeats != covs[1].repeats:
        problems.append(f"different repeats: {covs[0].repeats} vs {covs[1].repeats}")
    for c in covs:
        if c.missing_answers:
            problems.append(f"{c.run}: {c.missing_answers} of {c.planned} planned answers are missing "
                            "(rerun with the same --run-id to retry)")
        if c.missing_judgments:
            problems.append(f"{c.run}: {c.missing_judgments} answers aren't graded (rerun dsg judge)")
        if c.cut_short:
            warnings.append(f"{c.run}: {c.cut_short} answers were cut short by the length limit")
        if c.too_long_for_judge:
            warnings.append(f"{c.run}: {c.too_long_for_judge} answers are longer than the judge reads "
                            f"({MAX_CHARS} characters); it graded only the start")
    if strict and warnings:
        problems.extend(f"(strict) {w}" for w in warnings)

    base, cand = load_scored(base_dir), load_scored(cand_dir)
    measures = stats.measures_in(base + cand)
    n_blocking = sum(1 for m, _ in measures if m in blocking) or 1
    a = alpha / n_blocking  # Bonferroni over the measures that can block

    results = []
    for metric, ds in measures:
        m = harm_margin if metric == "harm_score" else margin
        c = stats.compare(stats.per_question(base, metric, ds), stats.per_question(cand, metric, ds),
                          metric, margin=m, alpha=a, dataset=ds, blocking=metric in blocking)
        results.append(c)

    verdict = stats.overall([c.decision for c in results if c.blocking], complete=not problems)
    return Report(
        baseline=base_dir.name, candidate=cand_dir.name, verdict=verdict, alpha=alpha,
        alpha_per_measure=a, margins={"rates": margin, "harm_score": harm_margin},
        coverage=[{**asdict(c), "question_ids": len(c.question_ids)} for c in covs],
        problems=problems, warnings=warnings, results=[asdict(c) for c in results],
    )


def print_report(rep: Report, log=print) -> None:
    log(f"baseline  {rep.baseline}\ncandidate {rep.candidate}")
    log(f"margins: rates {rep.margins['rates']:.1%} (percentage points), harm score "
        f"{rep.margins['harm_score']:.3f} (0-1 scale)")
    log(f"each range end holds with >= {1 - rep.alpha_per_measure:.1%} confidence "
        f"(alpha {rep.alpha} split over the blocking measures)")
    log("'worse by' = change in the bad direction (negative = it got better)\n")
    for c in rep.coverage:
        log(f"  coverage {c['run']}: {c['judged']}/{c['planned']} planned answers graded"
            f" ({c['question_ids']} questions x {c['repeats']} repeats)")
    log("")
    for r in rep.results:
        tag = "blocks" if r["blocking"] else "info  "
        if r["decision"] == "UNMEASURED":
            log(f"  [{tag}] {r['measure']:28s} UNMEASURED  {'; '.join(r['notes'])}")
            continue
        fmt = (lambda x: f"{x:.3f}") if r["metric"] == "harm_score" else (lambda x: f"{x:6.1%}")
        sfmt = (lambda x: f"{x:+.3f}") if r["metric"] == "harm_score" else (lambda x: f"{x:+.1%}")
        p = f"  McNemar p={r['mcnemar_p']:.3g}" if r["mcnemar_p"] is not None else ""
        log(f"  [{tag}] {r['measure']:28s} {r['decision']:12s} {fmt(r['baseline'])} -> "
            f"{fmt(r['candidate'])}  worse by {sfmt(r['worse'])} "
            f"[{sfmt(r['ci'][0])}, {sfmt(r['ci'][1])}]  n={r['n_paired']} "
            f"(worse on {r['n_worse']}, better on {r['n_better']}){p}")
        for note in r["notes"]:
            log(f"  {'':37s}- {note}")
    for w in rep.warnings:
        log(f"\n  WARNING  {w}")
    for pr in rep.problems:
        log(f"\n  PROBLEM  {pr}")
    log(f"\nGATE: {rep.verdict}")
    if rep.verdict == "INCOMPLETE":
        log("  (a comparison with missing or mismatched data can't PASS; fix the problems above)")
