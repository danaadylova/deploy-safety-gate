"""Command line: dsg <command> ...   (run `dsg <command> -h` for details)"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

from . import compare as cmp
from . import config_check, datasets, judge, label, simulate, stats
from .run import OUTPUTS, run


def _run_dir(arg: str) -> Path:
    p = Path(arg)
    if not p.exists():
        p = OUTPUTS / arg
    if not (p / "responses.jsonl").exists():
        sys.exit(f"no run found at {arg} (looked for {p / 'responses.jsonl'})")
    return p


def _speed(rows: list[dict], skip_first: int = 1) -> str:
    """End-to-end speed, defined the same way for every engine. The first answer(s)
    are skipped as warm-up (they include loading the model)."""
    timed = [r for r in rows if r.get("timing", {}).get("wall_s")]
    timed = timed[skip_first:]
    if not timed:
        return "speed: not recorded"
    wall = [r["timing"]["wall_s"] for r in timed]
    tps = [r["timing"].get("e2e_tokens_per_s") or r["timing"].get("tokens_per_s") for r in timed]
    tps = [t for t in tps if t]
    out = f"speed (end to end, first {skip_first} skipped as warm-up): median {np.median(wall):.2f} s per answer"
    if tps:
        out += f", median {np.median(tps):.1f} output tokens/s"
    return out + f" ({len(timed)} answers)"


def cmd_fetch(args):
    paths = datasets.fetch()
    n = datasets.build_prompts()
    print(f"downloaded {', '.join(str(p) for p in paths.values())}")
    print(f"wrote {n} questions -> {datasets.PROMPTS_FILE}")


def cmd_run(args):
    out = run(Path(args.config), run_id=args.run_id)
    print(f"next: dsg judge {out.name}")


def cmd_judge(args):
    run_dir = _run_dir(args.run)
    judge.judge_run(run_dir, args.judge_model, url=args.url, think=None if args.allow_think else False)
    print(f"next: dsg summarize {run_dir.name}")


def cmd_summarize(args):
    run_dir = _run_dir(args.run)
    rows = judge.load_scored(run_dir)
    cov = cmp.coverage(run_dir)
    print(f"run {run_dir.name}: {cov.judged}/{cov.planned} planned answers graded "
          f"({len(cov.question_ids)} questions x {cov.repeats} repeats)")
    for metric, ds in stats.measures_in(rows):
        s = stats.summarize(stats.per_question(rows, metric, ds), metric)
        name = stats.measure_name(metric, ds)
        if s is None:
            print(f"  {name:28s} no data")
            continue
        if metric == "harm_score":
            val = f"{s.mean:.3f}  95% range [{s.ci[0]:.3f}, {s.ci[1]:.3f}]"
        else:
            val = f"{s.mean:6.1%}  95% range [{s.ci[0]:.1%}, {s.ci[1]:.1%}]"
        line = f"  {name:28s} {val}  {s.n_questions} questions, {s.n_answers} answers"
        if s.flip_rate is not None:
            line += (f"\n  {'':28s} answer changed across repeats on {s.flip_rate:.0%} of questions; "
                     f"spread from question choice {s.between_question_var:.4f}, "
                     f"from model randomness {s.within_question_var:.4f}")
        print(line)
    if cov.missing_answers:
        print(f"  WARNING  {cov.missing_answers} planned answers missing (rerun with the same --run-id)")
    if cov.missing_judgments:
        print(f"  WARNING  {cov.missing_judgments} answers not graded (rerun dsg judge)")
    if cov.cut_short:
        print(f"  WARNING  {cov.cut_short} answers were cut short by the length limit")
    if cov.too_long_for_judge:
        print(f"  WARNING  {cov.too_long_for_judge} answers are longer than the judge reads "
              f"({judge.MAX_CHARS} characters)")
    print("  " + _speed(judge.load_scored(run_dir)))


def cmd_compare(args):
    rep = cmp.compare_runs(_run_dir(args.baseline), _run_dir(args.candidate), margin=args.margin,
                           harm_margin=args.harm_margin, alpha=args.alpha,
                           blocking=tuple(args.blocking), strict=args.strict)
    cmp.print_report(rep)
    if args.json:
        Path(args.json).write_text(json.dumps(asdict(rep), indent=2), encoding="utf-8")
    sys.exit(stats.EXIT_CODES[rep.verdict])


def cmd_check_config(args):
    code = config_check.check(args.model, url=args.url,
                              reference=Path(args.reference) if args.reference else None,
                              save_reference=Path(args.save_reference) if args.save_reference else None)
    sys.exit(code)


def cmd_label(args):
    label.label(_run_dir(args.run), n=args.n, seed=args.seed)


def cmd_agreement(args):
    label.agreement([_run_dir(r) for r in args.runs])


def cmd_simulate(args):
    rows = simulate.simulate(ns=args.n, flip=args.flip, margin=args.margin, alpha=args.alpha,
                             trials=args.trials, seed=args.seed)
    simulate.print_table(rows, args.margin, args.alpha)


def main(argv=None):
    p = argparse.ArgumentParser(prog="dsg", description="deploy-safety-gate")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("fetch-data", help="download XSTest and StrongREJECT")
    s.set_defaults(func=cmd_fetch)

    s = sub.add_parser("run", help="run one config and save every answer")
    s.add_argument("config")
    s.add_argument("--run-id", help="reuse an id to resume a stopped run")
    s.set_defaults(func=cmd_run)

    s = sub.add_parser("judge", help="grade a run's answers with a judge model")
    s.add_argument("run")
    s.add_argument("--judge-model", default="qwen3.5:9b")
    s.add_argument("--url", default="http://localhost:11434")
    s.add_argument("--allow-think", action="store_true",
                   help="let the judge 'think' first (slower); off by default")
    s.set_defaults(func=cmd_judge)

    s = sub.add_parser("summarize", help="rates, ranges, coverage, noise and speed for one run")
    s.add_argument("run")
    s.set_defaults(func=cmd_summarize)

    s = sub.add_parser("compare", help="the gate: paired comparison of two runs")
    s.add_argument("baseline")
    s.add_argument("candidate")
    s.add_argument("--margin", type=float, default=0.05,
                   help="allowed worsening for rates, in percentage points as a fraction "
                        "(0.05 = 5 points, e.g. 10%% -> 15%%)")
    s.add_argument("--harm-margin", type=float, default=0.05,
                   help="allowed worsening of the average harm score (0-1 scale)")
    s.add_argument("--alpha", type=float, default=0.05)
    s.add_argument("--blocking", nargs="+", default=list(stats.BLOCKING_DEFAULT),
                   choices=list(stats.METRICS), help="measures that can block a change")
    s.add_argument("--strict", action="store_true",
                   help="also block a PASS when answers were cut short or too long for the judge")
    s.add_argument("--json", help="also write the full result to this file")
    s.set_defaults(func=cmd_compare)

    s = sub.add_parser("check-config", help="setup-mistake check for an Ollama model")
    s.add_argument("model")
    s.add_argument("--url", default="http://localhost:11434")
    s.add_argument("--save-reference", help="save a snapshot, e.g. refs/gemma4-e4b.json")
    s.add_argument("--reference", help="compare against a saved snapshot")
    s.set_defaults(func=cmd_check_config)

    s = sub.add_parser("label", help="hand-check a sample of answers (refusal and harm)")
    s.add_argument("run")
    s.add_argument("-n", type=int, default=30)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_label)

    s = sub.add_parser("agreement", help="judge vs your labels, pooled over one or more runs")
    s.add_argument("runs", nargs="+")
    s.set_defaults(func=cmd_agreement)

    s = sub.add_parser("simulate", help="test the gate on made-up data with known effects")
    s.add_argument("--n", type=int, nargs="+", default=[50, 100, 200, 300], help="questions per comparison")
    s.add_argument("--margin", type=float, default=0.05)
    s.add_argument("--alpha", type=float, default=0.05 / 3,
                   help="per-measure alpha (default 0.05 split over 3 blocking measures)")
    s.add_argument("--flip", type=float, default=0.03, help="background rate of answers changing")
    s.add_argument("--trials", type=int, default=2000)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_simulate)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
