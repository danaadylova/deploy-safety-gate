"""Command line: dsg <command> ...   (run `dsg <command> -h` for details)"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

from . import config_check, datasets, judge, label, stats
from .run import OUTPUTS, run


def _run_dir(arg: str) -> Path:
    p = Path(arg)
    if not p.exists():
        p = OUTPUTS / arg
    if not (p / "responses.jsonl").exists():
        sys.exit(f"no run found at {arg} (looked for {p / 'responses.jsonl'})")
    return p


def _speed(rows: list[dict]) -> str:
    tps = [r["timing"]["tokens_per_s"] for r in rows if r.get("timing", {}).get("tokens_per_s")]
    wall = [r["timing"]["wall_s"] for r in rows if r.get("timing", {}).get("wall_s")]
    if not tps:
        return "speed: not recorded"
    return (f"speed: median {np.median(tps):.1f} tokens/s, median {np.median(wall):.2f} s per answer "
            f"({len(tps)} answers)")


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
    print(f"run {run_dir.name}: {len(rows)} judged answers")
    for metric in stats.METRICS:
        s = stats.summarize(stats.per_question(rows, metric), metric)
        if s is None:
            print(f"  {metric:14s} not enough questions")
            continue
        line = (f"  {metric:14s} {s.mean:6.1%}  95% CI [{s.ci[0]:.1%}, {s.ci[1]:.1%}]  "
                f"{s.n_questions} questions, {s.n_answers} answers")
        if s.flip_rate is not None:
            line += (f"  | answer flipped across repeats on {s.flip_rate:.0%} of questions; "
                     f"noise from question choice {s.between_question_var:.4f} vs "
                     f"model randomness {s.within_question_var:.4f}")
        print(line)
    print("  " + _speed(rows))


def cmd_compare(args):
    base_dir, cand_dir = _run_dir(args.baseline), _run_dir(args.candidate)
    base, cand = judge.load_scored(base_dir), judge.load_scored(cand_dir)
    metrics = args.metrics or list(stats.METRICS)
    alpha = args.alpha / len(metrics)  # Bonferroni: several metrics share the error budget
    print(f"baseline  {base_dir.name}\ncandidate {cand_dir.name}")
    print(f"margin {args.margin:.1%}; {1 - 2 * alpha:.1%} intervals (alpha {args.alpha} "
          f"split over {len(metrics)} metrics)")
    print("'worse by' = change in the bad direction (negative = it got better)\n")
    results = []
    for m in metrics:
        c = stats.compare(stats.per_question(base, m), stats.per_question(cand, m), m,
                          margin=args.margin, alpha=alpha)
        results.append(c)
        if c.decision == "UNMEASURED":
            print(f"  {m:14s} UNMEASURED  {'; '.join(c.notes)}")
            continue
        p = f"  McNemar p={c.mcnemar_p:.3g}" if c.mcnemar_p is not None else ""
        print(f"  {m:14s} {c.decision:12s} {c.baseline:6.1%} -> {c.candidate:6.1%}  "
              f"worse by {c.worse:+.1%} [{c.ci[0]:+.1%}, {c.ci[1]:+.1%}]  n={c.n_paired}{p}")
        for note in c.notes:
            print(f"  {'':14s} - {note}")
    print("\n  " + "baseline  " + _speed(base))
    print("  " + "candidate " + _speed(cand))
    verdict = stats.overall([c.decision for c in results])
    print(f"\nGATE: {verdict}")
    if args.json:
        Path(args.json).write_text(json.dumps(
            {"baseline": base_dir.name, "candidate": cand_dir.name, "verdict": verdict,
             "results": [asdict(c) for c in results]}, indent=2), encoding="utf-8")
    sys.exit(stats.EXIT_CODES[verdict])


def cmd_check_config(args):
    code = config_check.check(args.model, url=args.url,
                              reference=Path(args.reference) if args.reference else None,
                              save_reference=Path(args.save_reference) if args.save_reference else None)
    sys.exit(code)


def cmd_label(args):
    label.label(_run_dir(args.run), n=args.n, seed=args.seed)


def cmd_agreement(args):
    label.agreement(_run_dir(args.run))


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

    s = sub.add_parser("summarize", help="rates, error bars, noise and speed for one run")
    s.add_argument("run")
    s.set_defaults(func=cmd_summarize)

    s = sub.add_parser("compare", help="the gate: paired comparison of two runs")
    s.add_argument("baseline")
    s.add_argument("candidate")
    s.add_argument("--margin", type=float, default=0.05,
                   help="smallest regression you care about, as a fraction (default 0.05 = 5 points)")
    s.add_argument("--alpha", type=float, default=0.05)
    s.add_argument("--metrics", nargs="+", choices=list(stats.METRICS))
    s.add_argument("--json", help="also write the result to this file")
    s.set_defaults(func=cmd_compare)

    s = sub.add_parser("check-config", help="setup-mistake check for an Ollama model")
    s.add_argument("model")
    s.add_argument("--url", default="http://localhost:11434")
    s.add_argument("--save-reference", help="save a snapshot, e.g. refs/gemma4-e4b.json")
    s.add_argument("--reference", help="compare against a saved snapshot")
    s.set_defaults(func=cmd_check_config)

    s = sub.add_parser("label", help="hand-check a sample of answers to measure judge accuracy")
    s.add_argument("run")
    s.add_argument("-n", type=int, default=30)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_label)

    s = sub.add_parser("agreement", help="judge vs your labels")
    s.add_argument("run")
    s.set_defaults(func=cmd_agreement)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
