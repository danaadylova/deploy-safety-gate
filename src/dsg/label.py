"""Hand-check the judge (the "is the judge right?" step).

`dsg label <run>` shows a random sample of answers one at a time and asks whether
the model refused. Your labels go to outputs/<run>/human_labels.csv. Then it
prints how often the judge agreed with you, split by kind of mistake, so you
know how much to trust the judge's numbers.
"""

from __future__ import annotations

import csv
import random
import textwrap
from pathlib import Path

from .judge import load_scored

LABELS_FILE = "human_labels.csv"


def _wrap(text: str, width: int = 100, max_lines: int = 40) -> str:
    lines = []
    for para in (text or "").splitlines():
        lines.extend(textwrap.wrap(para, width) or [""])
    if len(lines) > max_lines:
        lines = lines[:max_lines] + [f"... ({len(lines) - max_lines} more lines)"]
    return "\n".join(lines)


def read_labels(run_dir: Path) -> dict[tuple[str, int], int]:
    path = run_dir / LABELS_FILE
    if not path.exists():
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        return {(r["prompt_id"], int(r["repeat"])): int(r["human_refused"]) for r in csv.DictReader(f)}


def label(run_dir: Path, n: int = 30, seed: int = 0, ask=input, log=print) -> None:
    rows = load_scored(run_dir)
    existing = read_labels(run_dir)
    pool = [r for r in rows if (r["prompt_id"], r["repeat"]) not in existing]
    random.Random(seed).shuffle(pool)
    sample = pool[:n]
    path = run_dir / LABELS_FILE
    new_file = not path.exists()
    log(f"{len(existing)} already labelled. Labelling {len(sample)} more. "
        "Answer y = refused / no real help, n = actually helped, s = skip, q = quit.")
    log("The judge's label is hidden so it doesn't sway you.\n")
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["prompt_id", "repeat", "human_refused"])
        for i, r in enumerate(sample, 1):
            log("=" * 100)
            log(f"[{i}/{len(sample)}] {r['prompt_id']} ({r['expected']} expected, {r['category']})")
            log("QUESTION:\n" + _wrap(r["prompt"]))
            log("-" * 100)
            log("ANSWER:\n" + _wrap(r["response"]))
            while True:
                a = ask("refused? [y/n/s/q] ").strip().lower()
                if a in ("y", "n", "s", "q"):
                    break
            if a == "q":
                break
            if a == "s":
                continue
            w.writerow([r["prompt_id"], r["repeat"], 1 if a == "y" else 0])
            f.flush()
    agreement(run_dir, log=log)


def agreement(run_dir: Path, log=print) -> dict | None:
    labels = read_labels(run_dir)
    if not labels:
        log("no human labels yet")
        return None
    judged = {(r["prompt_id"], r["repeat"]): r["refused"] for r in load_scored(run_dir)}
    pairs = [(labels[k], judged[k]) for k in labels if k in judged]
    if not pairs:
        log("no overlap between labels and judgments")
        return None
    agree = sum(h == j for h, j in pairs)
    missed = sum(h == 1 and j == 0 for h, j in pairs)       # judge said helped, you said refused
    false_ref = sum(h == 0 and j == 1 for h, j in pairs)    # judge said refused, you said helped
    res = {"n": len(pairs), "agreement": agree / len(pairs),
           "judge_said_helped_but_refused": missed, "judge_said_refused_but_helped": false_ref}
    log(f"\njudge agreed with you on {agree}/{len(pairs)} ({res['agreement']:.0%}). "
        f"Judge said 'helped' when you said 'refused': {missed}. "
        f"Judge said 'refused' when you said 'helped': {false_ref}.")
    return res
