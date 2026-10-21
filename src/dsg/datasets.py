"""Download the question sets and load them in one shared format.

Every question becomes a dict:
    id        stable id, e.g. "xstest:12" or "strongreject:40"
    dataset   "xstest" or "strongreject"
    prompt    the text sent to the model
    expected  "refuse" (harmful: the model should decline)
              or "comply" (harmless: the model should answer)
    category  the dataset's own category label
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import requests

SOURCES = {
    "xstest": "https://raw.githubusercontent.com/paul-rottger/xstest/main/xstest_prompts.csv",
    "strongreject": (
        "https://raw.githubusercontent.com/alexandrasouly/strongreject/main/"
        "strongreject_dataset/strongreject_dataset.csv"
    ),
}

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
PROMPTS_FILE = DATA_DIR / "prompts.jsonl"


def fetch(raw_dir: Path = RAW_DIR, timeout: float = 60) -> dict[str, Path]:
    """Download each source CSV into raw_dir. Returns {dataset: path}."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, url in SOURCES.items():
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        path = raw_dir / f"{name}.csv"
        path.write_bytes(resp.content)
        paths[name] = path
    return paths


def _read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def parse_xstest(path: Path) -> list[dict]:
    rows = []
    for r in _read_csv(path):
        rows.append(
            {
                "id": f"xstest:{r['id']}",
                "dataset": "xstest",
                "prompt": r["prompt"],
                # XSTest labels: "safe" = harmless question that only sounds risky.
                "expected": "comply" if r["label"].strip() == "safe" else "refuse",
                "category": r["type"],
            }
        )
    return rows


def parse_strongreject(path: Path) -> list[dict]:
    rows = []
    for i, r in enumerate(_read_csv(path)):
        rows.append(
            {
                "id": f"strongreject:{i}",
                "dataset": "strongreject",
                "prompt": r["forbidden_prompt"],
                "expected": "refuse",
                "category": r["category"],
            }
        )
    return rows


PARSERS = {"xstest": parse_xstest, "strongreject": parse_strongreject}


def build_prompts(raw_dir: Path = RAW_DIR, out: Path = PROMPTS_FILE) -> int:
    """Parse all raw CSVs into one prompts.jsonl. Returns the number of prompts."""
    rows = []
    for name, parse in PARSERS.items():
        rows.extend(parse(raw_dir / f"{name}.csv"))
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows)


def load_prompts(path: Path = PROMPTS_FILE) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `dsg fetch-data` first.")
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def select(prompts: list[dict], counts: dict[str, int], seed: int = 0) -> list[dict]:
    """Pick a fixed random subset per dataset.

    counts maps dataset -> how many (0 or missing = skip, -1 = all).
    For XSTest the subset is split evenly between harmless and harmful questions.
    The same seed always gives the same questions, so runs can be paired.
    """
    rng = random.Random(seed)
    chosen = []
    for name, n in counts.items():
        pool = sorted((p for p in prompts if p["dataset"] == name), key=lambda p: p["id"])
        if not n:
            continue
        if n < 0 or n >= len(pool):
            chosen.extend(pool)
            continue
        if name == "xstest":
            comply = [p for p in pool if p["expected"] == "comply"]
            refuse = [p for p in pool if p["expected"] == "refuse"]
            half = n // 2
            chosen.extend(rng.sample(comply, min(n - half, len(comply))))
            chosen.extend(rng.sample(refuse, min(half, len(refuse))))
        else:
            chosen.extend(rng.sample(pool, n))
    return sorted(chosen, key=lambda p: p["id"])
