"""Run one configuration: send every selected question to the model, save every answer.

Output: outputs/<run_id>/
    config.yaml      the exact config used
    meta.json        engine version, model details, question ids
    responses.jsonl  one line per (question, repeat)

Runs can be resumed: rerunning the same --run-id skips answers already saved.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

from . import datasets
from .engines import EngineError, OllamaEngine, make_engine

OUTPUTS = Path("outputs")


def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for key in ("name", "model"):
        if key not in cfg:
            raise ValueError(f"config {path} is missing '{key}'")
    cfg.setdefault("engine", {"type": "ollama"})
    cfg.setdefault("generation", {})
    cfg.setdefault("system", None)
    cfg.setdefault("repeats", 1)
    cfg.setdefault("data", {"xstest": 50, "strongreject": 50, "seed": 0})
    return cfg


def _model_details(engine, model: str) -> dict:
    if not isinstance(engine, OllamaEngine):
        return {}
    try:
        info = engine.show(model)
    except EngineError as e:
        return {"error": str(e)}
    return {
        "details": info.get("details"),
        "parameters": info.get("parameters"),
        "template": info.get("template"),
        "system": info.get("system"),
        "modified_at": info.get("modified_at"),
    }


def _done_keys(path: Path) -> set[tuple[str, int]]:
    if not path.exists():
        return set()
    keys = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if not r.get("error"):
                    keys.add((r["prompt_id"], r["repeat"]))
    return keys


def run(config_path: Path, run_id: str | None = None, prompts_file: Path = datasets.PROMPTS_FILE,
        outputs: Path = OUTPUTS, log=print) -> Path:
    cfg = load_config(config_path)
    run_id = run_id or f"{cfg['name']}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    out_dir = outputs / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    data_cfg = dict(cfg["data"])
    seed = data_cfg.pop("seed", 0)
    questions = datasets.select(datasets.load_prompts(prompts_file), data_cfg, seed=seed)

    engine = make_engine(cfg["engine"])
    (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    meta = {
        "run_id": run_id,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "engine_version": engine.version(),
        "model": cfg["model"],
        "model_info": _model_details(engine, cfg["model"]),
        "question_ids": [q["id"] for q in questions],
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    responses = out_dir / "responses.jsonl"
    done = _done_keys(responses)
    gen = dict(cfg["generation"])
    base_seed = gen.pop("seed", 0)
    total = len(questions) * cfg["repeats"]
    log(f"run {run_id}: {len(questions)} questions x {cfg['repeats']} repeats = {total} answers"
        f" ({len(done)} already saved)")

    t0 = time.monotonic()
    n_new = n_err = 0
    with open(responses, "a", encoding="utf-8") as f:
        for q in questions:
            for rep in range(cfg["repeats"]):
                if (q["id"], rep) in done:
                    continue
                # A different but fixed seed per repeat: repeats differ, reruns match.
                gen_rep = {**gen, "seed": base_seed + rep}
                record = {
                    "run_id": run_id, "prompt_id": q["id"], "dataset": q["dataset"],
                    "expected": q["expected"], "category": q["category"], "prompt": q["prompt"],
                    "repeat": rep, "model": cfg["model"],
                }
                try:
                    record.update(engine.generate(cfg["model"], q["prompt"], cfg["system"], gen_rep))
                    record["error"] = None
                except Exception as e:  # keep going; failed answers are retried on resume
                    record.update({"response": None, "error": str(e)})
                    n_err += 1
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
                n_new += 1
                if n_new % 10 == 0:
                    rate = n_new / (time.monotonic() - t0)
                    left = total - len(done) - n_new
                    log(f"  {n_new + len(done)}/{total}  (~{left / rate / 60:.1f} min left)")
    log(f"done: {n_new} new answers, {n_err} errors -> {responses}")
    return out_dir


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
