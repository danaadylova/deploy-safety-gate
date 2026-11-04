"""Run-level configuration check and the combined release decision.

`dsg check-config` inspects the model stored in Ollama. That misses mistakes made
in the request itself (a missing or wrong chat format sent by the calling code).
This module compares what a run ACTUALLY sent with a verified reference run, item
by item, without calling the model:

    questions            same question ids and repeats
    model                same model name and file details (format, family, size,
                         compression, modified time)
    stored model setup   template, stored settings and system prompt as Ollama reports them
    serving software     same engine type and version
    input                the exact text sent for every question. Only visible in raw
                         mode, where dsg builds the text. In chat mode the server formats
                         it, so the input is CANNOT VERIFY, never MATCH.
    system prompt        same system prompt
    generation settings  same options for every answer (temperature, seed, answer limit,
                         stop tokens, penalties, ...)
    thinking mode        same think switch

Each item is MATCH, MISMATCH or CANNOT VERIFY. A change declared in advance with
--allow ITEM is reported as ALLOWED instead of MISMATCH. Overall: MISMATCH if any
item is an undeclared MISMATCH; otherwise CANNOT VERIFY if any item can't be verified;
otherwise MATCH.

A MISMATCH means "the setup differs from the reference", not "the model is harmful".

Combined decision (`dsg gate`), from the study plan:
    APPROVE  configuration MATCH and behavior PASS
    BLOCK    an undeclared configuration MISMATCH or a behavior FAIL (says which)
    HOLD     everything else (CANNOT VERIFY, INCONCLUSIVE, INCOMPLETE)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .run import read_jsonl

ITEMS = ["questions", "model", "stored model setup", "serving software", "input",
         "system prompt", "generation settings", "thinking mode"]

# Chat-format markers from common model families, used to describe input differences.
MARKERS = ["<|begin_of_text|>", "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>",
           "<|im_start|>", "<|im_end|>", "<start_of_turn>", "<end_of_turn>", "<|user|>",
           "<|assistant|>", "<|system|>", "<|end|>", "[INST]", "[/INST]", "<s>", "</s>"]

DECISION_EXIT = {"APPROVE": 0, "BLOCK": 1, "HOLD": 2}


@dataclass
class Item:
    name: str
    status: str            # MATCH | MISMATCH | CANNOT VERIFY | ALLOWED
    detail: str = ""


@dataclass
class RunCheck:
    reference: str
    candidate: str
    verdict: str
    items: list[Item] = field(default_factory=list)

    def mismatched(self) -> list[str]:
        return [i.name for i in self.items if i.status == "MISMATCH"]

    def unverified(self) -> list[str]:
        return [i.name for i in self.items if i.status == "CANNOT VERIFY"]


def _load(run_dir: Path) -> tuple[dict, dict, dict]:
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    answers = {(r["prompt_id"], r["repeat"]): r
               for r in read_jsonl(run_dir / "responses.jsonl") if not r.get("error")}
    return cfg, meta, answers


def _sent_text(r: dict) -> str | None:
    """The exact input text, if dsg built it (raw mode). None if the server formatted it."""
    req = r.get("request") or {}
    if req.get("endpoint") == "/api/generate" and req.get("raw") is True:
        return req.get("prompt")
    return None


def _describe_text_diff(ref: str, cand: str) -> str:
    missing = [m for m in MARKERS if m in ref and m not in cand]
    extra = [m for m in MARKERS if m in cand and m not in ref]
    parts = []
    if missing:
        parts.append("reference format markers missing: " + " ".join(missing))
    if extra:
        parts.append("unexpected markers: " + " ".join(extra))
    if not parts:
        i = next((k for k, (a, b) in enumerate(zip(ref, cand)) if a != b), min(len(ref), len(cand)))
        parts.append(f"text differs from character {i}: {ref[max(0, i - 20):i + 30]!r} vs "
                     f"{cand[max(0, i - 20):i + 30]!r}")
    return "; ".join(parts)


def _same(name: str, a, b, what: str) -> Item:
    if a == b:
        return Item(name, "MATCH")
    return Item(name, "MISMATCH", f"{what}: {a!r} -> {b!r}")


def check_runs(ref_dir: Path, cand_dir: Path, allow: list[str] | tuple = ()) -> RunCheck:
    unknown = [a for a in allow if a not in ITEMS]
    if unknown:
        raise ValueError(f"unknown item(s) for --allow: {unknown}. Choose from: {ITEMS}")
    rcfg, rmeta, rans = _load(ref_dir)
    ccfg, cmeta, cans = _load(cand_dir)
    items: list[Item] = []

    # questions
    rq = (rmeta.get("question_ids"), int(rcfg.get("repeats", 1)))
    cq = (cmeta.get("question_ids"), int(ccfg.get("repeats", 1)))
    if rq == cq:
        items.append(Item("questions", "MATCH", f"{len(rq[0] or [])} questions x {rq[1]} repeats"))
    else:
        items.append(Item("questions", "MISMATCH", "different question ids or repeats"))

    # model
    rinfo, cinfo = rmeta.get("model_info") or {}, cmeta.get("model_info") or {}
    if not rinfo or not cinfo or "error" in rinfo or "error" in cinfo:
        items.append(Item("model", "CANNOT VERIFY" if rcfg["model"] == ccfg["model"] else "MISMATCH",
                          f"model file details not available; names {rcfg['model']!r} / {ccfg['model']!r}"))
        items.append(Item("stored model setup", "CANNOT VERIFY", "not reported by this serving software"))
    else:
        keys = ("details", "modified_at")
        a = (rcfg["model"],) + tuple(json.dumps(rinfo.get(k), sort_keys=True) for k in keys)
        b = (ccfg["model"],) + tuple(json.dumps(cinfo.get(k), sort_keys=True) for k in keys)
        items.append(Item("model", "MATCH", rcfg["model"]) if a == b else
                     Item("model", "MISMATCH", f"name, file details or modified time differ "
                          f"({rcfg['model']} -> {ccfg['model']})"))
        diffs = [k for k in ("template", "parameters", "system") if rinfo.get(k) != cinfo.get(k)]
        items.append(Item("stored model setup", "MISMATCH", "differs: " + ", ".join(diffs)) if diffs
                     else Item("stored model setup", "MATCH"))

    # serving software
    items.append(_same("serving software",
                       ((rcfg.get("engine") or {}).get("type", "ollama"), rmeta.get("engine_version")),
                       ((ccfg.get("engine") or {}).get("type", "ollama"), cmeta.get("engine_version")),
                       "engine type/version"))

    # per-answer comparisons, on the answers both runs have
    keys = sorted(set(rans) & set(cans))
    missing = len(set(rans) ^ set(cans))

    # input
    ref_txt = {k: _sent_text(rans[k]) for k in keys}
    cand_txt = {k: _sent_text(cans[k]) for k in keys}
    if not keys:
        items.append(Item("input", "CANNOT VERIFY", "no answers in common"))
    elif any(v is None for v in ref_txt.values()):
        items.append(Item("input", "CANNOT VERIFY",
                          "the reference run doesn't record the exact input (chat mode: the server "
                          "formats it). Use a raw-mode reference whose input was verified."))
    elif any(v is None for v in cand_txt.values()):
        items.append(Item("input", "CANNOT VERIFY",
                          "this run doesn't record the exact input (chat mode: the server formats it); "
                          "only the template is visible, which is not verified input"))
    else:
        bad = [k for k in keys if ref_txt[k] != cand_txt[k]]
        if bad:
            k = bad[0]
            items.append(Item("input", "MISMATCH",
                              f"{len(bad)}/{len(keys)} inputs differ; e.g. {k[0]}: "
                              + _describe_text_diff(ref_txt[k], cand_txt[k])))
        else:
            note = f"all {len(keys)} inputs identical"
            if missing:
                note += f" ({missing} answers present in only one run were not compared)"
            items.append(Item("input", "MATCH", note))

    # system prompt
    items.append(_same("system prompt", rcfg.get("system"), ccfg.get("system"), "system prompt"))

    # generation settings and thinking mode, from what was actually sent
    def opts(r):
        req = r.get("request") or {}
        return json.dumps({"options": req.get("options"), "format": req.get("format")}, sort_keys=True)
    bad = [k for k in keys if opts(rans[k]) != opts(cans[k])]
    if not keys:
        items.append(Item("generation settings", "CANNOT VERIFY", "no answers in common"))
    elif bad:
        ro = (rans[bad[0]].get("request") or {}).get("options") or {}
        co = (cans[bad[0]].get("request") or {}).get("options") or {}
        changed = sorted(k for k in set(ro) | set(co) if ro.get(k) != co.get(k))
        items.append(Item("generation settings", "MISMATCH",
                          f"{len(bad)}/{len(keys)} answers sent different settings: "
                          + ", ".join(f"{k} {ro.get(k)!r} -> {co.get(k)!r}" for k in changed)))
    else:
        items.append(Item("generation settings", "MATCH", "identical options for every answer"))

    rthink = {(rans[k].get("request") or {}).get("think") for k in keys}
    cthink = {(cans[k].get("request") or {}).get("think") for k in keys}
    items.append(_same("thinking mode", rthink, cthink, "think switch sent"))

    for it in items:
        if it.status == "MISMATCH" and it.name in allow:
            it.status, it.detail = "ALLOWED", "declared change: " + it.detail
    if any(i.status == "MISMATCH" for i in items):
        verdict = "MISMATCH"
    elif any(i.status == "CANNOT VERIFY" for i in items):
        verdict = "CANNOT VERIFY"
    else:
        verdict = "MATCH"
    return RunCheck(ref_dir.name, cand_dir.name, verdict, items)


def print_check(rc: RunCheck, log=print) -> None:
    log(f"configuration check (no model calls)\nreference {rc.reference}\ncandidate {rc.candidate}\n")
    for i in rc.items:
        log(f"  {i.name:20s} {i.status:14s} {i.detail}")
    log(f"\nCONFIGURATION: {rc.verdict}")
    if rc.verdict == "MISMATCH":
        log("  (the setup differs from the reference; this is not by itself evidence of harm)")


def decide(config_verdict: str, behavior_verdict: str, mismatched: list[str]) -> tuple[str, str]:
    """The study's combined release rule. Returns (decision, reason)."""
    triggers = []
    if config_verdict == "MISMATCH":
        triggers.append(f"configuration mismatch ({', '.join(mismatched)})")
    if behavior_verdict == "FAIL":
        triggers.append("measured safety worsening (behavior FAIL)")
    if triggers:
        return "BLOCK", "block for review, triggered by: " + "; ".join(triggers)
    if config_verdict == "MATCH" and behavior_verdict == "PASS":
        return "APPROVE", "configuration MATCH and behavior PASS"
    waiting = []
    if config_verdict != "MATCH":
        waiting.append(f"configuration {config_verdict}")
    if behavior_verdict != "PASS":
        waiting.append(f"behavior {behavior_verdict}")
    return "HOLD", "hold for more evidence: " + ", ".join(waiting)


def print_decision(rc: RunCheck, behavior_verdict: str, log=print) -> str:
    decision, reason = decide(rc.verdict, behavior_verdict, rc.mismatched())
    harm = {"FAIL": "worsening shown beyond the margin", "PASS": "no worsening beyond the margin",
            }.get(behavior_verdict, "not shown either way")
    log("\n" + "=" * 70)
    log(f"  setup compliance        {rc.verdict}")
    log(f"  measured safety change  {behavior_verdict} ({harm})")
    log(f"  RELEASE DECISION        {decision}: {reason}")
    if decision == "BLOCK" and behavior_verdict != "FAIL":
        log("  note: blocked for a setup difference, not for measured harm")
    log("=" * 70)
    return decision
