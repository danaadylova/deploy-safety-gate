"""Configuration check (Proposal D, layer 1): catch setup mistakes without running the model.

What it does today, for a model served by Ollama:
  1. Reads the model's chat template, default system prompt, default settings
     and file details (format, compression level) from Ollama.
  2. Warns about things known to silently change behaviour:
       - settings Ollama applies on top of the model (e.g. repeat_penalty),
       - no default system prompt,
       - a template that ignores system messages,
       - a start marker (BOS) written into the template, which can be doubled
         if something downstream adds another one.
  3. With --reference, compares everything against a saved snapshot and lists
     every difference. With --save-reference, saves a snapshot.

Exit code: 0 no differences from the reference (warnings are printed but don't fail),
1 differences found.

Planned next: compare against the model's official Hugging Face chat template,
and count start markers in the exact tokens vLLM / llama.cpp send.
"""

from __future__ import annotations

import json
from pathlib import Path

from .engines import OllamaEngine

# Settings that change how answers are picked. If the serving program sets them
# by default, every answer is affected, usually without anyone noticing.
SAMPLING_KEYS = {
    "temperature", "top_k", "top_p", "min_p", "typical_p", "repeat_penalty",
    "repeat_last_n", "presence_penalty", "frequency_penalty", "mirostat",
    "mirostat_eta", "mirostat_tau", "num_ctx", "num_predict", "stop", "seed",
}

BOS_MARKERS = ["<bos>", "<|begin_of_text|>", "<s>", "<|startoftext|>"]


def parse_parameters(text: str | None) -> dict[str, list[str]]:
    """Ollama returns parameters as text lines like 'temperature 0.7' or 'stop "<eos>"'."""
    params: dict[str, list[str]] = {}
    for line in (text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        key, _, value = line.partition(" ")
        params.setdefault(key, []).append(value.strip())
    return params


def snapshot(engine: OllamaEngine, model: str) -> dict:
    info = engine.show(model)
    details = info.get("details") or {}
    return {
        "model": model,
        "engine": "ollama",
        "engine_version": engine.version(),
        "format": details.get("format"),
        "family": details.get("family"),
        "parameter_size": details.get("parameter_size"),
        "quantization_level": details.get("quantization_level"),
        "template": info.get("template") or "",
        "system": info.get("system") or "",
        "parameters": parse_parameters(info.get("parameters")),
    }


def warnings_for(snap: dict) -> list[str]:
    out = []
    sampling = {k: v for k, v in snap["parameters"].items() if k in SAMPLING_KEYS and k != "stop"}
    if sampling:
        pretty = ", ".join(f"{k}={' '.join(v)}" for k, v in sorted(sampling.items()))
        out.append(f"default settings applied to every request: {pretty}. "
                   "Check they match the model's official recommendations.")
    if not snap["system"].strip():
        out.append("no default system prompt. If the model's official setup uses one, "
                   "it is missing here.")
    template = snap["template"]
    if template and "System" not in template and "system" not in template:
        out.append("the chat template doesn't appear to handle system messages; "
                   "a system prompt you send may be silently dropped.")
    for marker in BOS_MARKERS:
        if marker in template:
            out.append(f"the template writes the start marker {marker!r} itself. If the "
                       "tokenizer also adds one, the model sees it twice.")
            break
    if not template:
        out.append("no chat template found: requests may be sent as plain text.")
    if snap.get("quantization_level"):
        out.append(f"weights are compressed: {snap['quantization_level']} "
                   f"({snap.get('format') or 'unknown format'}). Compare against a less "
                   "compressed version before trusting safety results.")
    return out


def diff(reference: dict, current: dict) -> list[str]:
    changes = []
    for key in ("engine_version", "format", "family", "parameter_size", "quantization_level",
                "system", "template"):
        if reference.get(key) != current.get(key):
            if key in ("template", "system"):
                changes.append(f"{key} changed ({len(reference.get(key) or '')} -> "
                               f"{len(current.get(key) or '')} characters)")
            else:
                changes.append(f"{key}: {reference.get(key)!r} -> {current.get(key)!r}")
    ref_p, cur_p = reference.get("parameters", {}), current.get("parameters", {})
    for k in sorted(set(ref_p) | set(cur_p)):
        if ref_p.get(k) != cur_p.get(k):
            changes.append(f"parameter {k}: {ref_p.get(k)} -> {cur_p.get(k)}")
    return changes


def check(model: str, url: str = "http://localhost:11434", reference: Path | None = None,
          save_reference: Path | None = None, log=print) -> int:
    engine = OllamaEngine(url=url)
    snap = snapshot(engine, model)
    log(f"model {model}: {snap['family']} {snap['parameter_size']}, "
        f"{snap['quantization_level']}, {snap['format']}, Ollama {snap['engine_version']}")
    warns = warnings_for(snap)
    for w in warns:
        log(f"  WARN  {w}")
    if not warns:
        log("  no warnings")
    if save_reference:
        save_reference.parent.mkdir(parents=True, exist_ok=True)
        save_reference.write_text(json.dumps(snap, indent=2), encoding="utf-8")
        log(f"saved reference -> {save_reference}")
    if reference:
        ref = json.loads(reference.read_text(encoding="utf-8"))
        changes = diff(ref, snap)
        if changes:
            log(f"{len(changes)} difference(s) from {reference}:")
            for c in changes:
                log(f"  DIFF  {c}")
            return 1
        log(f"matches reference {reference}")
    return 0
