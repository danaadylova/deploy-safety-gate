"""Configuration check (Proposal D, layer 1): catch setup mistakes without running the model.

What it does today, for a model served by Ollama:
  1. Reads the model's chat template, Modelfile (RENDERER / PARSER / CAPABILITY
     lines), default system prompt, default settings and file details (format,
     compression level) from Ollama.
  2. Reports the chat formatting as "visible" or "cannot verify". It is "cannot
     verify" when the template is only a placeholder and the formatting is done by
     Ollama's built-in code (a RENDERER line), as for gemma4. It never assumes
     hidden formatting is correct.
  3. Warns about things known to silently change behaviour:
       - settings stored with the model (e.g. temperature, repeat_penalty),
       - built-in engine defaults it can't see (so configs must set them),
       - output post-processing (PARSER) and thinking mode,
       - no default system prompt; a visible template that ignores system messages,
       - a start marker (BOS) written into a visible template, which can be doubled.
  4. With --reference, compares everything against a saved snapshot and lists
     every difference. With --save-reference, saves a snapshot (licence text removed).

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


PLACEHOLDER_TEMPLATES = {"{{ .Prompt }}", "{{.Prompt}}"}


def strip_license(modelfile: str) -> str:
    """Drop the LICENSE \"\"\"...\"\"\" block: long, and not about how the model runs."""
    out, skipping = [], False
    for line in (modelfile or "").splitlines():
        if not skipping and line.startswith("LICENSE"):
            rest = line[len("LICENSE"):].strip()
            if rest.startswith('"""') and not (rest.endswith('"""') and len(rest) > 3):
                skipping = True
            out.append('LICENSE """(removed)"""')
            continue
        if skipping:
            if line.rstrip().endswith('"""'):
                skipping = False
            continue
        out.append(line)
    return "\n".join(out).strip()


def modelfile_directives(modelfile: str) -> dict:
    found = {"renderer": None, "parser": None, "capabilities": []}
    for line in (modelfile or "").splitlines():
        key, _, value = line.strip().partition(" ")
        if key == "RENDERER":
            found["renderer"] = value.strip()
        elif key == "PARSER":
            found["parser"] = value.strip()
        elif key == "CAPABILITY":
            found["capabilities"].append(value.strip())
    return found


def snapshot(engine: OllamaEngine, model: str) -> dict:
    info = engine.show(model)
    details = info.get("details") or {}
    modelfile = strip_license(info.get("modelfile") or "")
    directives = modelfile_directives(modelfile)
    return {
        "model": model,
        "engine": "ollama",
        "engine_version": engine.version(),
        "format": details.get("format"),
        "family": details.get("family"),
        "parameter_size": details.get("parameter_size"),
        "quantization_level": details.get("quantization_level"),
        "template": info.get("template") or "",
        "renderer": directives["renderer"],
        "parser": directives["parser"],
        "capabilities": sorted(directives["capabilities"] or info.get("capabilities") or []),
        "system": info.get("system") or "",
        "parameters": parse_parameters(info.get("parameters")),
        "modelfile": modelfile,
    }


def formatting_status(snap: dict) -> str:
    """'visible' if the chat format can be read from the template, else 'cannot verify'."""
    template = (snap.get("template") or "").strip()
    if snap.get("renderer") or template in PLACEHOLDER_TEMPLATES or not template:
        return "cannot verify"
    return "visible"


def warnings_for(snap: dict) -> list[str]:
    out = []
    template = (snap.get("template") or "").strip()
    hidden = formatting_status(snap) == "cannot verify"
    if hidden:
        who = (f"Ollama's built-in '{snap['renderer']}' code (RENDERER line)" if snap.get("renderer")
               else "code that isn't visible here")
        out.append(f"CANNOT VERIFY the chat formatting: the template is only a placeholder "
                   f"({template or 'empty'!r}); formatting is done by {who}. Don't assume it is correct; "
                   "check the model's official format another way.")
    if snap.get("parser"):
        out.append(f"answers are post-processed by Ollama's built-in '{snap['parser']}' parser "
                   "(e.g. separating 'thinking' and tool calls). What you receive may not be exactly "
                   "what the model wrote.")
    if "thinking" in (snap.get("capabilities") or []):
        out.append("the model can 'think' before answering. Set thinking on or off explicitly in "
                   "every config; it can change whether the model refuses.")
    sampling = {k: v for k, v in snap["parameters"].items() if k in SAMPLING_KEYS and k != "stop"}
    if sampling:
        pretty = ", ".join(f"{k}={' '.join(v)}" for k, v in sorted(sampling.items()))
        out.append(f"default settings applied to every request: {pretty}. "
                   "Check they match the model's official recommendations.")
    if not snap["system"].strip():
        out.append("no default system prompt. If the model's official setup uses one, "
                   "it is missing here.")
    if not hidden:
        if "System" not in template and "system" not in template:
            out.append("the chat template doesn't appear to handle system messages; "
                       "a system prompt you send may be silently dropped.")
        for marker in BOS_MARKERS:
            if marker in template:
                out.append(f"the template writes the start marker {marker!r} itself. If the "
                           "tokenizer also adds one, the model sees it twice.")
                break
    out.append("built-in engine defaults that aren't listed above (e.g. a repetition penalty) "
               "can't be seen here: set every sampling setting explicitly in the run configs.")
    if snap.get("quantization_level"):
        out.append(f"weights are compressed: {snap['quantization_level']} "
                   f"({snap.get('format') or 'unknown format'}). Compare against a less "
                   "compressed version before trusting safety results.")
    return out


def diff(reference: dict, current: dict) -> list[str]:
    changes = []
    for key in ("engine_version", "format", "family", "parameter_size", "quantization_level",
                "renderer", "parser", "capabilities", "system", "template", "modelfile"):
        if reference.get(key) != current.get(key):
            if key in ("template", "system", "modelfile"):
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
    log(f"  chat formatting: {formatting_status(snap)}"
        + (f" (renderer {snap['renderer']}, parser {snap['parser']})" if snap.get("renderer") else ""))
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
