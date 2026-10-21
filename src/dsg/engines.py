"""Talk to the program that serves the model.

Two engines:
  ollama        Ollama's own API (what's on the Mac today). Supports "raw" mode,
                which skips the chat template - useful for planting setup mistakes.
  openai        Any OpenAI-compatible server: vLLM, llama.cpp's llama-server,
                LM Studio. Use this for the vLLM vs llama.cpp comparison later.

Every call returns a dict with the answer text, the exact request that was sent,
and timing numbers (for the speed/cost side of each comparison).
"""

from __future__ import annotations

import time

import requests

# Ollama option names for common generation settings.
_OLLAMA_OPTION_NAMES = {
    "temperature": "temperature",
    "top_p": "top_p",
    "top_k": "top_k",
    "min_p": "min_p",
    "seed": "seed",
    "max_tokens": "num_predict",
    "repeat_penalty": "repeat_penalty",
    "num_ctx": "num_ctx",
}


class EngineError(RuntimeError):
    pass


class OllamaEngine:
    """Ollama's native API. mode is one of:

    chat              normal use: Ollama applies the model's chat template
    raw_no_template   send the bare question with no template at all (a planted mistake)
    raw_custom        wrap the question in engine.raw_template, e.g. a different
                      model family's template (another planted mistake)
    """

    def __init__(self, url: str = "http://localhost:11434", mode: str = "chat",
                 raw_template: str | None = None, think: bool | None = None,
                 timeout: float = 600):
        if mode not in ("chat", "raw_no_template", "raw_custom"):
            raise ValueError(f"unknown ollama mode: {mode}")
        if mode == "raw_custom" and not raw_template:
            raise ValueError("mode raw_custom needs engine.raw_template containing {prompt}")
        self.url = url.rstrip("/")
        self.mode = mode
        self.raw_template = raw_template
        self.think = think
        self.timeout = timeout

    # --- model info (used by config checks and run metadata) ---
    def version(self) -> str | None:
        try:
            r = requests.get(f"{self.url}/api/version", timeout=10)
            r.raise_for_status()
            return r.json().get("version")
        except requests.RequestException:
            return None

    def show(self, model: str) -> dict:
        r = requests.post(f"{self.url}/api/show", json={"model": model}, timeout=30)
        if r.status_code != 200:
            raise EngineError(f"/api/show failed for {model}: {r.status_code} {r.text[:300]}")
        return r.json()

    # --- generation ---
    def _options(self, gen: dict) -> dict:
        opts = {}
        for key, value in gen.items():
            if value is None:
                continue
            opts[_OLLAMA_OPTION_NAMES.get(key, key)] = value
        return opts

    def generate(self, model: str, prompt: str, system: str | None, gen: dict,
                 fmt: dict | str | None = None) -> dict:
        options = self._options(gen)
        if self.mode == "chat":
            messages = []
            if system is not None:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            payload = {"model": model, "messages": messages, "stream": False, "options": options}
            endpoint = "/api/chat"
        else:
            text = prompt if self.mode == "raw_no_template" else self.raw_template.replace("{prompt}", prompt)
            if system is not None and self.mode == "raw_no_template":
                text = f"{system}\n\n{text}"
            payload = {"model": model, "prompt": text, "raw": True, "stream": False, "options": options}
            endpoint = "/api/generate"
        if self.think is not None and self.mode == "chat":
            payload["think"] = self.think
        if fmt is not None:
            payload["format"] = fmt

        start = time.monotonic()
        r = requests.post(f"{self.url}{endpoint}", json=payload, timeout=self.timeout)
        wall = time.monotonic() - start
        if r.status_code == 400 and "think" in payload and "think" in r.text.lower():
            # The model doesn't support the think switch; retry without it.
            payload.pop("think")
            start = time.monotonic()
            r = requests.post(f"{self.url}{endpoint}", json=payload, timeout=self.timeout)
            wall = time.monotonic() - start
        if r.status_code != 200:
            raise EngineError(f"{endpoint} {r.status_code}: {r.text[:300]}")
        data = r.json()

        if endpoint == "/api/chat":
            msg = data.get("message", {})
            text_out, thinking = msg.get("content", ""), msg.get("thinking")
        else:
            text_out, thinking = data.get("response", ""), data.get("thinking")

        eval_count = data.get("eval_count")
        eval_ns = data.get("eval_duration")
        return {
            "response": text_out,
            "thinking": thinking,
            "request": {"endpoint": endpoint, **payload},
            "timing": {
                "wall_s": round(wall, 3),
                "prompt_tokens": data.get("prompt_eval_count"),
                "output_tokens": eval_count,
                "tokens_per_s": round(eval_count / (eval_ns / 1e9), 2) if eval_count and eval_ns else None,
                "load_s": round(data["load_duration"] / 1e9, 3) if data.get("load_duration") else None,
            },
            "done_reason": data.get("done_reason"),
        }


class OpenAICompatEngine:
    """vLLM, llama.cpp's llama-server, LM Studio, etc. (POST /v1/chat/completions)."""

    def __init__(self, url: str = "http://localhost:8000", api_key: str | None = None,
                 timeout: float = 600):
        self.url = url.rstrip("/")
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.timeout = timeout

    def version(self) -> str | None:
        return None

    def generate(self, model: str, prompt: str, system: str | None, gen: dict,
                 fmt: dict | str | None = None) -> dict:
        messages = []
        if system is not None:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload = {"model": model, "messages": messages}
        for key, value in gen.items():
            if value is not None:
                payload[key] = value
        if fmt is not None:
            payload["response_format"] = {"type": "json_object"}
        start = time.monotonic()
        r = requests.post(f"{self.url}/v1/chat/completions", json=payload,
                          headers=self.headers, timeout=self.timeout)
        wall = time.monotonic() - start
        if r.status_code != 200:
            raise EngineError(f"/v1/chat/completions {r.status_code}: {r.text[:300]}")
        data = r.json()
        choice = data["choices"][0]
        usage = data.get("usage") or {}
        out_tokens = usage.get("completion_tokens")
        return {
            "response": choice["message"].get("content") or "",
            "thinking": choice["message"].get("reasoning_content"),
            "request": {"endpoint": "/v1/chat/completions", **payload},
            "timing": {
                "wall_s": round(wall, 3),
                "prompt_tokens": usage.get("prompt_tokens"),
                "output_tokens": out_tokens,
                # Includes queueing and prompt time, so it understates pure generation speed.
                "tokens_per_s": round(out_tokens / wall, 2) if out_tokens and wall else None,
                "load_s": None,
            },
            "done_reason": choice.get("finish_reason"),
        }


def make_engine(cfg: dict):
    cfg = dict(cfg or {})
    kind = cfg.pop("type", "ollama")
    if kind == "ollama":
        return OllamaEngine(**cfg)
    if kind == "openai":
        return OpenAICompatEngine(**cfg)
    raise ValueError(f"unknown engine type: {kind}")
