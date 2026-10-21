"""A tiny stand-in for Ollama so the pipeline can be tested without a model.

Behaviour:
- In chat mode, the "model" refuses any question containing a word from REFUSE_WORDS.
- In raw mode (no template), it refuses less: only questions containing "bomb".
  This mimics a setup mistake weakening safety, so the gate should FAIL.
- Judge requests (those with a JSON `format`) are graded by looking for "can't".
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

REFUSE_WORDS = ("kill", "bomb", "steal", "hack", "drug", "weapon", "poison", "attack")


def _answer(prompt: str, strict: bool) -> str:
    words = REFUSE_WORDS if strict else ("bomb",)
    if any(w in prompt.lower() for w in words):
        return "Sorry, I can't help with that."
    return "Sure. Here are detailed, specific steps: first do this, then do that."


def _judge(prompt: str) -> str:
    answer = prompt.split("ANSWER from the assistant:", 1)[-1]
    refused = 1 if "can't" in answer else 0
    return json.dumps({"refused": refused, "convincing": 1 if refused else 4,
                       "specific": 1 if refused else 4, "reason": "test"})


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/version":
            return self._send({"version": "0.0-fake"})
        self._send({"error": "not found"}, 404)

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        timing = {"eval_count": 20, "eval_duration": 1_000_000_000, "prompt_eval_count": 30,
                  "load_duration": 1_000_000, "done_reason": "stop"}
        if self.path == "/api/show":
            return self._send({
                "template": "<bos>{{ range .Messages }}<start_of_turn>{{ .Role }}\n{{ .Content }}"
                            "<end_of_turn>\n{{ end }}<start_of_turn>model\n",
                "system": "",
                "parameters": 'temperature 1\nrepeat_penalty 1.1\nstop "<end_of_turn>"',
                "details": {"format": "gguf", "family": "gemma", "parameter_size": "4B",
                            "quantization_level": "Q4_K_M"},
            })
        if self.path == "/api/chat":
            prompt = req["messages"][-1]["content"]
            text = _judge(prompt) if "format" in req else _answer(prompt, strict=True)
            return self._send({"message": {"role": "assistant", "content": text}, **timing})
        if self.path == "/api/generate":
            return self._send({"response": _answer(req["prompt"], strict=False), **timing})
        self._send({"error": "not found"}, 404)


def start() -> tuple[HTTPServer, str]:
    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_port}"
