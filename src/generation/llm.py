"""Thin LLM client. One function, two backends, no framework.

Phase 5's brief needs a generator before Phase 6 builds the answer path, so the
transport lives here and stays deliberately small: send a system+user prompt, get
text back. Prompt templates (src/generation/prompts.py), citation enforcement and
answer assembly are Phase 6's job and do not belong in here.

`available()` is a real capability check, not a config read, because the interesting
failure is "Ollama is configured but not running" — every caller must be able to
degrade rather than crash. The check is cached per process: it costs a socket
connection, and callers loop over many events.
"""
import json
import urllib.error
import urllib.request

from src.common.config import CFG

_AVAILABLE: bool | None = None


class LLMUnavailable(RuntimeError):
    pass


def backend() -> str:
    return CFG["generation"].get("backend", "ollama")


def _ollama_url(path: str) -> str:
    return CFG["generation"]["ollama_url"].rstrip("/") + path


def available(refresh: bool = False) -> bool:
    """True when the configured backend can actually serve a request right now."""
    global _AVAILABLE
    if _AVAILABLE is not None and not refresh:
        return _AVAILABLE
    try:
        if backend() == "ollama":
            with urllib.request.urlopen(_ollama_url("/api/tags"), timeout=3) as r:
                _AVAILABLE = r.status == 200
        else:
            import os
            _AVAILABLE = bool(os.environ.get("ANTHROPIC_API_KEY"))
    except Exception:
        _AVAILABLE = False
    return _AVAILABLE


def complete(system: str, user: str, temperature: float | None = None,
             timeout: int = 120) -> str:
    """Single-turn completion. Raises LLMUnavailable so callers can fall back."""
    temp = CFG["generation"].get("temperature", 0.1) if temperature is None else temperature
    if not available():
        raise LLMUnavailable(
            f"backend {backend()!r} unreachable "
            f"({CFG['generation'].get('ollama_url')}) — start Ollama or set backend: anthropic")

    if backend() == "ollama":
        payload = {
            "model": CFG["generation"]["ollama_model"],
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "stream": False,
            "options": {"temperature": temp},
        }
        req = urllib.request.Request(
            _ollama_url("/api/chat"), data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())["message"]["content"].strip()
        except urllib.error.URLError as e:
            raise LLMUnavailable(f"ollama request failed: {e}") from e

    from anthropic import Anthropic          # optional dependency, imported on use
    resp = Anthropic().messages.create(
        model=CFG["generation"].get("anthropic_model", "claude-sonnet-5"),
        max_tokens=1024, temperature=temp, system=system,
        messages=[{"role": "user", "content": user}],
    )
    return "".join(b.text for b in resp.content if b.type == "text").strip()
