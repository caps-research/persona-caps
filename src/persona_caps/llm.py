"""Multi-backend LLM client.

Supports three backends with a uniform `chat()` interface:

- ``openai``  — OpenAI native API (e.g. gpt-4o, gpt-4o-mini)
- ``anthropic`` — Anthropic native API (e.g. claude-sonnet-4)
- ``minimax`` — MiniMax via Anthropic-compatible endpoint
  (uses the ``anthropic`` SDK pointed at ``MINIMAX_BASE_URL``)

Why the Anthropic SDK for MiniMax? Per the MiniMax platform docs
(https://platform.minimax.io/docs/guides/quickstart-sdk) the recommended
integration path is the Anthropic SDK against ``/anthropic/v1/messages``,
because MiniMax-M2.7 returns interleaved ``thinking`` + ``text`` content
blocks that the Anthropic message format models natively.
"""

from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass
from typing import Callable, Literal, TypeVar

from dotenv import load_dotenv

load_dotenv()

Backend = Literal["openai", "anthropic", "minimax", "gemini", "ollama"]

T = TypeVar("T")


def _is_transient_error(err: BaseException) -> bool:
    msg = str(err).lower()
    return any(
        s in msg
        for s in (
            "429",
            "rate limit",
            "rate_limit",
            "timeout",
            "timed out",
            "rc=28",  # curl: (28) Operation timed out
            "rc=7",   # curl: (7) Failed to connect
            "rc=52",  # curl: (52) Empty reply from server
            "rc=56",  # curl: (56) Recv failure
            "503",
            "502",
            "504",
            "overloaded",
            "service unavailable",
            "connection",
            "temporarily",
        )
    )


def _retry_with_backoff(
    fn: Callable[[], T], *, max_retries: int = 8, base: float = 2.0, cap: float = 60.0
) -> T:
    """Retry ``fn`` on transient API errors (429 / 5xx / timeout) with exponential backoff."""
    last_err: BaseException | None = None
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as e:
            last_err = e
            if attempt >= max_retries or not _is_transient_error(e):
                raise
            delay = min(base ** attempt, cap) + random.uniform(0, 1.5)
            time.sleep(delay)
    assert last_err is not None
    raise last_err


@dataclass
class ChatMessage:
    role: Literal["system", "user", "assistant"]
    content: str


class LLMClient:
    def __init__(
        self,
        backend: Backend,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> None:
        self.backend = backend
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

        # SDK-level max_retries=0: we layer our own _retry_with_backoff so we can
        # log retry timing for the paper; SDK silent retries would mask it.
        if backend == "openai":
            from openai import OpenAI

            self._client = OpenAI(
                api_key=api_key or os.getenv("OPENAI_API_KEY"),
                max_retries=0,
                timeout=120.0,
            )
        elif backend == "anthropic":
            import anthropic

            self._client = anthropic.Anthropic(
                api_key=api_key or os.getenv("ANTHROPIC_API_KEY"),
                max_retries=0,
                timeout=120.0,
            )
        elif backend == "minimax":
            import anthropic

            key = api_key or os.getenv("MINIMAX_API_KEY")
            url = base_url or os.getenv("MINIMAX_BASE_URL", "https://api.minimax.io/anthropic")
            if not key:
                raise ValueError("MINIMAX_API_KEY is required for the minimax backend")
            # MiniMax gateway: Bearer in Authorization (use auth_token). api_key -> X-Api-Key (Anthropic native only).
            self._client = anthropic.Anthropic(
                auth_token=key, base_url=url, max_retries=0, timeout=120.0
            )
        elif backend == "gemini":
            from google import genai

            key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            if not key:
                raise ValueError("GEMINI_API_KEY is required for the gemini backend")
            self._client = genai.Client(api_key=key)
        elif backend == "ollama":
            # Ollama native /api/chat. We don't use httpx here: on macOS the
            # Local Network Privacy gate denies Python LAN access (returns
            # EHOSTUNREACH masquerading as a route failure) while the system
            # /usr/bin/curl is implicitly granted. So we shell out via curl.
            url = base_url or os.getenv("OLLAMA_BASE_URL") or os.getenv(
                "OLLAMA_HOST", "http://localhost:11434"
            )
            self._ollama_url = url.rstrip("/")
            self._client = None  # not used; kept for API symmetry
        else:
            raise ValueError(f"unknown backend: {backend}")

    def chat(self, messages: list[ChatMessage], *, system: str | None = None) -> str:
        if self.backend == "openai":
            return self._chat_openai(messages, system)
        if self.backend == "ollama":
            return self._chat_ollama(messages, system)
        if self.backend == "gemini":
            return self._chat_gemini(messages, system)
        return self._chat_anthropic(messages, system)

    def _chat_ollama(self, messages: list[ChatMessage], system: str | None) -> str:
        import json as _json
        import subprocess

        payload_msgs = []
        if system:
            payload_msgs.append({"role": "system", "content": system})
        payload_msgs.extend({"role": m.role, "content": m.content} for m in messages)
        body = {
            "model": self.model,
            "messages": payload_msgs,
            "stream": False,
            "think": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }

        def _call() -> str:
            proc = subprocess.run(
                [
                    "curl", "-sS", "--fail-with-body",
                    "--max-time", "120",
                    "-H", "Content-Type: application/json",
                    "-d", _json.dumps(body),
                    f"{self._ollama_url}/api/chat",
                ],
                capture_output=True,
                text=True,
                timeout=130,
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    f"curl/ollama failed (rc={proc.returncode}): {proc.stderr.strip()[:300]} :: {proc.stdout.strip()[:300]}"
                )
            data = _json.loads(proc.stdout)
            return (data.get("message") or {}).get("content", "") or ""

        return _retry_with_backoff(_call)

    def _chat_openai(self, messages: list[ChatMessage], system: str | None) -> str:
        payload = []
        if system:
            payload.append({"role": "system", "content": system})
        payload.extend({"role": m.role, "content": m.content} for m in messages)

        def _call():
            return self._client.chat.completions.create(
                model=self.model,
                messages=payload,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )

        resp = _retry_with_backoff(_call)
        return resp.choices[0].message.content or ""

    def _chat_gemini(self, messages: list[ChatMessage], system: str | None) -> str:
        # Gemini's google-genai SDK takes a single 'contents' string or a list of parts.
        # We collapse a multi-turn user/assistant exchange into a single conversational
        # blob — the persona-caps chat() flow only ever sends one user turn anyway.
        from google.genai import types

        parts = []
        if system:
            parts.append(f"[System]\n{system}\n")
        for m in messages:
            if m.role == "system":
                parts.append(f"[System]\n{m.content}\n")
            elif m.role == "user":
                parts.append(f"[User]\n{m.content}")
            else:
                parts.append(f"[Assistant]\n{m.content}")
        prompt = "\n\n".join(parts)

        cfg = types.GenerateContentConfig(
            temperature=self.temperature,
            max_output_tokens=self.max_tokens,
        )

        def _call():
            return self._client.models.generate_content(
                model=self.model, contents=prompt, config=cfg,
            )

        resp = _retry_with_backoff(_call)
        return resp.text or ""

    def _chat_anthropic(self, messages: list[ChatMessage], system: str | None) -> str:
        payload = [{"role": m.role, "content": m.content} for m in messages if m.role != "system"]
        kwargs: dict = {
            "model": self.model,
            "messages": payload,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        if system:
            kwargs["system"] = system

        def _call():
            return self._client.messages.create(**kwargs)

        resp = _retry_with_backoff(_call)
        out: list[str] = []
        for block in resp.content:
            btype = getattr(block, "type", None)
            if btype == "text":
                out.append(block.text)
        return "".join(out)


# Convenience factory for the three backends we benchmark in the paper.
_PRESETS: dict[str, tuple[Backend, str]] = {
    "gpt-4o": ("openai", "gpt-4o"),
    "gpt-4o-mini": ("openai", "gpt-4o-mini"),
    "claude": ("anthropic", "claude-sonnet-4-5"),
    "minimax": ("minimax", "MiniMax-M2.7"),
    "minimax-fast": ("minimax", "MiniMax-M2.7-highspeed"),
    "gemini-flash": ("gemini", "gemini-2.5-flash"),
    "gemini-pro": ("gemini", "gemini-2.5-pro"),
    # Local Ollama (3090 box) — set OLLAMA_HOST or OLLAMA_BASE_URL to point at it.
    "qwen35-9b": ("ollama", "qwen3.5:latest"),
    "qwen35-9b-abl": ("ollama", "lukey03/qwen3.5-9b-abliterated:latest"),
    "qwen35-27b": ("ollama", "qwen3.5:27b"),
    "qwen35-27b-abl": ("ollama", "huihui_ai/qwen3.5-abliterated:latest"),
    "llama3-8b": ("ollama", "llama3:8b"),
}


def get_client(name: str, **kwargs) -> LLMClient:
    """Resolve a friendly name (e.g. 'minimax') to an LLMClient.

    Falls back to treating ``name`` as ``backend:model`` (e.g. ``openai:gpt-4o``)
    if it is not a registered preset.

    For the ``minimax`` preset specifically, if the env var ``MINIMAX_MODEL``
    is set we respect it (so users with access to e.g. ``-highspeed`` variants
    don't need a config change).
    """
    if name in _PRESETS:
        backend, model = _PRESETS[name]
        if name == "minimax":
            model = os.getenv("MINIMAX_MODEL", model)
        return LLMClient(backend=backend, model=model, **kwargs)
    if ":" in name:
        backend, model = name.split(":", 1)
        return LLMClient(backend=backend, model=model, **kwargs)  # type: ignore[arg-type]
    raise ValueError(f"unknown client name: {name}; presets: {list(_PRESETS)}")
