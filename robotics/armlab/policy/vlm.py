"""Vision-language model clients behind one small interface.

- AnthropicVLM: Claude via the official `anthropic` SDK (structured JSON output).
- OpenAICompatVLM: any OpenAI-compatible chat endpoint (OpenAI models such as GPT-6 Astra,
  NVIDIA NIM / build.nvidia.com hosted Cosmos Reason, vLLM servers on Modal, ...).
- ScriptedVLM: deterministic stand-in for tests and dry runs.
"""
from __future__ import annotations

import base64
import io
import json
import os
import time
from dataclasses import dataclass
from typing import Callable, Union

import numpy as np
from PIL import Image

from .types import Usage

Part = Union[str, np.ndarray, "VideoPart"]


@dataclass
class VideoPart:
    """An mp4 clip, for models that accept video input (e.g. Cosmos Reason via vLLM/NIM)."""
    data: bytes
    mime: str = "video/mp4"


def png_b64(img: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    return base64.standard_b64encode(buf.getvalue()).decode()


def jpeg_b64(img: np.ndarray, quality: int = 85) -> str:
    buf = io.BytesIO()
    Image.fromarray(img).convert("RGB").save(buf, format="JPEG", quality=quality)
    return base64.standard_b64encode(buf.getvalue()).decode()


def extract_json(text: str) -> dict:
    """Parse a JSON object from model text, tolerating code fences or prose around it."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        text = text.rsplit("```", 1)[0]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start:end + 1])
        raise


class VLM:
    name = "vlm"

    def complete(self, system: str, parts: list[Part], schema: dict | None = None,
                 max_tokens: int = 16000) -> tuple[str, Usage]:  # pragma: no cover - interface
        raise NotImplementedError


class MissingCredentials(RuntimeError):
    pass


class AnthropicVLM(VLM):
    """Claude via the Anthropic SDK. Credentials come from ANTHROPIC_API_KEY (or an `ant auth login` profile)."""

    def __init__(self, model: str = "claude-opus-5-5", effort: str = "medium", fallbacks: bool = True,
                 timeout_s: float = 600.0):
        import anthropic

        self.anthropic = anthropic
        self.model = model
        self.effort = effort
        self.fallbacks = fallbacks
        self.name = f"anthropic:{model}:{effort}"
        self.client = anthropic.Anthropic(timeout=timeout_s, max_retries=3)

    def _content(self, parts: list[Part]) -> list[dict]:
        out = []
        for p in parts:
            if isinstance(p, str):
                out.append({"type": "text", "text": p})
            elif isinstance(p, np.ndarray):
                out.append({"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": png_b64(p)}})
            else:
                raise ValueError("AnthropicVLM does not accept video; pass sampled frames instead")
        return out

    def complete(self, system, parts, schema=None, max_tokens=16000):
        output_config: dict = {"effort": self.effort}
        if schema is not None:
            output_config["format"] = {"type": "json_schema", "schema": schema}
        kwargs = dict(
            model=self.model,
            max_tokens=max_tokens,
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": self._content(parts)}],
            output_config=output_config,
        )
        t0 = time.time()
        if self.fallbacks:
            # Server-side refusal fallback: a declined request is re-run on Anthropic's recommended model.
            try:
                resp = self.client.beta.messages.create(betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kwargs)
            except self.anthropic.BadRequestError as e:
                if "fallback" not in str(e).lower():
                    raise
                self.fallbacks = False
                resp = self.client.messages.create(**kwargs)
        else:
            resp = self.client.messages.create(**kwargs)
        latency = time.time() - t0
        if resp.stop_reason == "refusal":
            raise RuntimeError(f"model refused: {getattr(resp, 'stop_details', None)}")
        text = "".join(b.text for b in resp.content if b.type == "text")
        u = resp.usage
        usage = Usage(
            input_tokens=(u.input_tokens or 0) + (u.cache_read_input_tokens or 0) + (u.cache_creation_input_tokens or 0),
            output_tokens=u.output_tokens or 0,
            cached_input_tokens=u.cache_read_input_tokens or 0,
            latency_s=latency,
            calls=1,
        )
        return text, usage


class OpenAICompatVLM(VLM):
    """OpenAI-compatible Chat Completions endpoint.

    Examples:
      OpenAI:            OpenAICompatVLM(model="<gpt-6-astra model id>")             # OPENAI_API_KEY
      NVIDIA hosted NIM: OpenAICompatVLM(model="nvidia/cosmos-reason2-8b",
                                         base_url="https://integrate.api.nvidia.com/v1",
                                         api_key_env="NVIDIA_API_KEY")
      Self-hosted vLLM:  OpenAICompatVLM(model="nvidia/Cosmos-Reason2-8B", base_url="https://<modal-app>/v1",
                                         api_key_env="ARMLAB_VLLM_KEY")
    """

    def __init__(self, model: str, base_url: str | None = None, api_key_env: str = "OPENAI_API_KEY",
                 json_mode: bool = True, reasoning_effort: str | None = None, extra_body: dict | None = None,
                 timeout_s: float = 600.0):
        from openai import OpenAI

        key = os.environ.get(api_key_env) or ("EMPTY" if api_key_env == "ARMLAB_VLLM_KEY" else None)
        if not key:
            raise MissingCredentials(f"Set {api_key_env} to use {model}")
        self.client = OpenAI(api_key=key, base_url=base_url, timeout=timeout_s, max_retries=3)
        # OpenAI's own API uses max_completion_tokens; vLLM / NIM servers accept max_tokens.
        self.token_param = "max_tokens" if base_url else "max_completion_tokens"
        self.model = model
        self.json_mode = json_mode
        self.reasoning_effort = reasoning_effort
        self.extra_body = extra_body or {}
        self.name = f"openai-compat:{model}"

    def _content(self, parts):
        out = []
        for p in parts:
            if isinstance(p, str):
                out.append({"type": "text", "text": p})
            elif isinstance(p, np.ndarray):
                out.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{jpeg_b64(p)}"}})
            elif isinstance(p, VideoPart):
                b64 = base64.standard_b64encode(p.data).decode()
                out.append({"type": "video_url", "video_url": {"url": f"data:{p.mime};base64,{b64}"}})
        return out

    def complete(self, system, parts, schema=None, max_tokens=16000):
        sys_text = system
        if schema is not None:
            sys_text += "\n\nRespond with a single JSON object matching this JSON schema:\n" + json.dumps(schema)
        kwargs = dict(model=self.model, messages=[{"role": "system", "content": sys_text},
                                                  {"role": "user", "content": self._content(parts)}],
                      **{self.token_param: max_tokens})
        if schema is not None and self.json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort
        if self.extra_body:
            kwargs["extra_body"] = self.extra_body
        t0 = time.time()
        resp = self.client.chat.completions.create(**kwargs)
        latency = time.time() - t0
        text = resp.choices[0].message.content or ""
        u = resp.usage
        cached = 0
        if u is not None and getattr(u, "prompt_tokens_details", None) is not None:
            cached = getattr(u.prompt_tokens_details, "cached_tokens", 0) or 0
        usage = Usage(input_tokens=getattr(u, "prompt_tokens", 0) or 0, output_tokens=getattr(u, "completion_tokens", 0) or 0,
                      cached_input_tokens=cached, latency_s=latency, calls=1)
        return text, usage


class ScriptedVLM(VLM):
    """Returns responses from a callable (for tests / offline demos). Optionally sleeps to mimic latency."""

    def __init__(self, respond: Callable[[str, list[Part]], dict | str], latency_s: float = 0.0):
        self.respond = respond
        self.latency_s = latency_s
        self.name = "scripted-vlm"
        self.calls: list[tuple[str, list[Part]]] = []

    def complete(self, system, parts, schema=None, max_tokens=16000):
        self.calls.append((system, parts))
        if self.latency_s:
            time.sleep(self.latency_s)
        out = self.respond(system, parts)
        text = out if isinstance(out, str) else json.dumps(out)
        return text, Usage(input_tokens=1000, output_tokens=len(text) // 4, latency_s=self.latency_s, calls=1)


def make_vlm(spec: str, **kw) -> VLM:
    """Build a client from a short spec string.

    anthropic[:model[:effort]]       e.g. anthropic:claude-opus-5-5:medium
    openai:<model>                   uses OPENAI_API_KEY (and OPENAI_BASE_URL if set)
    nvidia:<model>                   build.nvidia.com hosted endpoint, NVIDIA_API_KEY
    vllm:<model>@<base_url>          self-hosted OpenAI-compatible server, ARMLAB_VLLM_KEY
    """
    kind, _, rest = spec.partition(":")
    if kind == "anthropic":
        model, _, effort = rest.partition(":")
        return AnthropicVLM(model=model or "claude-opus-5-5", effort=effort or "medium", **kw)
    if kind == "openai":
        return OpenAICompatVLM(model=rest, base_url=os.environ.get("OPENAI_BASE_URL"), **kw)
    if kind == "nvidia":
        return OpenAICompatVLM(model=rest, base_url="https://integrate.api.nvidia.com/v1", api_key_env="NVIDIA_API_KEY", **kw)
    if kind == "vllm":
        model, _, base = rest.partition("@")
        return OpenAICompatVLM(model=model, base_url=base, api_key_env="ARMLAB_VLLM_KEY", **kw)
    raise ValueError(f"unknown VLM spec {spec!r}")
