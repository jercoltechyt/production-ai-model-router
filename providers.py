# -*- coding: utf-8 -*-
"""
Provider adapter helpers and zero-dependency HTTP invokers for OpenAI, Anthropic, and Google Gemini (Nov 2026 APIs).

Maps unified router `reasoning_effort` ("low" | "medium" | "high") to each vendor's native parameter
without triggering deprecated `max_tokens` or `temperature=0` HTTP 400 errors on reasoning models.
"""

from __future__ import annotations

import json
import os
import urllib.request
from typing import Any


def build_openai_payload(
    model: str,
    prompt: str,
    reasoning_effort: str = "medium",
    max_completion_tokens: int = 2048,
    prompt_cache_key: str | None = None,
) -> dict[str, Any]:
    """
    Builds a modern OpenAI Chat Completions / Responses payload (GPT-6 Astra, GPT-6.1 Sol, GPT-5.6 Luna).
    Uses `max_completion_tokens` and `reasoning.effort` instead of deprecated `max_tokens` or `temperature=0`.
    """
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "reasoning": {"effort": reasoning_effort},
        "max_completion_tokens": max_completion_tokens,
    }
    if prompt_cache_key:
        payload["prompt_cache_key"] = prompt_cache_key
    return payload


def build_anthropic_payload(
    model: str,
    prompt: str,
    reasoning_effort: str = "medium",
    static_system_prefix: str | None = None,
    max_tokens: int = 2048,
) -> dict[str, Any]:
    """
    Builds an Anthropic Messages payload (Claude Haiku 5.5, Sonnet 5.5, Opus 5.5)
    with native `effort` control and optional ephemeral Prompt Cache breakpoint.
    """
    payload: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "effort": reasoning_effort,
        "messages": [{"role": "user", "content": prompt}],
    }
    if static_system_prefix:
        payload["system"] = [
            {
                "type": "text",
                "text": static_system_prefix,
                "cache_control": {"type": "ephemeral"},
            }
        ]
    return payload


def build_gemini_payload(
    model: str,
    prompt: str,
    reasoning_effort: str = "medium",
    cached_content_name: str | None = None,
) -> dict[str, Any]:
    """
    Builds a Google Gemini `generateContent` configuration (Gemini 3.8 Flash-Lite, 4 Argon, 4 Carbon)
    using `thinkingLevel` ("LOW" | "MEDIUM" | "HIGH").
    """
    config: dict[str, Any] = {
        "model": model,
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "thinkingConfig": {"thinkingLevel": reasoning_effort.upper()},
        },
    }
    if cached_content_name:
        config["cachedContent"] = cached_content_name
    return config


def create_live_or_mock_invoker(
    prompt: str,
    static_system_prefix: str | None = None,
    dry_run: bool = True,
):
    """
    Returns an `(model, effort) -> str` invoker compatible with `route_with_cascade`.
    When `dry_run=False` and vendor API keys (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`)
    are set in the environment, executes zero-dependency HTTPS calls via `urllib.request`.
    """

    def _invoker(model: str, effort: str) -> str:
        if dry_run:
            return json.dumps(
                {
                    "provider_model": model,
                    "reasoning_effort": effort,
                    "status": "verified",
                    "cache_prefix_attached": bool(static_system_prefix),
                }
            )

        if model.startswith("gpt-"):
            api_key = os.environ["OPENAI_API_KEY"]
            body = build_openai_payload(model, prompt, reasoning_effort=effort)
            req = urllib.request.Request(
                "https://api.openai.com/v1/chat/completions",
                data=json.dumps(body).encode("utf-8"),
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data["choices"][0]["message"]["content"]

        if model.startswith("claude-"):
            api_key = os.environ["ANTHROPIC_API_KEY"]
            body = build_anthropic_payload(model, prompt, reasoning_effort=effort, static_system_prefix=static_system_prefix)
            req = urllib.request.Request(
                "https://api.anthropic.com/v1/messages",
                data=json.dumps(body).encode("utf-8"),
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data["content"][0]["text"]

        api_key = os.environ["GEMINI_API_KEY"]
        body = build_gemini_payload(model, prompt, reasoning_effort=effort)
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        req = urllib.request.Request(
            url,
            data=json.dumps({k: v for k, v in body.items() if k != "model"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["candidates"][0]["content"]["parts"][0]["text"]

    return _invoker
