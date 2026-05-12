"""Async LLM API client compatible with vLLM / OpenAI chat completions."""
from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import aiohttp


@dataclass
class LLMResponse:
    status: str  # "success" or "error"
    text: str = ""
    reasoning: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    error: str = ""
    raw: dict = field(default_factory=dict)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4) if text else 0


def _normalize_text_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                t = item.get("text") or item.get("content") or ""
                if t:
                    parts.append(str(t))
        return "\n".join(parts).strip()
    return str(content)


def _parse_response(payload: dict[str, Any], prompt_text: str = "") -> LLMResponse:
    choices = payload.get("choices") or []
    if not choices:
        return LLMResponse(status="error", error="no choices in response")
    msg = choices[0].get("message") or {}

    visible = _normalize_text_content(msg.get("content", ""))
    reasoning = _normalize_text_content(
        msg.get("reasoning_content") or msg.get("reasoning") or msg.get("thinking") or ""
    )
    if not reasoning and visible.startswith("<think>"):
        m = re.search(r"<think>(.*?)</think>(.*)", visible, re.DOTALL)
        if m:
            reasoning = m.group(1).strip()
            visible = m.group(2).strip()
        else:
            m = re.match(r"<think>(.*)", visible, re.DOTALL)
            if m:
                reasoning = m.group(1).strip()
                visible = ""

    usage = payload.get("usage") or {}
    prompt_tokens = int(usage.get("prompt_tokens") or _estimate_tokens(prompt_text))
    completion_tokens = int(usage.get("completion_tokens") or _estimate_tokens(reasoning + visible))
    total_tokens = int(usage.get("total_tokens") or (prompt_tokens + completion_tokens))

    return LLMResponse(
        status="success",
        text=visible,
        reasoning=reasoning,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        raw=payload,
    )


async def call_llm(
    session: aiohttp.ClientSession,
    prompt: str,
    *,
    model: str,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    enable_thinking: bool | None = None,
    thinking_budget: int | None = None,
    timeout: int = 600,
    api_base_url: str = "",
    api_key: str = "",
    max_retries: int = 3,
) -> LLMResponse:
    base_url = api_base_url or os.environ.get("API_BASE_URL", "http://localhost:8001/v1/chat/completions")
    key = api_key or os.environ.get("API_KEY", "dummy")

    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "stream": False,
    }
    if max_tokens > 0:
        payload["max_tokens"] = max_tokens
    if enable_thinking is not None:
        payload["chat_template_kwargs"] = {"enable_thinking": enable_thinking}
    if thinking_budget is not None:
        payload["thinking"] = {"budget_tokens": thinking_budget}

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json; charset=utf-8",
    }
    req_timeout = aiohttp.ClientTimeout(total=timeout)

    for attempt in range(max_retries + 1):
        try:
            async with session.post(base_url, headers=headers, json=payload, timeout=req_timeout) as resp:
                body = await resp.text()
                if resp.status >= 400:
                    err = f"HTTP {resp.status}: {body[:300]}"
                    if attempt < max_retries:
                        await asyncio.sleep(min(20.0, 2.0 ** attempt))
                        continue
                    return LLMResponse(status="error", error=err)
                return _parse_response(json.loads(body), prompt)
        except (aiohttp.ClientError, asyncio.TimeoutError, json.JSONDecodeError) as exc:
            if attempt < max_retries:
                await asyncio.sleep(min(20.0, 2.0 ** attempt))
                continue
            return LLMResponse(status="error", error=str(exc))

    return LLMResponse(status="error", error="max retries exceeded")
