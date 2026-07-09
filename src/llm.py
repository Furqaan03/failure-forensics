"""Thin OpenAI wrapper that returns text + the raw prompt/response for tracing."""
from __future__ import annotations

from openai import OpenAI
from pydantic import BaseModel

_client: OpenAI | None = None


def _openai() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


class LLMCall(BaseModel):
    prompt: str
    response: str
    input_tokens: int
    output_tokens: int


def call_llm(prompt: str, model: str = "gpt-4o-mini", temperature: float = 0.0) -> LLMCall:
    resp = _openai().chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    usage = resp.usage
    return LLMCall(
        prompt=prompt,
        response=resp.choices[0].message.content or "",
        input_tokens=usage.prompt_tokens if usage else 0,
        output_tokens=usage.completion_tokens if usage else 0,
    )
