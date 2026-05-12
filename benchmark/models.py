"""Model registry. Tiers are named (small/medium/specialized/leading); each
tier maps to a model name resolvable by langchain (gpt*, claude*, gemini*, or
anything else → local Ollama at http://localhost:11434)."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class ModelSpec:
    tier: str
    name: str
    provider: str  # "openai" | "anthropic" | "gemini" | "ollama"


def _provider_for(name: str) -> str:
    n = (name or "").lower()
    if "gpt" in n:
        return "openai"
    if "claude" in n:
        return "anthropic"
    if "gemini" in n:
        return "gemini"
    return "ollama"


def tier(name: str, default: str) -> ModelSpec:
    actual = os.getenv(f"BENCH_MODEL_{name.upper()}", default)
    return ModelSpec(tier=name, name=actual, provider=_provider_for(actual))


DEFAULT_TIERS = {
    "small": tier("small", "llama3.2:3b"),
    "medium": tier("medium", "qwen2.5:7b"),
    "specialized": tier("specialized", "medgemma:27b"),
    "large": tier("large", "qwen3.5:27b"),
    "leads": tier("leads", "hf.co/mradermacher/leads-mistral-7b-v1-GGUF:latest"),
    "leading": tier("leading", "claude-sonnet-4-6"),
}


def is_available(spec: ModelSpec) -> tuple[bool, str]:
    """Return (available, reason_if_not)."""
    if spec.provider == "openai" and not os.getenv("OPENAI_API_KEY"):
        return False, "OPENAI_API_KEY not set"
    if spec.provider == "anthropic" and not os.getenv("ANTHROPIC_API_KEY"):
        return False, "ANTHROPIC_API_KEY not set"
    if spec.provider == "gemini" and not os.getenv("GEMINI_API_KEY"):
        return False, "GEMINI_API_KEY not set"
    if spec.provider == "ollama":
        # Best-effort check via /api/tags
        import requests
        try:
            r = requests.get("http://localhost:11434/api/tags", timeout=2)
            if r.status_code != 200:
                return False, f"Ollama unreachable (status {r.status_code})"
            models = [m.get("name", "") for m in r.json().get("models", [])]
            if spec.name not in models:
                return False, f"Ollama model '{spec.name}' not pulled (have: {', '.join(models[:5])}…)"
        except Exception as e:
            return False, f"Ollama probe failed: {e}"
    return True, ""


def build(spec: ModelSpec):
    """Return a langchain-compatible chat model for this spec."""
    if spec.provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=spec.name, api_key=os.getenv("OPENAI_API_KEY"), temperature=0)
    if spec.provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=spec.name, api_key=os.getenv("ANTHROPIC_API_KEY"), temperature=0)
    if spec.provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=spec.name, api_key=os.getenv("GEMINI_API_KEY"), temperature=0)
    from langchain_ollama import ChatOllama
    return ChatOllama(model=spec.name, temperature=0, base_url="http://localhost:11434")
