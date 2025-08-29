"""Utilities for LLM assisted feature engineering."""
from typing import List

try:
    import openai
except Exception:  # pragma: no cover - optional dependency
    openai = None


def synthesize_features(texts: List[str], model: str = "gpt-4o-mini") -> List[str]:
    """Return list of synthetic features from input texts using an LLM.

    If OpenAI API is unavailable, returns empty list.
    """
    if openai is None:
        return []
    resp = openai.ChatCompletion.create(model=model, messages=[{"role": "user", "content": "\n".join(texts)}])
    return [c['message']['content'] for c in resp['choices']]
