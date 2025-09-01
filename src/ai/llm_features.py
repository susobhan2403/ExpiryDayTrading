"""Utilities for LLM assisted feature engineering."""
from typing import List, Optional
import os

from src.config import load_settings

try:
    import openai
    key = load_settings().get("OPEN_API_KEY", "")
    if key:
        openai.api_key = key
except Exception:  # pragma: no cover - optional dependency
    openai = None


def synthesize_features(texts: List[str], model: Optional[str] = None) -> List[str]:
    """Return list of synthetic features from input texts using an LLM.

    If ``model`` is ``None`` the value is read from ``settings.json`` or the
    ``OPENAI_MODEL`` environment variable.  If the OpenAI API is unavailable,
    returns an empty list.
    """
    if openai is None:
        return []
    if model is None:
        cfg_model = ""
        try:
            cfg_model = load_settings().get("OPENAI_MODEL", "")
        except Exception:
            pass
        model = os.environ.get("OPENAI_MODEL", cfg_model or "gpt-4o-mini")
    resp = openai.ChatCompletion.create(
        model=model, messages=[{"role": "user", "content": "\n".join(texts)}]
    )
    return [c["message"]["content"] for c in resp["choices"]]
