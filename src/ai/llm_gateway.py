"""Lightweight wrappers for optional LLM calls used by the trading engine.

Provides:
- noise filtering on AlertEvents via embeddings/keywords
- signal validation returning confidence scores
- contextual rationale generation
- feedback logging for future model tuning
"""
from __future__ import annotations

import json
import math
import pathlib
import datetime as dt
from dataclasses import asdict
from typing import Dict, List, Tuple

try:  # pragma: no cover - optional dependency
    import openai  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    openai = None  # type: ignore

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "out"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FEEDBACK_PATH = OUT_DIR / "llm_feedback.jsonl"


class LLMGateway:
    """Facade for LLM helpers with graceful fallbacks when API is absent."""

    def __init__(self) -> None:
        self.client = openai

    # ------------------ noise filtering ------------------
    def _is_structural(self, message: str) -> bool:
        msg = message or ""
        if self.client:
            try:  # pragma: no cover - API path
                emb = self.client.Embedding.create(model="text-embedding-3-small", input=msg)
                noise = self.client.Embedding.create(
                    model="text-embedding-3-small", input="market rumour transient noise"
                )
                vec = emb["data"][0]["embedding"]
                noise_vec = noise["data"][0]["embedding"]
                dot = sum(a * b for a, b in zip(vec, noise_vec))
                norm_a = math.sqrt(sum(a * a for a in vec))
                norm_b = math.sqrt(sum(b * b for b in noise_vec))
                sim = dot / (norm_a * norm_b + 1e-9)
                return sim < 0.8
            except Exception:
                pass
        keywords = ["rumour", "rumor", "noise", "tweet", "whisper"]
        return not any(k in msg.lower() for k in keywords)

    def filter_alerts(self, alerts: List) -> List:
        """Return only alerts deemed structural."""
        out = []
        for al in alerts:
            msg = getattr(al, "message", "")
            if self._is_structural(msg):
                out.append(al)
        return out

    # ------------------ signal validation ------------------
    def validate_signal(self, features: Dict[str, float], headlines: List[str]) -> Tuple[float, str]:
        """Return (confidence, rationale) for given features using an LLM."""
        if self.client:
            try:  # pragma: no cover - API path
                prompt = (
                    "You are a quantitative trading assistant. Given numeric features and macro headlines, "
                    "respond with a single confidence value between 0 and 1 followed by a short reason."
                )
                content = json.dumps({"features": features, "headlines": headlines})
                resp = self.client.ChatCompletion.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "system", "content": prompt}, {"role": "user", "content": content}],
                )
                txt = resp["choices"][0]["message"]["content"].strip()
                conf_str = txt.split()[0]
                conf = float(conf_str)
                return max(0.0, min(1.0, conf)), txt
            except Exception:
                pass
        conf = float(abs(features.get("trend_score", 0.0)))
        return max(0.0, min(1.0, conf)), ""

    # ------------------ contextual decisioning ------------------
    def contextualize(self, inst_info: Dict[str, float], scenario: str, techs: Dict[str, float]) -> str:
        """Summarise flows and breakout context for alerts."""
        if self.client:
            try:  # pragma: no cover - API path
                prompt = (
                    "Summarise institutional flow and key technical context for a trade alert in one sentence."
                )
                content = json.dumps({"flows": inst_info, "scenario": scenario, "techs": techs})
                resp = self.client.ChatCompletion.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "system", "content": prompt}, {"role": "user", "content": content}],
                )
                return resp["choices"][0]["message"]["content"].strip()
            except Exception:
                pass
        bias = inst_info.get("fii", 0.0) - inst_info.get("dii", 0.0)
        if bias > 0:
            bias_str = f"FII buying {inst_info['fii']:.0f} vs DII {inst_info['dii']:.0f}"
        elif bias < 0:
            bias_str = f"DII buying {inst_info['dii']:.0f} vs FII {inst_info['fii']:.0f}"
        else:
            bias_str = "Flows balanced"
        breakout_flags = [
            name for name in ("orb_up", "orb_down", "micro_bull", "micro_bear") if techs.get(name)
        ]
        breakout_str = ", ".join(breakout_flags) if breakout_flags else "no breakouts"
        return f"{bias_str}; scenario {scenario}; {breakout_str}."

    # ------------------ feedback logging ------------------
    def log_feedback(self, symbol: str, decision) -> None:
        """Persist decision and rationale for later review."""
        try:
            rec = {
                "ts": dt.datetime.utcnow().isoformat(),
                "symbol": symbol,
                "decision": asdict(decision),
            }
            with FEEDBACK_PATH.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
        except Exception:
            pass
