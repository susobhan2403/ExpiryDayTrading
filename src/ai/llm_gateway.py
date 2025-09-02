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
import statistics
import asyncio

from src.config import load_settings

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

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        self.client = openai
        self.model = model
        if self.client:  # load API key from settings.json if available
            try:
                cfg = load_settings()
                key = cfg.get("OPEN_API_KEY", "")
                if key:
                    self.client.api_key = key
                else:
                    self.client = None
            except Exception:
                self.client = None

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
                    model=self.model,
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
                    model=self.model,
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

    # ------------------ regime & thresholds ------------------
    def classify_regime(self, metrics: Dict[str, float]) -> str:
        """Classify market regime using simple prompt schema.

        Parameters
        ----------
        metrics: dict
            Dictionary of recent numeric metrics such as volatility and
            trend indicators.

        Returns
        -------
        str
            Regime label like ``TRENDING`` or ``RANGING``.
        """
        if self.client:
            try:  # pragma: no cover - API path
                prompt = (
                    "Classify the market regime as TRENDING, RANGING or VOLATILE "
                    'given the following metrics. Respond with JSON `{"regime": "<label>"}`.'
                )
                content = json.dumps({"metrics": metrics})
                resp = self.client.ChatCompletion.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": content},
                    ],
                )
                data = json.loads(resp["choices"][0]["message"]["content"].strip())
                return str(data.get("regime", "UNKNOWN")).upper()
            except Exception:
                pass
        # Fallback heuristic: ADX for trendiness and VND for volatility
        adx = float(metrics.get("adx", 0.0))
        vnd = float(metrics.get("vnd", 0.0))
        if adx >= 25 and vnd >= 0.5:
            return "TRENDING"
        if vnd > 1.5:
            return "VOLATILE"
        return "RANGING"

    def suggest_thresholds(self, history: Dict[str, List[float]]) -> Dict[str, float]:
        """Suggest multipliers for spike/volume gates based on recent history.

        ``history`` should contain lists keyed by ``"pcr"`` and ``"iv"``.
        Returns a mapping ``{"pcr": mult, "iv": mult}``.
        """
        if self.client:
            try:  # pragma: no cover - API path
                prompt = (
                    "Given recent z-score histories for put-call ratio deltas and IV changes, "
                    'suggest multipliers for spike detection thresholds as JSON `{"pcr": x, "iv": y}`'
                )
                content = json.dumps(history)
                resp = self.client.ChatCompletion.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": content},
                    ],
                )
                data = json.loads(resp["choices"][0]["message"]["content"].strip())
                p_mult = float(data.get("pcr", 1.0))
                i_mult = float(data.get("iv", 1.0))
                return {"pcr": p_mult, "iv": i_mult}
            except Exception:
                pass
        # Fallback heuristic: widen thresholds when history volatile
        def _mult(vals: List[float]) -> float:
            if len(vals) < 5:
                return 1.0
            s = statistics.pstdev(vals)
            if s > 2.5:
                return 1.3
            if s < 0.8:
                return 0.8
            return 1.0

        return {
            "pcr": _mult(history.get("pcr", [])),
            "iv": _mult(history.get("iv", [])),
        }

    async def aclassify_regime(self, metrics: Dict[str, float]) -> str:
        """Async wrapper for :func:`classify_regime`."""
        return await asyncio.to_thread(self.classify_regime, metrics)

    async def asuggest_thresholds(self, history: Dict[str, List[float]]) -> Dict[str, float]:
        """Async wrapper for :func:`suggest_thresholds`."""
        return await asyncio.to_thread(self.suggest_thresholds, history)

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
