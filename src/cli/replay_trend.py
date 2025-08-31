from __future__ import annotations
import argparse
import logging
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
from prometheus_client import CollectorRegistry, Counter, Gauge, start_http_server, write_to_textfile

from src.strategy.trend_consensus import TrendConsensus

logger = logging.getLogger(__name__)


def replay(
    csv_path: str,
    metrics_out: str,
    prom_port: Optional[int] = None,
    weights: Optional[Dict[int, float]] = None,
    threshold: float = 0.6,
    confirm: int = 3,
    alpha: float = 0.3,
) -> Dict[str, float]:
    """Replay historical 1m bars through TrendConsensus.

    Parameters
    ----------
    csv_path: path to CSV file with columns ts, open, high, low, close, volume
    metrics_out: path to CSV file where metrics will be written
    prom_port: if provided, start an HTTP server to expose Prometheus metrics
    weights, threshold, confirm, alpha: parameters for TrendConsensus

    Returns
    -------
    dict with flip_count and avg_holding_seconds
    """
    df = pd.read_csv(csv_path, parse_dates=["ts"])
    if "volume" not in df.columns:
        df["volume"] = 0.0
    df = df.set_index("ts").sort_index()

    tc = TrendConsensus(weights=weights, threshold=threshold, confirm=confirm, alpha=alpha)

    registry = CollectorRegistry()
    flip_counter = Counter("trend_flip_total", "Number of trend direction flips", registry=registry)
    hold_gauge = Gauge("trend_last_holding_seconds", "Duration of last completed trend", registry=registry)
    avg_hold_gauge = Gauge("trend_avg_holding_seconds", "Average holding time", registry=registry)

    if prom_port:
        start_http_server(prom_port, registry=registry)

    last_dir: Optional[str] = None
    last_flip_ts: Optional[pd.Timestamp] = None
    flip_count = 0
    holding_times: list[float] = []

    for ts, row in df.iterrows():
        frame = pd.DataFrame([row], index=[ts])
        res = tc.evaluate(frame)
        if last_dir is None:
            last_dir = res.direction
            last_flip_ts = ts
            continue
        if res.direction != last_dir:
            flip_count += 1
            flip_counter.inc()
            if last_flip_ts is not None:
                hold_sec = (ts - last_flip_ts).total_seconds()
                holding_times.append(hold_sec)
                hold_gauge.set(hold_sec)
                logger.info("Scenario flip: %s -> %s @ %s", last_dir, res.direction, ts)
            last_dir = res.direction
            last_flip_ts = ts

    avg_hold = sum(holding_times) / len(holding_times) if holding_times else 0.0
    avg_hold_gauge.set(avg_hold)

    metrics_path = Path(metrics_out)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"flip_count": [flip_count], "avg_holding_seconds": [avg_hold]}).to_csv(
        metrics_path, index=False
    )
    write_to_textfile(metrics_path.with_suffix(".prom"), registry)
    return {"flip_count": flip_count, "avg_holding_seconds": avg_hold}



def main() -> None:
    ap = argparse.ArgumentParser(description="Replay historical data through TrendConsensus")
    ap.add_argument("--csv", required=True, help="CSV file with 1m OHLCV data")
    ap.add_argument(
        "--metrics-out",
        default=str(Path("out") / "trend_metrics.csv"),
        help="Path to write metrics CSV",
    )
    ap.add_argument("--prom-port", type=int, default=0, help="Expose Prometheus metrics on port")
    ap.add_argument("--confirm", type=int, default=3, help="TrendConsensus confirm parameter")
    ap.add_argument("--threshold", type=float, default=0.6, help="TrendConsensus threshold")
    ap.add_argument("--alpha", type=float, default=0.3, help="TrendConsensus smoothing factor")
    ap.add_argument("--weights", help="Comma separated timeframe:weight pairs, e.g. '1:1,5:0.5'")
    args = ap.parse_args()

    weights: Optional[Dict[int, float]] = None
    if args.weights:
        weights = {}
        for part in args.weights.split(","):
            tf, w = part.split(":")
            weights[int(tf)] = float(w)

    port = args.prom_port if args.prom_port > 0 else None
    logging.basicConfig(level=logging.INFO, format="[replay] %(message)s")
    replay(args.csv, args.metrics_out, prom_port=port, weights=weights,
           threshold=args.threshold, confirm=args.confirm, alpha=args.alpha)


if __name__ == "__main__":
    main()
