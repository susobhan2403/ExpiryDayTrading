from __future__ import annotations
import os, json, time, threading, pathlib, datetime as dt
from typing import Dict, List, Optional

import pandas as pd

from src.provider.kite import KiteProvider

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "out" / "stream"
OUT_DIR.mkdir(parents=True, exist_ok=True)


class OrderbookCollector:
    """
    Collects top-5 orderbook via KiteTicker (full mode) and aggregates 1s features:
    - spread (L1)
    - bid/ask imbalance = (sum(bid_qty) - sum(ask_qty)) / (sum(bid_qty) + sum(ask_qty))
    - queue imbalance at L1 = (bid_qty1 - ask_qty1) / (bid_qty1 + ask_qty1)
    - quote stability = fraction of last N seconds where best bid/ask unchanged
    - CVD (cumulative volume delta) using tick direction by last traded price change
    Persists CSV per symbol at out/stream/{symbol}_YYYYMMDD.csv
    """

    def __init__(self, symbols: List[str]):
        self.provider = KiteProvider()
        self.symbols = [s.upper() for s in symbols]
        self.tokens = {}
        for s in self.symbols:
            tok = self.provider._resolve_index_token(s)
            if tok: self.tokens[s] = int(tok)
        self._tick_lock = threading.Lock()
        self._buf: Dict[int, List[Dict]] = {}
        self._last_ltp: Dict[int, float] = {}
        self._cvd: Dict[int, float] = {tok: 0.0 for tok in self.tokens.values()}
        self._last_best: Dict[int, tuple] = {}
        self._stability: Dict[int, List[int]] = {}
        self._stop = False
        self._thread: Optional[threading.Thread] = None

    def start(self):
        from kiteconnect import KiteTicker
        sess = json.loads(open(ROOT/".kite_session.json","r").read())
        api_key = os.getenv("KITE_API_KEY") or json.loads(open(ROOT/"settings.json").read()).get("KITE_API_KEY","")
        if not api_key:
            raise SystemExit("KITE_API_KEY missing in env or settings.json")
        kws = KiteTicker(api_key, sess["access_token"])

        def on_ticks(ws, ticks):
            now = dt.datetime.now()
            with self._tick_lock:
                for t in ticks:
                    tok = int(t.get("instrument_token"))
                    ltp = float(t.get("last_price") or 0.0)
                    depth = t.get("depth") or {}
                    buys = depth.get("buy") or []
                    sells = depth.get("sell") or []
                    bid1 = buys[0]["price"] if buys else 0.0
                    ask1 = sells[0]["price"] if sells else 0.0
                    bq1 = buys[0]["quantity"] if buys else 0
                    aq1 = sells[0]["quantity"] if sells else 0
                    spread = (ask1 - bid1) if (ask1 and bid1) else 0.0
                    sbq = sum(x.get("quantity",0) for x in buys)
                    saq = sum(x.get("quantity",0) for x in sells)
                    imb = (sbq - saq)/max(1.0, (sbq + saq))
                    qi = (bq1 - aq1)/max(1.0, (bq1 + aq1))
                    # CVD by tick direction
                    last = self._last_ltp.get(tok, ltp)
                    dirn = 1 if ltp > last else (-1 if ltp < last else 0)
                    self._cvd[tok] = self._cvd.get(tok,0.0) + dirn*max(1.0, (sbq+saq))
                    self._last_ltp[tok] = ltp
                    # best stability
                    best = (bid1, ask1)
                    last_best = self._last_best.get(tok)
                    stable = 1 if (last_best == best) else 0
                    self._last_best[tok] = best
                    self._stability.setdefault(tok, []).append(stable)
                    # buffer
                    self._buf.setdefault(tok, []).append({
                        "ts": now, "ltp": ltp, "spread": spread, "imb": imb, "qi": qi,
                    })

        def on_connect(ws, response):
            tokens = list(self.tokens.values())
            if tokens:
                ws.subscribe(tokens)
                ws.set_mode(ws.MODE_FULL, tokens)

        def on_close(ws, code, reason):
            ws.stop()

        kws.on_ticks = on_ticks
        kws.on_connect = on_connect
        kws.on_close = on_close

        # aggregator thread
        def loop_agg():
            last_flush: Dict[int, dt.datetime] = {}
            while not self._stop:
                time.sleep(1.0)
                out_rows = []
                with self._tick_lock:
                    for sym, tok in self.tokens.items():
                        buf = self._buf.get(tok, [])
                        if not buf:
                            continue
                        # take last second worth (approx all since last flush)
                        rows = buf
                        self._buf[tok] = []
                        # features
                        ts = rows[-1]["ts"]
                        ltp = rows[-1]["ltp"]
                        spread = float(pd.Series([r["spread"] for r in rows]).median())
                        imb = float(pd.Series([r["imb"] for r in rows]).mean())
                        qi = float(pd.Series([r["qi"] for r in rows]).mean())
                        cvd = self._cvd.get(tok, 0.0)
                        stab_arr = self._stability.get(tok, [])
                        stab = float(sum(stab_arr[-10:]) / max(1, len(stab_arr[-10:])))
                        out_rows.append({
                            "ts": ts.isoformat(), "symbol": sym, "ltp": ltp, "spread": round(spread,4),
                            "imb": round(imb,4), "qi": round(qi,4), "cvd": round(cvd,2), "quote_stab": round(stab,3)
                        })
                if out_rows:
                    df = pd.DataFrame(out_rows)
                    date_tag = dt.datetime.now().strftime('%Y%m%d')
                    f = OUT_DIR / f"{date_tag}.csv"
                    header = not f.exists()
                    df.to_csv(f, mode='a', index=False, header=header)

        self._thread = threading.Thread(target=loop_agg, daemon=True)
        self._thread.start()
        try:
            kws.connect(threaded=True, disable_ssl_verification=True)
            while not self._stop:
                time.sleep(0.5)
        finally:
            try: kws.close()
            except Exception: pass

    def stop(self):
        self._stop = True
        if self._thread:
            self._thread.join(timeout=2.0)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="NIFTY,BANKNIFTY")
    args = ap.parse_args()
    symbols = [s.strip().upper() for s in args.symbols.split(',') if s.strip()]
    ob = OrderbookCollector(symbols)
    try:
        ob.start()
    except KeyboardInterrupt:
        ob.stop()

