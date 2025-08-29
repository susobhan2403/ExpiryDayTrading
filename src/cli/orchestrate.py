from __future__ import annotations
import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
import datetime as dt
import pathlib
from typing import List, Optional

# Paths
ROOT = pathlib.Path(__file__).resolve().parents[2]


def load_settings() -> dict:
    cfg = {}
    try:
        p = ROOT / 'settings.json'
        if p.exists():
            cfg = json.loads(p.read_text())
    except Exception:
        cfg = {}
    return cfg or {}


def choose(param: Optional[str], *candidates, default: Optional[str] = None):
    for c in (param,) + candidates:
        if isinstance(c, str) and c:
            return c
        if isinstance(c, (int, float, bool)):
            return c
    return default


class Orchestrator:
    def __init__(self, symbols: List[str], mode: str, poll_seconds: int,
                 use_telegram: bool, slack_webhook: str,
                 provider: str = 'KITE',
                 run_stream: bool = True,
                 run_aggregator: bool = True,
                 aggregate_interval: int = 60,
                 engine_run_once: bool = False,
                 eod_train_time: Optional[str] = None,
                 eod_symbols: Optional[List[str]] = None,
                 eod_k_atr: float = 1.0,
                 eod_time_barrier: int = 15,
                 eod_calibrate: str = 'platt',
                 ):
        self.symbols = [s.upper() for s in symbols]
        self.mode = mode
        self.poll_seconds = int(poll_seconds)
        self.use_telegram = bool(use_telegram)
        self.slack_webhook = slack_webhook or ''
        self.provider = provider
        self.run_stream = run_stream
        self.run_aggregator = run_aggregator
        self.aggregate_interval = int(aggregate_interval)
        self.engine_run_once = bool(engine_run_once)
        self.proc_stream: Optional[subprocess.Popen] = None
        self.proc_engine: Optional[subprocess.Popen] = None
        self.stop_event = threading.Event()
        self.agg_thread: Optional[threading.Thread] = None
        self.eod_thread: Optional[threading.Thread] = None
        self.eod_train_time = eod_train_time  # 'HH:MM' IST string
        self.eod_symbols = eod_symbols or self.symbols
        self.eod_k_atr = float(eod_k_atr)
        self.eod_time_barrier = int(eod_time_barrier)
        self.eod_calibrate = eod_calibrate

    def start_stream(self):
        if not self.run_stream:
            return
        cmd = [sys.executable, '-m', 'src.stream.orderbook', '--symbols', ','.join(self.symbols)]
        self.proc_stream = subprocess.Popen(cmd, cwd=str(ROOT))
        print('[orchestrate] started stream:', cmd)

    def start_aggregator(self):
        if not self.run_aggregator:
            return
        def loop():
            from src.stream.aggregate import aggregate_day
            while not self.stop_event.is_set():
                try:
                    date_tag = dt.datetime.now().strftime('%Y%m%d')
                    aggregate_day(date_tag, self.symbols)
                except Exception as e:
                    print('[orchestrate] aggregator error:', e)
                # sleep
                for _ in range(self.aggregate_interval):
                    if self.stop_event.is_set():
                        break
                    time.sleep(1)
        self.agg_thread = threading.Thread(target=loop, daemon=True)
        self.agg_thread.start()
        print('[orchestrate] started aggregator: interval', self.aggregate_interval, 's')

    def start_engine(self):
        cmd = [sys.executable, str(ROOT / 'engine.py'),
               '--symbols', ','.join(self.symbols),
               '--provider', self.provider,
               '--poll-seconds', str(self.poll_seconds),
               '--mode', self.mode]
        if self.use_telegram:
            cmd.append('--use-telegram')
        if self.slack_webhook:
            cmd += ['--slack-webhook', self.slack_webhook]
        if self.engine_run_once:
            cmd.append('--run-once')
        self.proc_engine = subprocess.Popen(cmd, cwd=str(ROOT))
        print('[orchestrate] started engine:', cmd)

    def _parse_hhmm(self, hhmm: str) -> tuple[int,int]:
        s = (hhmm or '').strip()
        if not s:
            raise ValueError('empty time string')
        s = s.replace('.', ':')
        if ':' in s:
            parts = s.split(':')
            if len(parts) == 2:
                return int(parts[0]), int(parts[1])
            # tolerate hour only before colon
            return int(parts[0]), int(parts[1] if parts[1] else 0)
        # no colon: allow HHMM or HH
        if len(s) == 4 and s.isdigit():
            return int(s[:2]), int(s[2:])
        if s.isdigit():
            return int(s), 0
        raise ValueError(f'invalid time format: {hhmm}')

    def _seconds_until_ist(self, hhmm: str) -> int:
        try:
            hh, mm = self._parse_hhmm(hhmm)
            import pytz
            IST = pytz.timezone('Asia/Kolkata')
            now = dt.datetime.now(IST)
            target = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
            if target <= now:
                target = target + dt.timedelta(days=1)
            return int((target - now).total_seconds())
        except Exception:
            # Fallback naive
            hh, mm = self._parse_hhmm(hhmm)
            now = dt.datetime.now()
            target = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
            if target <= now:
                target = target + dt.timedelta(days=1)
            return int((target - now).total_seconds())

    def start_eod_trainer(self):
        if not self.eod_train_time:
            return
        def loop():
            while not self.stop_event.is_set():
                sec = self._seconds_until_ist(self.eod_train_time)
                # sleep in chunks
                while sec > 0 and not self.stop_event.is_set():
                    sl = min(60, sec)
                    time.sleep(sl); sec -= sl
                if self.stop_event.is_set():
                    break
                # EOD: build features and train for each symbol
                for sym in self.eod_symbols:
                    try:
                        print(f"[orchestrate] EOD build_features {sym}")
                        subprocess.run([sys.executable, '-m', 'src.cli.build_features', '--symbol', sym,
                                        '--k-atr', str(self.eod_k_atr), '--time-barrier', str(self.eod_time_barrier)],
                                       cwd=str(ROOT), check=False)
                        print(f"[orchestrate] EOD train_lgbm {sym}")
                        subprocess.run([sys.executable, '-m', 'src.cli.train_lgbm', '--symbol', sym,
                                        '--task', 'clf', '--target', 'label', '--calibrate', self.eod_calibrate],
                                       cwd=str(ROOT), check=False)
                    except Exception as e:
                        print('[orchestrate] EOD training error:', e)
        self.eod_thread = threading.Thread(target=loop, daemon=True)
        self.eod_thread.start()
        # normalize printout time
        try:
            hh, mm = self._parse_hhmm(self.eod_train_time)
            tstr = f"{hh:02d}:{mm:02d}"
        except Exception:
            tstr = str(self.eod_train_time)
        print('[orchestrate] scheduled EOD training at', tstr, 'IST for', ','.join(self.eod_symbols))

    def stop(self):
        print('[orchestrate] stopping...')
        self.stop_event.set()
        # stop engine
        if self.proc_engine and self.proc_engine.poll() is None:
            try:
                self.proc_engine.terminate()
            except Exception:
                pass
        # stop stream
        if self.proc_stream and self.proc_stream.poll() is None:
            try:
                self.proc_stream.terminate()
            except Exception:
                pass
        if self.agg_thread:
            self.agg_thread.join(timeout=5)
        if self.eod_thread:
            self.eod_thread.join(timeout=5)
        # wait processes
        if self.proc_engine:
            try:
                self.proc_engine.wait(timeout=10)
            except Exception:
                pass
        if self.proc_stream:
            try:
                self.proc_stream.wait(timeout=10)
            except Exception:
                pass
        print('[orchestrate] stopped')

    def run(self):
        def handle_sig(sig, frame):
            self.stop()
            sys.exit(0)
        signal.signal(signal.SIGINT, handle_sig)
        if hasattr(signal, 'SIGTERM'):
            signal.signal(signal.SIGTERM, handle_sig)

        self.start_stream()
        self.start_aggregator()
        self.start_eod_trainer()
        self.start_engine()
        try:
            # wait on engine
            if self.proc_engine is not None:
                self.proc_engine.wait()
        finally:
            self.stop()


def main():
    cfg = load_settings()
    # Defaults from settings.json if present
    orch_cfg = cfg.get('ORCHESTRATE', {}) if isinstance(cfg, dict) else {}

    ap = argparse.ArgumentParser(description='Intraday/Expiry Orchestrator')
    ap.add_argument('--symbols', default=orch_cfg.get('SYMBOLS', cfg.get('SYMBOLS', 'NIFTY,BANKNIFTY')))
    ap.add_argument('--mode', choices=['auto','intraday','expiry'], default=orch_cfg.get('MODE', 'auto'))
    ap.add_argument('--poll-seconds', type=int, default=int(orch_cfg.get('POLL_SECONDS', cfg.get('POLL_SECONDS', 60))))
    ap.add_argument('--provider', default=cfg.get('PROVIDER', 'KITE'))
    ap.add_argument('--use-telegram', action='store_true' if cfg.get('USE_TELEGRAM', False) else 'store_false')
    ap.add_argument('--slack-webhook', default=cfg.get('SLACK_WEBHOOK', ''))
    ap.add_argument('--no-stream', action='store_true', help='Disable orderbook stream')
    ap.add_argument('--no-aggregator', action='store_true', help='Disable 1m aggregator')
    ap.add_argument('--aggregate-interval', type=int, default=int(orch_cfg.get('AGGREGATE_INTERVAL', 60)))
    ap.add_argument('--engine-run-once', action='store_true', help='Forward --run-once to engine for off-hours debug')
    ap.add_argument('--eod-train', default=orch_cfg.get('EOD_TRAIN_TIME', ''), help="HH:MM IST time for EOD training (e.g., 16:05)")
    ap.add_argument('--eod-k-atr', type=float, default=float(orch_cfg.get('EOD_K_ATR', 1.0)))
    ap.add_argument('--eod-time-barrier', type=int, default=int(orch_cfg.get('EOD_TIME_BARRIER', 15)))
    ap.add_argument('--eod-calibrate', choices=['none','platt','isotonic'], default=orch_cfg.get('EOD_CALIBRATE', 'platt'))
    args = ap.parse_args()

    symbols = [s.strip().upper() for s in (args.symbols or '').split(',') if s.strip()]
    if not symbols:
        symbols = ['NIFTY']

    orch = Orchestrator(
        symbols=symbols,
        mode=args.mode,
        poll_seconds=args.poll_seconds,
        use_telegram=bool(cfg.get('USE_TELEGRAM', False) or args.use_telegram),
        slack_webhook=args.slack_webhook,
        provider=args.provider,
        run_stream=not args.no_stream,
        run_aggregator=not args.no_aggregator,
        aggregate_interval=args.aggregate_interval,
        engine_run_once=bool(args.engine_run_once),
        eod_train_time=(args.eod_train or None),
        eod_symbols=symbols,
        eod_k_atr=args.eod_k_atr,
        eod_time_barrier=args.eod_time_barrier,
        eod_calibrate=args.eod_calibrate,
    )
    orch.run()


if __name__ == '__main__':
    main()
