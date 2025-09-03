#!/usr/bin/env python3
"""
Enhanced Engine CLI Runner

This script provides a command-line interface for the EnhancedTradingEngine,
maintaining compatibility with the existing orchestrator while using the
new enhanced engine capabilities.
"""

from __future__ import annotations
import argparse
import datetime as dt
import json
import logging
import os
import pathlib
import sys
import time
from typing import List, Optional

import pytz

from src.config import load_settings
from src.engine_enhanced import EnhancedTradingEngine, MarketData, TradingDecision
from src.provider.kite import KiteProvider

# Constants
IST = pytz.timezone("Asia/Kolkata")
ROOT = pathlib.Path(__file__).resolve().parent
OUT_DIR = ROOT / "out"
LOGS_DIR = ROOT / "logs"

# Defaults
DEFAULT_SYMBOLS = ["NIFTY", "BANKNIFTY"]
DEFAULT_PROVIDER = "KITE"
DEFAULT_POLL_SECS = 60


def setup_logging() -> logging.Logger:
    """Setup logging configuration."""
    LOGS_DIR.mkdir(exist_ok=True)
    
    logger = logging.getLogger("enhanced_engine")
    logger.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # File handler
    log_file = LOGS_DIR / "enhanced_engine.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger


def is_market_open() -> bool:
    """Check if market is currently open (IST)."""
    now = dt.datetime.now(IST)
    
    # Check if it's a weekday (Monday=0, Friday=4)
    if now.weekday() > 4:
        return False
    
    # Market hours: 9:15 AM to 3:30 PM IST
    market_start = now.replace(hour=9, minute=15, second=0, microsecond=0)
    market_end = now.replace(hour=15, minute=30, second=0, microsecond=0)
    
    return market_start <= now <= market_end


def get_next_expiry(symbol: str) -> dt.datetime:
    """Get next expiry for the symbol."""
    # This is a simplified implementation
    # In reality, this would fetch from provider or config
    now = dt.datetime.now(IST)
    
    # For weekly options, find next Thursday
    days_ahead = 3 - now.weekday()  # Thursday = 3
    if days_ahead <= 0:  # Already past Thursday this week
        days_ahead += 7
    
    next_expiry = now + dt.timedelta(days=days_ahead)
    expiry = next_expiry.replace(hour=15, minute=30, second=0, microsecond=0)
    
    return expiry


def create_sample_market_data(symbol: str, provider: KiteProvider) -> Optional[MarketData]:
    """Create sample market data for testing."""
    try:
        # Get spot price
        quotes = provider.get_quotes([symbol])
        if not quotes or symbol not in quotes:
            return None
        
        quote = quotes[symbol]
        spot = quote.get('last_price', 0.0)
        
        if spot <= 0:
            return None
        
        # Create basic market data
        timestamp = dt.datetime.now(IST)
        
        return MarketData(
            timestamp=timestamp,
            index=symbol,
            spot=spot,
            futures_mid=spot * 1.001,  # Simplified
            strikes=[spot - 100, spot - 50, spot, spot + 50, spot + 100],
            call_mids={},
            put_mids={},
            call_oi={},
            put_oi={}
        )
    
    except Exception as e:
        logging.getLogger("enhanced_engine").error(f"Error creating market data: {e}")
        return None


def run_engine_loop(
    symbols: List[str],
    provider_name: str,
    poll_seconds: int,
    run_once: bool,
    mode: str,
    logger: logging.Logger
) -> None:
    """Run the main engine loop."""
    
    logger.info(f"Starting enhanced engine for symbols: {symbols}")
    logger.info(f"Provider: {provider_name}, Poll: {poll_seconds}s, Mode: {mode}")
    
    # Initialize provider
    if provider_name.upper() != "KITE":
        logger.error("Only KITE provider is supported")
        sys.exit(1)
    
    try:
        provider = KiteProvider()
    except Exception as e:
        logger.error(f"Failed to initialize provider: {e}")
        sys.exit(1)
    
    # Initialize engines for each symbol
    engines = {}
    for symbol in symbols:
        try:
            expiry = get_next_expiry(symbol)
            engine = EnhancedTradingEngine(
                index=symbol,
                expiry=expiry,
                min_tau_hours=2.0
            )
            engines[symbol] = engine
            logger.info(f"Initialized engine for {symbol}, expiry: {expiry}")
        except Exception as e:
            logger.error(f"Failed to initialize engine for {symbol}: {e}")
    
    if not engines:
        logger.error("No engines initialized successfully")
        sys.exit(1)
    
    # Main loop
    iteration = 0
    while True:
        iteration += 1
        start_time = time.time()
        
        try:
            # Check market hours (unless run-once for testing)
            if not run_once and not is_market_open():
                logger.info("Market closed. Waiting...")
                time.sleep(60)
                continue
            
            logger.info(f"Processing iteration {iteration}")
            
            # Process each symbol
            for symbol, engine in engines.items():
                try:
                    # Get market data
                    market_data = create_sample_market_data(symbol, provider)
                    if not market_data:
                        logger.warning(f"No market data for {symbol}")
                        continue
                    
                    # Process with engine
                    decision = engine.process_market_data(market_data)
                    
                    # Log decision
                    logger.info(
                        f"{symbol}: {decision.action} "
                        f"(confidence: {decision.confidence:.2f}, "
                        f"spot: {market_data.spot:.2f})"
                    )
                    
                    # Save output (simplified)
                    OUT_DIR.mkdir(exist_ok=True)
                    output_file = OUT_DIR / f"{symbol}_latest.json"
                    output_data = {
                        "timestamp": market_data.timestamp.isoformat(),
                        "symbol": symbol,
                        "spot": market_data.spot,
                        "decision": decision.action,
                        "confidence": decision.confidence,
                        "processing_time_ms": decision.processing_time_ms
                    }
                    
                    with open(output_file, 'w') as f:
                        json.dump(output_data, f, indent=2)
                
                except Exception as e:
                    logger.error(f"Error processing {symbol}: {e}")
            
            processing_time = time.time() - start_time
            logger.info(f"Iteration {iteration} completed in {processing_time:.2f}s")
            
            # Exit if run-once
            if run_once:
                logger.info("Run-once mode, exiting")
                break
            
            # Sleep until next poll
            sleep_time = max(0, poll_seconds - processing_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        except KeyboardInterrupt:
            logger.info("Received interrupt signal, shutting down...")
            break
        except Exception as e:
            logger.error(f"Unexpected error in main loop: {e}")
            time.sleep(10)  # Brief pause before continuing


def main():
    """Main function with CLI argument parsing."""
    
    # Load settings
    settings = load_settings()
    
    # Parse arguments
    ap = argparse.ArgumentParser(
        description="Enhanced Trading Engine CLI Runner"
    )
    ap.add_argument(
        "--symbols", 
        default=",".join(DEFAULT_SYMBOLS), 
        help="Comma-separated symbols, e.g., BANKNIFTY,NIFTY"
    )
    ap.add_argument(
        "--provider", 
        default=DEFAULT_PROVIDER, 
        help="Data provider (KITE only)"
    )
    ap.add_argument(
        "--poll-seconds", 
        type=int, 
        default=DEFAULT_POLL_SECS,
        help="Polling interval in seconds"
    )
    ap.add_argument(
        "--run-once", 
        action="store_true",
        help="Run once and exit (for testing)"
    )
    ap.add_argument(
        "--use-telegram", 
        action="store_true",
        help="Enable telegram notifications (not implemented)"
    )
    ap.add_argument(
        "--slack-webhook", 
        default=os.getenv("SLACK_WEBHOOK", ""),
        help="Slack webhook URL (not implemented)"
    )
    ap.add_argument(
        "--mode", 
        choices=["auto", "intraday", "expiry"], 
        default="auto",
        help="Trading mode"
    )
    ap.add_argument(
        "--replay", 
        help="Replay mode (not implemented)"
    )
    ap.add_argument(
        "--speed", 
        type=float, 
        default=1.0,
        help="Replay speed (not implemented)"
    )
    
    args = ap.parse_args()
    
    # Setup logging
    logger = setup_logging()
    
    # Validate arguments
    if args.replay:
        logger.error("Replay mode not implemented in enhanced engine")
        sys.exit(1)
    
    # Parse symbols
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    if not symbols:
        symbols = DEFAULT_SYMBOLS
    
    # Run engine
    try:
        run_engine_loop(
            symbols=symbols,
            provider_name=args.provider,
            poll_seconds=args.poll_seconds,
            run_once=args.run_once,
            mode=args.mode,
            logger=logger
        )
    except Exception as e:
        logger.error(f"Engine failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()