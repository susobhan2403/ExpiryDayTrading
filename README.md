# ExpiryDayTrading - Enhanced Trading Engine

A modern, modular trading engine for Indian index options with enhanced capabilities.

## Quick Start

### Running the Enhanced Engine

```bash
# Basic usage
python engine_runner.py --symbols NIFTY,BANKNIFTY --run-once

# With specific polling interval
python engine_runner.py --symbols NIFTY --poll-seconds 30

# See all options
python engine_runner.py --help
```

### Entry Points

- **Primary CLI**: `engine_runner.py` - Enhanced trading engine runner
- **Core Engine**: `src/engine_enhanced.py` - Main enhanced trading engine implementation

## Architecture

The enhanced engine uses a modular architecture with the following key components:

### Core Components
- `src/engine_enhanced.py` - Enhanced trading engine with India-specific conventions
- `src/config.py` - Configuration management
- `engine_runner.py` - CLI interface and orchestration

### Calculations & Metrics
- `src/calculations/` - Core calculations (ATM, IV, Max Pain, PCR)
- `src/metrics/` - Enhanced metrics framework
- `src/features/` - Options processing and robust metrics

### Strategy & Decision Making
- `src/strategy/enhanced_gates.py` - Multi-factor gating with regime detection
- `src/strategy/scenario_classifier.py` - Advanced scenario classification

### Data & Providers
- `src/provider/kite.py` - Kite Connect API integration
- `src/provider/option_chain_builder.py` - Efficient option chain construction

### Output & Observability
- `src/output/` - Enhanced logging and formatting
- `src/observability/` - Comprehensive decision explanations

## Configuration Band presets

Band width defaults are read from `settings.json`.  Top-level keys such as
`BAND_MAX_STRIKES_ABOVE`, `BAND_MAX_STRIKES_BELOW`, `FAR_OTM_FILTER_POINTS`
and `PIN_DISTANCE_POINTS` provide global defaults.  To customise these for a
specific index, add an entry under the `PRESETS` section keyed by the index
symbol:

```json
{
  "BAND_MAX_STRIKES_ABOVE": 4,
  "BAND_MAX_STRIKES_BELOW": 3,
  "FAR_OTM_FILTER_POINTS": 800,
  "PIN_DISTANCE_POINTS": 150,
  "PRESETS": {
    "NIFTY": {
      "BAND_MAX_STRIKES_ABOVE": 5,
      "BAND_MAX_STRIKES_BELOW": 4,
      "FAR_OTM_FILTER_POINTS": 700,
      "PIN_DISTANCE_POINTS": 120
    }
  }
}
```

In this example, NIFTY uses the custom values while other symbols fall back to
the global defaults.

## Prometheus Metrics

The engine exposes runtime metrics using the [Prometheus Python client](https://github.com/prometheus/client_python). When the engine starts it runs a lightweight HTTP server (port `8000` by default) that serves metrics at `/metrics`.

### Available Gauges

| Metric | Description |
| --- | --- |
| `vnd` | Volatility normalised distance from MaxPain |
| `mph_norm` | Normalised MaxPain drift speed |
| `iv_z` | ATM implied volatility z-score |
| `pcr_z` | ΔPCR z-score |
| `spread_pct` | Bid/ask spread as a fraction of price |
| `depth_stability` | Quote stability over recent seconds |
| `scenario_probability{scenario="..."}` | Probability for each trading scenario |

`METRICS_PORT` environment variable can be used to change the listening port.

### Grafana

Add a Prometheus data source pointing at the engine host. Example panels:

* **Gauge** – track `vnd` to monitor distance from MaxPain.
* **Time-series** – `scenario_probability{scenario="Squeeze continuation (one-way)"}` to visualise scenario confidence.
* **Table** – query `scenario_probability` and group by `scenario` to list all current scenario probabilities.

These widgets provide quick operational insight into market conditions without inspecting logs.

## LLM Model Configuration

Optional features that rely on large language models default to OpenAI's
`gpt-4o-mini`.  The model can be changed by setting an `OPENAI_MODEL` value in
`settings.json` or by exporting an `OPENAI_MODEL` environment variable.  For
example, to switch to the upcoming `gpt-5` model:

```json
{
  "OPENAI_MODEL": "gpt-5"
}
```

This allows the engine to adopt newer models as they become available without
code changes.
