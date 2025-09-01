# ExpiryDayTrading

## Band presets

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
