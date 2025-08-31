# ExpiryDayTrading

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
