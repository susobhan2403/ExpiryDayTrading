import pandas as pd

from src.cli.replay_trend import replay


def test_replay_trend_metrics(tmp_path):
    # Create synthetic 1m OHLCV data: first upward trend then downward trend
    idx = pd.date_range('2025-01-01', periods=100, freq='1min')
    close = list(range(100))
    for i in range(50, 100):
        close[i] = 100 - (i - 50)
    data = {
        'ts': idx,
        'open': close,
        'high': close,
        'low': close,
        'close': close,
        'volume': [1] * 100,
    }
    df = pd.DataFrame(data)
    csv_path = tmp_path / 'bars.csv'
    df.to_csv(csv_path, index=False)
    metrics_path = tmp_path / 'metrics.csv'

    report = replay(str(csv_path), str(metrics_path), prom_port=None,
                    weights={1: 1.0}, threshold=0.0, confirm=1)

    metrics = pd.read_csv(metrics_path)
    assert report['flip_count'] == metrics['flip_count'].iloc[0]
    assert metrics['flip_count'].iloc[0] >= 1
    assert metrics['avg_holding_seconds'].iloc[0] >= 0
