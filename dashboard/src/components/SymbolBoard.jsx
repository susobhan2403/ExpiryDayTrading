import React from 'react';
import IndicatorPanel from './IndicatorPanel';
import useLogStream from '../hooks/useLogStream';

const INDICATORS = [
  { key: 'distance', title: 'Distance' },
  { key: 'pcr', title: 'PCR' },
  { key: 'atm', title: 'ATM' },
  { key: 'scenario', title: 'Scenario' },
  { key: 'action', title: 'Action' },
  { key: 'decision', title: 'Decision' },
];

export default function SymbolBoard({ symbol }) {
  const data = useLogStream(symbol);
  const spot = data.spot;
  return (
    <div className="symbol-board">
      <h3 className="symbol">
        {symbol}
        {spot && (
          <>
            {' '}
            <span className="spot-price">{spot.price.toFixed(2)}</span>
            {' '}
            <span className={`spot-change ${spot.diff >= 0 ? 'up' : 'down'}`}>
              {spot.diff >= 0 ? '+' : ''}{spot.diff.toFixed(2)} ({spot.pct.toFixed(2)}%) {spot.diff > 0 ? '\u25B2' : (spot.diff < 0 ? '\u25BC' : '')}
            </span>
          </>
        )}
      </h3>
      <div className="indicators">
        {INDICATORS.map(({ key, title }) => (
          <IndicatorPanel key={key} title={title} lines={data[key] || []} />
        ))}
      </div>
    </div>
  );
}
