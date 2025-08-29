import React from 'react';
import IndicatorPanel from './IndicatorPanel';
import useLogStream from '../hooks/useLogStream';

const INDICATORS = ['distance','pcr','atm','scenario','action','verdict'];

export default function SymbolBoard({ symbol }) {
  const linesByIndicator = useLogStream(symbol);
  return (
    <div className="symbol-board">
      <h3 className="symbol">{symbol}</h3>
      <div className="indicators">
        {INDICATORS.map(ind => (
          <IndicatorPanel key={ind} title={ind} lines={linesByIndicator[ind] || []} />
        ))}
      </div>
    </div>
  );
}
