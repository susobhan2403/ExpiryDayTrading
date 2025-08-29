import React from 'react';
import SymbolBoard from './components/SymbolBoard';

const DEFAULT_SYMBOLS = ['NIFTY','BANKNIFTY','SENSEX','MIDCPNIFTY'];

export default function App() {
  return (
    <div className="app">
      <h2>Index Dashboard</h2>
      <div className="grid">
        {DEFAULT_SYMBOLS.map(sym => (
          <SymbolBoard key={sym} symbol={sym} />
        ))}
      </div>
    </div>
  );
}
