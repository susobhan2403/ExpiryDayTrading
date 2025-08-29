import { useEffect, useState } from 'react';

const indicatorForLine = (line) => {
  const t = line.trim();
  if (/\d{2}:\d{2} IST \| [A-Z]+/.test(t)) return 'spot';
  if (t.startsWith('D=')) return 'distance';
  if (t.startsWith('PCR')) return 'pcr';
  if (t.startsWith('ATM')) return 'atm';
  if (t.startsWith('Scenario:')) return 'scenario';
  if (t.startsWith('Action:')) return 'action';
  if (t.startsWith('Final Verdict')) return 'decision';
  return 'misc';
};

const colorize = (line) => {
  const t = line.trim();
  if (t.startsWith('Action:')) {
    if (t.includes('TRADE')) return `<span class="action-trade">${line}</span>`;
    return line;
  }
  if (t.startsWith('Final Verdict')) {
    let l = line.replace('Final Verdict', 'Decision');
    if (l.includes('Enter Now')) return `<span class="decision-enter">${l}</span>`;
    if (l.includes('Exit Now')) return `<span class="decision-exit">${l}</span>`;
    return `<span class="decision-hold">${l}</span>`;
  }
  return line.replace(/EXIT NOW/g, '<span class="exit">EXIT NOW</span>');
};

export default function useLogStream(symbol) {
  const [lines, setLines] = useState({});
  const [spot, setSpot] = useState(null);

  useEffect(() => {
    const es = new EventSource(`/events?symbol=${symbol}`);
    let prev = null;
    es.onmessage = (e) => {
      try {
        const { line } = JSON.parse(e.data);
        const indicator = indicatorForLine(line);
        if (indicator === 'spot') {
          const m = line.match(/IST \| [A-Z]+\s+(\d+(?:\.\d+)?)/);
          if (m) {
            const price = parseFloat(m[1]);
            const diff = prev !== null ? price - prev : 0;
            const pct = prev !== null ? (diff / prev) * 100 : 0;
            prev = price;
            setSpot({ price, diff, pct });
          }
          return;
        }
        const colored = colorize(line);

        // Only keep the most recent line for each indicator and ignore
        // duplicates so that stale/noisy data doesn't accumulate.
        setLines((prevLines) => {
          const prevLine = prevLines[indicator] ? prevLines[indicator][0] : null;
          if (prevLine === colored) return prevLines;
          return { ...prevLines, [indicator]: [colored] };
        });
      } catch (err) {
        console.error(err);
      }
    };
    return () => es.close();
  }, [symbol]);

  return { ...lines, spot };
}
