import { useEffect, useState } from 'react';

const indicatorForLine = (line) => {
  const t = line.trim();
  if (t.startsWith('D=')) return 'distance';
  if (t.startsWith('PCR')) return 'pcr';
  if (t.startsWith('ATM')) return 'atm';
  if (t.startsWith('Scenario:')) return 'scenario';
  if (t.startsWith('Action:')) return 'action';
  if (t.startsWith('Final Verdict')) return 'verdict';
  return 'misc';
};

const colorize = (line) => {
  return line.replace(/EXIT NOW/g, '<span class="exit">EXIT NOW</span>');
};

export default function useLogStream(symbol) {
  const [lines, setLines] = useState({});

  useEffect(() => {
    const es = new EventSource(`/events?symbol=${symbol}`);
    es.onmessage = (e) => {
      try {
        const { line } = JSON.parse(e.data);
        const indicator = indicatorForLine(line);
        setLines((prev) => {
          const arr = prev[indicator] ? [...prev[indicator], colorize(line)] : [colorize(line)];
          return { ...prev, [indicator]: arr.slice(-50) };
        });
      } catch (err) {
        console.error(err);
      }
    };
    return () => es.close();
  }, [symbol]);

  return lines;
}
