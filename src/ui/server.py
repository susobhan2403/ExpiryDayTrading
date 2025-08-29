from __future__ import annotations
import pathlib
import time
from typing import List

from flask import Flask, request, jsonify, render_template_string

ROOT = pathlib.Path(__file__).resolve().parents[2]
LOG_FILE = ROOT / 'logs' / 'engine.log'

TEMPLATE = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Index Monitor</title>
  <style>
    body { font-family: Arial, sans-serif; background: #0f111a; color: #e6e6e6; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
    .panel { border: 1px solid #333; padding: 12px; border-radius: 6px; background: #1b1e28; }
    .title { font-weight: bold; margin-bottom: 8px; }
    pre { white-space: pre-wrap; word-wrap: break-word; font-size: 12px; }
    .symbol { color: #80cbc4; }
    .exit { color: #ff5370; font-weight: bold; }
  </style>
  <script>
    const symbols = {{ symbols|tojson }};
    async function fetchLines(sym) {
      try{
        const res = await fetch('/api/lines?symbol=' + encodeURIComponent(sym));
        const js = await res.json();
        const el = document.getElementById('panel_' + sym);
        const lines = js.lines.join('\n');
        // simple highlight
        const html = lines.replaceAll('EXIT NOW', '<span class=\'exit\'>EXIT NOW</span>');
        el.innerHTML = html;
      } catch(e){ console.error(e); }
    }
    function tick(){ symbols.forEach(fetchLines); }
    setInterval(tick, 3000);
    window.onload = tick;
  </script>
  </head>
  <body>
    <h2>Index Monitor</h2>
    <div class="grid">
      {% for s in symbols %}
        <div class="panel">
          <div class="title"><span class="symbol">{{s}}</span></div>
          <pre id="panel_{{s}}">Loading...</pre>
        </div>
      {% endfor %}
    </div>
  </body>
  </html>
"""

app = Flask(__name__)

def tail_lines_for_symbol(symbol: str, max_lines: int = 50) -> List[str]:
    if not LOG_FILE.exists():
        return []
    try:
        data = LOG_FILE.read_text(encoding='utf-8', errors='ignore').splitlines()[-2000:]
    except Exception:
        data = []
    sym = symbol.upper()
    out: List[str] = []
    for ln in reversed(data):
        # naive filter: keep blocks starting headers or lines mentioning the symbol
        cond = (
            (f"| {sym} " in ln) or
            ln.strip().startswith('D=') or
            ln.strip().startswith('PCR ') or
            ln.strip().startswith('ATM ') or
            ln.strip().startswith('Scenario:') or
            ln.strip().startswith('Action:') or
            ('Enter when atleast' in ln) or
            ('Exit when atleast' in ln) or
            ('Final Verdict' in ln)
        )
        if cond:
            out.append(ln)
        if len(out) >= max_lines:
            break
    return list(reversed(out))

@app.route('/')
def index():
    syms = request.args.get('symbols', 'NIFTY,BANKNIFTY,SENSEX,MIDCPNIFTY')
    symbols = [s.strip().upper() for s in syms.split(',') if s.strip()]
    return render_template_string(TEMPLATE, symbols=symbols)

@app.route('/api/lines')
def api_lines():
    sym = request.args.get('symbol', 'NIFTY').upper()
    lines = tail_lines_for_symbol(sym)
    return jsonify({'symbol': sym, 'lines': lines})

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=8080)
    ap.add_argument('--symbols', default='NIFTY,BANKNIFTY,SENSEX,MIDCPNIFTY')
    args = ap.parse_args()
    app.config['SYMBOLS'] = args.symbols
    app.run(host=args.host, port=args.port, debug=False)

if __name__ == '__main__':
    main()
