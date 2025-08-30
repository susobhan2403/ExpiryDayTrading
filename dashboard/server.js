import http from 'http';
import fs from 'fs';
import path from 'path';
import url from 'url';
import readline from 'readline';
import { fileURLToPath } from 'url';

// Resolve paths relative to the location of this file so that the server works
// regardless of the current working directory (e.g. when launched via npm
// scripts from within the "dashboard" folder).
const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, '..');
const LOG_FILE = path.join(ROOT, 'logs', 'engine.log');

// Initialize Kite client for prev-close queries
async function initKite() {
  let apiKey = process.env.KITE_API_KEY;
  if (!apiKey) {
    try {
      const cfg = JSON.parse(fs.readFileSync(path.join(ROOT, 'settings.json'), 'utf8'));
      apiKey = cfg.KITE_API_KEY || '';
    } catch {}
  }
  if (!apiKey) return null;
  try {
    const { KiteConnect } = await import('kiteconnect');
    const sess = JSON.parse(fs.readFileSync(path.join(ROOT, '.kite_session.json'), 'utf8'));
    const kc = new KiteConnect({ api_key: apiKey });
    kc.setAccessToken(sess.access_token);
    return kc;
  } catch {
    return null;
  }
}

function indexQuoteKey(symbol) {
  const s = symbol.toUpperCase();
  if (s === 'BANKNIFTY') return 'NSE:NIFTY BANK';
  if (s === 'NIFTY') return 'NSE:NIFTY 50';
  if (s === 'FINNIFTY') return 'NSE:NIFTY FIN SERVICE';
  if (s === 'MIDCPNIFTY') return 'NSE:NIFTY MID SELECT';
  if (s === 'SENSEX') return 'BSE:SENSEX';
  return 'NSE:NIFTY 50';
}

// Extract the previous trading session's closing spot from the rollup CSV
// emitted by the engine.  This mirrors the fallback logic used by the
// /prevclose endpoint so that the spot difference can still be computed when
// Kite quotes are unavailable or incomplete.
function prevCloseCsv(symbol) {
  try {
    const f = path.join(ROOT, 'out', `${symbol}_rollup.csv`);
    if (!fs.existsSync(f)) return null;
    const txt = fs.readFileSync(f, 'utf8').trim();
    const lines = txt.split(/\r?\n/);
    if (lines.length <= 1) return null; // header only
    let lastDay = null;
    for (let i = lines.length - 1; i >= 1; i--) {
      const arr = lines[i].split(',');
      if (arr.length < 2) continue;
      const ts = arr[0];
      const day = ts.slice(0, 10);
      const hour = parseInt(ts.slice(11, 13), 10);
      if (!lastDay) { lastDay = day; continue; }
      if (day !== lastDay && hour < 16) {
        const spot = parseFloat(arr[1]);
        return Number.isFinite(spot) ? spot : null;
      }
    }
    return null;
  } catch {
    return null;
  }
}

const kite = await initKite();

// Determine the directory to serve static assets from. In production the
// dashboard is built into "dashboard/dist". When running locally, or if the
// build output is missing, fall back to serving the source "dashboard" folder
// so that an index page is still available.
const DIST_DIR = path.join(__dirname, 'dist');
const STATIC_DIR = fs.existsSync(DIST_DIR)
  ? DIST_DIR
  : __dirname;

const server = http.createServer(async (req, res) => {
  const parsed = url.parse(req.url, true);

  if (parsed.pathname === '/prevclose') {
    const symbol = (parsed.query.symbol || '').toUpperCase();

    // Helper: CSV fallback from engine rollup when Kite is unavailable
    const csvFallback = () => {
      try {
        const f = path.join(ROOT, 'out', `${symbol}_rollup.csv`);
        if (!fs.existsSync(f)) return null;
        const txt = fs.readFileSync(f, 'utf8').trim();
        const lines = txt.split(/\r?\n/);
        if (lines.length <= 1) return null; // header only
        let lastDay = null;
        for (let i = lines.length - 1; i >= 1; i--) { // skip header at 0
          const arr = lines[i].split(',');
          if (arr.length < 2) continue;
          const ts = arr[0];
          const day = ts.slice(0, 10);
          const hour = parseInt(ts.slice(11, 13), 10);
          if (!lastDay) { lastDay = day; continue; }
          if (day !== lastDay && hour < 16) {
            // Use the last spot from the previous trading session before the
            // market close (roughly 16:00 IST). Some rollup files contain
            // after-hours entries which previously caused the fallback to
            // return near-current prices, yielding tiny or zero differences.
            const spot = parseFloat(arr[1]);
            return Number.isFinite(spot) ? spot : null;
          }
        }
        return null;
      } catch {
        return null;
      }
    };

    // Primary: Kite quote ohlc.close
    try {
      if (!kite) throw new Error('kite not initialized');
      const key = indexQuoteKey(symbol);
      const q = await kite.quote([key]);
      const close = q[key]?.ohlc?.close;
      if (typeof close === 'number' && isFinite(close)) {
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ close }));
        return;
      }
      // fallback if close missing
      const fb = csvFallback();
      if (fb !== null) {
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ close: fb }));
        return;
      }
      throw new Error('close unavailable');
    } catch (err) {
      const fb = csvFallback();
      if (fb !== null) {
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ close: fb, note: 'csv_fallback' }));
        return;
      }
      res.writeHead(500, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: String(err) }));
    }
    return;
  }

  if (parsed.pathname === '/spotdiff') {
    const symbol = (parsed.query.symbol || '').toUpperCase();
    try {
      if (!kite) throw new Error('kite not initialized');
      const key = indexQuoteKey(symbol);
      const q = await kite.quote([key]);
      const node = q[key] || {};

      // Prefer computing the difference from ``last_price`` and the
      // previous close as those fields are always provided for indices.
      // Earlier logic relied on ``net_change``/``change`` which are often
      // omitted by the API, causing the dashboard to show 0.00 (0.00%).
      let last = node.last_price;
      let close = node.ohlc?.close;

      // If the previous close is missing (e.g., when Kite is unavailable),
      // fall back to the rollup CSV produced by the engine.
      if (!(typeof close === 'number' && isFinite(close))) {
        const fb = prevCloseCsv(symbol);
        if (typeof fb === 'number' && isFinite(fb)) close = fb;
      }

      let diff =
        typeof last === 'number' && isFinite(last) &&
        typeof close === 'number' && isFinite(close)
          ? last - close
          : undefined;
      let pct =
        typeof diff === 'number' && typeof close === 'number'
          ? (diff / close) * 100
          : undefined;

      // As a final fallback, use Kite's ``net_change``/``change`` fields if
      // they exist.
      if (
        !(typeof diff === 'number' && isFinite(diff) &&
          typeof pct === 'number' && isFinite(pct))
      ) {
        diff = node.net_change;
        pct = node.change;
      }

      if (
        typeof diff === 'number' && isFinite(diff) &&
        typeof pct === 'number' && isFinite(pct)
      ) {
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ diff, pct }));
        return;
      }
      throw new Error('diff unavailable');
    } catch (err) {
      res.writeHead(500, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: String(err) }));
    }
    return;
  }

  if (parsed.pathname === '/events') {
    const symbol = (parsed.query.symbol || '').toUpperCase();
    res.writeHead(200, {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      'Connection': 'keep-alive'
    });

    // Start reading near the end of the log file so that a fresh client
    // immediately receives the most recent log lines rather than waiting
    // for new data to arrive.  Read roughly the last 64KB which should
    // contain plenty of recent entries without replaying the entire
    // history for large log files.
    let filePos = 0;
    try {
      const size = fs.statSync(LOG_FILE).size;
      filePos = Math.max(0, size - 64 * 1024);
    } catch (err) {
      filePos = 0;
    }
    let currentSymbol = null;
    const sendNewLines = () => {
      fs.stat(LOG_FILE, (err, stats) => {
        if (err) return;
        if (stats.size > filePos) {
          // Read only the newly appended portion of the log file. The `end`
          // option is inclusive, so subtract one to avoid re-reading the last
          // byte which would otherwise cause duplicate lines to be emitted.
          const stream = fs.createReadStream(LOG_FILE, { start: filePos, end: stats.size - 1 });
          filePos = stats.size;
          const rl = readline.createInterface({ input: stream });
          rl.on('line', (line) => {
            // Remove ANSI color codes before processing so that pattern
            // matching works even if the log writer included escape
            // sequences for styling (e.g. "\x1b[31m"). Also strip any
            // carriage returns so that the same line isn't displayed twice
            // in the dashboard.
            const plain = line
              .replace(/\x1b\[[0-9;]*m/g, '')
              .replace(/\r/g, '');
            const upper = plain.toUpperCase();
            const header = upper.match(/IST \| ([A-Z]+)/);
            if (header) {
              currentSymbol = header[1];
            }
            const isHeader = /IST \| [A-Z]+/.test(upper);
            const indicator = isHeader || /^D=|^PCR |^ATM |^SCENARIO:|^ACTION:|^FINAL VERDICT/.test(upper.trim());
            const match = (!symbol || currentSymbol === symbol) && indicator;
            if (match) {
              res.write(`data: ${JSON.stringify({ line: plain })}\n\n`);
            }
          });
          rl.on('close', () => {});
        }
      });
    };
    const interval = setInterval(sendNewLines, 1000);
    sendNewLines();
    req.on('close', () => clearInterval(interval));
    return;
  }

  // Serve static assets
  let pathname = parsed.pathname || '/';
  if (pathname === '/') pathname = '/index.html';
  const filePath = path.join(STATIC_DIR, pathname);
  fs.readFile(filePath, (err, data) => {
    if (err) {
      res.writeHead(404);
      res.end('Not Found');
      return;
    }
    const ext = path.extname(filePath).toLowerCase();
    const type = ext === '.html' ? 'text/html' :
                 ext === '.js' ? 'text/javascript' :
                 ext === '.css' ? 'text/css' :
                 'application/octet-stream';
    res.writeHead(200, { 'Content-Type': type });
    res.end(data);
  });
});

const PORT = process.env.PORT || 3000;
server.listen(PORT, () => {
  console.log(`Dashboard server listening on port ${PORT}`);
});
