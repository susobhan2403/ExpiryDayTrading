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

// Determine the directory to serve static assets from. In production the
// dashboard is built into "dashboard/dist". When running locally, or if the
// build output is missing, fall back to serving the source "dashboard" folder
// so that an index page is still available.
const DIST_DIR = path.join(__dirname, 'dist');
const STATIC_DIR = fs.existsSync(DIST_DIR)
  ? DIST_DIR
  : __dirname;

const server = http.createServer((req, res) => {
  const parsed = url.parse(req.url, true);

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
