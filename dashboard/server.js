import express from 'express';
import fs from 'fs';
import path from 'path';
import readline from 'readline';

const app = express();
const ROOT = path.resolve(process.cwd());
const LOG_FILE = path.join(ROOT, 'logs', 'engine.log');

// Determine the directory to serve static assets from. In production the
// dashboard is built into "dashboard/dist". When running locally, or if the
// build output is missing, fall back to serving the source "dashboard" folder
// so that an index page is still available.
const DIST_DIR = path.join(ROOT, 'dashboard', 'dist');
const STATIC_DIR = fs.existsSync(DIST_DIR)
  ? DIST_DIR
  : path.join(ROOT, 'dashboard');

app.get('/events', (req, res) => {
  const symbol = (req.query.symbol || '').toUpperCase();
  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');
  res.flushHeaders();

  let filePos = 0;

  const sendNewLines = () => {
    fs.stat(LOG_FILE, (err, stats) => {
      if (err) return;
      if (stats.size > filePos) {
        const stream = fs.createReadStream(LOG_FILE, { start: filePos, end: stats.size });
        const rl = readline.createInterface({ input: stream });
        rl.on('line', (line) => {
          const upper = line.toUpperCase();
          const match = !symbol || upper.includes(`| ${symbol} `) ||
            /^D=|^PCR |^ATM |^SCENARIO:|^ACTION:|^FINAL VERDICT/.test(line.trim());
          if (match) {
            res.write(`data: ${JSON.stringify({ line })}\n\n`);
          }
        });
        filePos = stats.size;
      }
    });
  };

  const interval = setInterval(sendNewLines, 1000);
  sendNewLines();

  req.on('close', () => clearInterval(interval));
});

app.use(express.static(STATIC_DIR));

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Dashboard server listening on port ${PORT}`);
});
