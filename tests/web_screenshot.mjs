#!/usr/bin/env node
// opens the web demo in headless chrome with webgpu, lets it render, saves a
// screenshot, printing page errors:
//
//     node tests/web_screenshot.mjs [--out shot.png] [--wait SECONDS] [--eval JS] [--clip x,y,w,h,scale] [--query QUERY]
//                                   [--software] [--check]
//
// serves web/ (models built by web/build.sh). needs google-chrome-stable and a
// gpu chrome can use, or --software: chrome's own renderer, for machines with no gpu (ci),
// where the page falls back to webgl2. --check fails unless something was drawn (a
// 'triangles drawn' over 0); page errors always fail.
import { spawn } from 'node:child_process';
import { createServer } from 'node:http';
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, extname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = join(dirname(fileURLToPath(import.meta.url)), '..', 'web');
const args = process.argv.slice(2);
let out = 'web.png', wait = 6, script = '', clip = null, query = '', software = false, check = false;
for (let k = 0; k < args.length; k++) {
  if (args[k] === '--out') out = args[++k];
  else if (args[k] === '--wait') wait = Number(args[++k]);
  else if (args[k] === '--eval') script = args[++k];
  else if (args[k] === '--query') query = args[++k];
  else if (args[k] === '--software') software = true;
  else if (args[k] === '--check') check = true;
  else if (args[k] === '--clip') { const [x, y, width, height, scale] = args[++k].split(',').map(Number); clip = { x, y, width, height, scale }; }
  else { console.error(`unknown flag ${args[k]}`); process.exit(2); }
}

const types = { '.js': 'text/javascript', '.wgsl': 'text/plain', '.html': 'text/html', '.css': 'text/css', '.gz': 'application/gzip' };
// files read once: a range request each read the whole 55 mb of pages again, a second of
// queueing per page that github pages never shows.
const files = new Map();
const server = createServer((req, res) => {
  let path = decodeURIComponent(new URL(req.url, 'http://x').pathname);
  if (path.endsWith('/')) path += 'index.html';
  try {
    if (!files.has(path)) files.set(path, readFileSync(join(root, path)));
    const body = files.get(path);
    const type = types[extname(path)] || 'application/octet-stream';
    // range requests, as github pages serves them: the demo streams pages.
    const range = /^bytes=(\d+)-(\d+)$/.exec(req.headers.range || '');
    if (range) {
      const start = Number(range[1]), end = Math.min(Number(range[2]), body.length - 1);
      res.writeHead(206, { 'content-type': type, 'content-length': end - start + 1, 'content-range': `bytes ${start}-${end}/${body.length}` });
      res.end(body.subarray(start, end + 1));
      return;
    }
    res.writeHead(200, { 'content-type': type, 'content-length': body.length });
    res.end(body);
  } catch { res.writeHead(404); res.end(); }
});
await new Promise((r) => server.listen(0, '127.0.0.1', r));
const port = server.address().port;
const profile = mkdtempSync(join(tmpdir(), 'colossus-web-'));
const chrome = spawn('google-chrome-stable', [
  '--headless=new', `--user-data-dir=${profile}`, '--remote-debugging-port=0', '--no-first-run', '--window-size=1600,900',
  ...(software ? ['--use-angle=swiftshader', '--enable-unsafe-swiftshader']
              : ['--enable-unsafe-webgpu', '--enable-features=Vulkan,SkiaGraphite', '--ignore-gpu-blocklist', '--use-angle=vulkan']),
  'about:blank',
], { stdio: ['ignore', 'ignore', 'pipe'] });
let failed = false;
try {
  const devtools = await new Promise((resolve, reject) => {
    let log = '';
    chrome.stderr.on('data', (d) => {
      log += d;
      const m = log.match(/DevTools listening on (ws:\/\/\S+)/);
      if (m) resolve(m[1]);
    });
    setTimeout(() => reject(new Error('Chrome did not start')), 20000);
  });
  console.error('devtools', devtools);
  const httpBase = devtools.replace(/^ws:\/\/([^/]+).*/, 'http://$1');
  const target = await (await fetch(`${httpBase}/json/new?about:blank`, { method: 'PUT' })).json();
  console.error('target', target.webSocketDebuggerUrl);
  const ws = new WebSocket(target.webSocketDebuggerUrl);
  let id = 0;
  const waits = new Map();
  ws.onmessage = (m) => {
    const d = JSON.parse(m.data);
    if (d.id && waits.has(d.id)) { waits.get(d.id)(d); waits.delete(d.id); }
    else if (d.method === 'Runtime.exceptionThrown') { failed = true; console.error('page:', d.params.exceptionDetails.exception?.description); }
    else if (d.method === 'Runtime.consoleAPICalled' && ['error', 'warning'].includes(d.params.type)) {
      failed ||= d.params.type === 'error';
      console.error('page:', d.params.args.map((a) => a.value ?? a.description).join(' '));
    }
  };
  const send = (method, params = {}, seconds = 20) => new Promise((resolve, reject) => {
    const i = ++id;
    const timer = setTimeout(() => reject(new Error(`${method} timed out`)), seconds * 1000);
    waits.set(i, (d) => { clearTimeout(timer); resolve(d); });
    ws.send(JSON.stringify({ id: i, method, params }));
  });
  await new Promise((r) => (ws.onopen = r));
  console.error('connected');
  await send('Runtime.enable');
  console.error('runtime enabled');
  console.error('navigate:', JSON.stringify(await send('Page.navigate', { url: `http://127.0.0.1:${port}/${query ? '?' + query : ''}` })));
  for (let t = 0; t < wait; t++) {
    await new Promise((r) => setTimeout(r, 1000));
    const s = await send('Runtime.evaluate', { expression: 'document.readyState + " " + (document.getElementById("loading")?.textContent ?? "loaded")', returnByValue: true }, 5)
      .catch((e) => ({ result: { result: { value: e.message } } }));
    console.error(`${t + 1}s:`, s.result?.result?.value);
  }
  if (script) {
    await send('Runtime.evaluate', { expression: script, awaitPromise: true });
    await new Promise((r) => setTimeout(r, 1500));
  }
  const text = await send('Runtime.evaluate', {
    expression: `[...document.querySelectorAll('dd')].map(d => d.previousElementSibling.textContent + ': ' + d.textContent).join('\\n') + '\\n' + (document.getElementById('loading')?.textContent || '')`,
    returnByValue: true,
  });
  console.log(text.result.result.value);
  if (check) {
    const drawn = /triangles drawn: ([\d.,]+)\s*([kMB]?)/.exec(text.result.result.value);
    const n = drawn ? parseFloat(drawn[1].replace(/,/g, '')) : 0;
    if (!(n > 0)) {
      failed = true;
      console.error('check: nothing drawn');
    }
  }
  const shot = await send('Page.captureScreenshot', clip ? { format: 'png', clip } : { format: 'png' });
  writeFileSync(out, Buffer.from(shot.result.data, 'base64'));
  console.log(`wrote ${out}`);
  ws.close();
} finally {
  chrome.kill();
  server.close();
  await new Promise((r) => setTimeout(r, 500));
  rmSync(profile, { recursive: true, force: true, maxRetries: 5, retryDelay: 200 });
}
process.exit(failed ? 1 : 0);
