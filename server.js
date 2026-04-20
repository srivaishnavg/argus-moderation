#!/usr/bin/env node
/**
 * Argus Annotator — Local Dev Server
 * 
 * Serves the static app AND proxies image requests to cdn.stage.highfive-api.com
 * so the browser can load frame images without CORS errors.
 * 
 * Usage:
 *   node server.js
 *   node server.js --port 3001
 * 
 * Then open: http://localhost:3000
 */

const http  = require('http');
const https = require('https');
const fs    = require('fs');
const path  = require('path');
const url   = require('url');

const PORT = process.argv.includes('--port')
  ? +process.argv[process.argv.indexOf('--port') + 1]
  : 3000;

const MIME = {
  '.html': 'text/html',
  '.js':   'application/javascript',
  '.css':  'text/css',
  '.json': 'application/json',
  '.png':  'image/png',
  '.jpg':  'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.svg':  'image/svg+xml',
  '.ico':  'image/x-icon',
};

const server = http.createServer((req, res) => {
  const parsed = url.parse(req.url, true);
  const pathname = parsed.pathname;

  // ── CORS proxy: /proxy?url=https://cdn.stage.highfive-api.com/...
  if (pathname === '/proxy') {
    const target = parsed.query.url;
    if (!target) {
      res.writeHead(400); res.end('Missing url param'); return;
    }

    // Only allow proxying from known Argus CDN domains
    const allowed = ['cdn.stage.highfive-api.com', 'cdn.highfive-api.com', 'stage.highfive-api.com'];
    let targetHost;
    try { targetHost = new URL(target).hostname; } catch(e) {
      res.writeHead(400); res.end('Invalid url'); return;
    }
    if (!allowed.includes(targetHost)) {
      res.writeHead(403); res.end(`Domain ${targetHost} not in proxy allowlist`); return;
    }

    const options = url.parse(target);
    const protocol = target.startsWith('https') ? https : http;

    const proxyReq = protocol.get(options, proxyRes => {
      res.writeHead(proxyRes.statusCode, {
        'Content-Type':  proxyRes.headers['content-type'] || 'application/octet-stream',
        'Cache-Control': 'public, max-age=3600',
        'Access-Control-Allow-Origin': '*',
      });
      proxyRes.pipe(res);
    });

    proxyReq.on('error', err => {
      console.error('Proxy error:', err.message);
      res.writeHead(502); res.end('Proxy error: ' + err.message);
    });

    proxyReq.setTimeout(15000, () => {
      proxyReq.destroy();
      res.writeHead(504); res.end('Proxy timeout');
    });

    return;
  }

  // ── Static file server: /public/*
  let filePath = path.join(__dirname, 'public', pathname === '/' ? 'index.html' : pathname);

  fs.stat(filePath, (err, stat) => {
    if (err || !stat.isFile()) {
      // Fallback to index.html for SPA-style routing
      filePath = path.join(__dirname, 'public', 'index.html');
    }

    fs.readFile(filePath, (err, data) => {
      if (err) {
        res.writeHead(404); res.end('Not found'); return;
      }
      const ext  = path.extname(filePath);
      const mime = MIME[ext] || 'application/octet-stream';
      res.writeHead(200, {
        'Content-Type': mime,
        'Cross-Origin-Opener-Policy':   'same-origin',
        'Cross-Origin-Embedder-Policy': 'require-corp',
        'Access-Control-Allow-Origin':  '*',
      });
      res.end(data);
    });
  });
});

server.listen(PORT, () => {
  console.log('');
  console.log('  ██████╗  Argus Cluster Annotator');
  console.log('  ██╔══██╗ Local Dev Server');
  console.log('  ██████╔╝');
  console.log('  ██╔══██╗ http://localhost:' + PORT);
  console.log('  ██║  ██║ Proxy: http://localhost:' + PORT + '/proxy?url=...');
  console.log('  ╚═╝  ╚═╝');
  console.log('');
  console.log('  Open http://localhost:' + PORT + ' in your browser');
  console.log('  Frame images will be proxied to bypass CORS.');
  console.log('  Press Ctrl+C to stop.');
  console.log('');
});
