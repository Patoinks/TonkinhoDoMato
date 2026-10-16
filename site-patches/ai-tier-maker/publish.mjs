import { createRequire } from 'node:module';
import { readFileSync, writeFileSync, readdirSync, mkdirSync, existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { join, dirname } from 'node:path';
import { createHash } from 'node:crypto';
import { gzipSync } from 'node:zlib';
import { Readable } from 'node:stream';
import assert from 'node:assert/strict';

const here = dirname(fileURLToPath(import.meta.url));
const out = join(here, 'evidence');
const staged = join(out, 'staged');
mkdirSync(staged, { recursive: true });
const require = createRequire(import.meta.url);
const firebase = 'C:/Users/Martim/AppData/Roaming/npm/node_modules/firebase-tools';
const auth = require(firebase + '/lib/auth.js');
const { Client } = require(firebase + '/lib/apiv2.js');
const api = require(firebase + '/lib/hosting/api.js');
const account = auth.getGlobalDefaultAccount();
assert.ok(account, 'Firebase login required');
auth.setActiveAccount({ project: 'anw-games' }, account);
const client = new Client({ urlPrefix: 'https://firebasehosting.googleapis.com', apiVersion: 'v1beta1', auth: true });
const liveVersion = async () => (await api.getChannel('anw-games', 'anw-games', 'live')).release.version.name;
const source = await liveVersion();
const original = (await client.get('/' + source)).body;
let entries = [], pageToken = '';
do {
  const r = await client.get('/' + source + '/files', { queryParams: { pageSize: 1000, pageToken } });
  entries.push(...(r.body.files ?? [])); pageToken = r.body.nextPageToken ?? '';
} while (pageToken);
const files = Object.fromEntries(entries.map(f => [f.path, f.hash]));
const json = (name, value) => writeFileSync(join(out, name), JSON.stringify(value, null, 2));
const getJson = name => JSON.parse(readFileSync(join(out, name), 'utf8'));
const sha = bytes => createHash('sha256').update(gzipSync(bytes, { level: 9 })).digest('hex');
const navTag = '<script defer src="/ai-tier-maker/site-navigation.js" data-ai-tier-navigation></script>';

if (process.argv.includes('--prepare')) {
  const replacements = {};
  const stage = (path, bytes) => {
    const target = join(staged, path);
    mkdirSync(dirname(target), { recursive: true }); writeFileSync(target, bytes);
    replacements[path] = sha(bytes);
  };
  // Only add a navigation script to the existing HTML; retain all app bundles.
  for (const path of Object.keys(files).filter(p => p.endsWith('.html') && !p.startsWith('/ai-tier-maker/'))) {
    const response = await fetch('https://anw-games.web.app' + path, { cache: 'no-store' });
    assert.equal(response.status, 200, path);
    const html = await response.text();
    if (!html.includes('id="root"')) continue;
    assert.ok(html.includes('</head>'), path);
    const next = html.replace(/<script[^>]*data-ai-tier-navigation[^>]*><\/script>\s*/g, '').replace('</head>', navTag + '\n</head>');
    stage(path, Buffer.from(next));
  }
  const walk = (dir, prefix) => {
    for (const e of readdirSync(dir, { withFileTypes: true })) {
      const path = prefix + '/' + e.name;
      if (e.isDirectory()) walk(join(dir, e.name), path);
      else stage(path, readFileSync(join(dir, e.name)));
    }
  };
  walk(join(here, 'public'), '/ai-tier-maker');
  const config = structuredClone(original.config);
  config.rewrites = [{ glob: '/ai-tier-maker{,/**}', path: '/ai-tier-maker/index.html' }, ...(config.rewrites ?? []).filter(r => r.glob !== '/ai-tier-maker{,/**}')];
  config.headers = [...(config.headers ?? []).filter(r => r.glob !== '/ai-tier-maker{,/**}'), { glob: '/ai-tier-maker{,/**}', headers: { 'Cache-Control': 'no-cache, must-revalidate' } }];
  const changed = Object.keys(files).filter(p => replacements[p] && replacements[p] !== files[p]);
  assert.ok(changed.every(p => p.endsWith('.html') || p.startsWith('/ai-tier-maker/')));
  json('baseline.json', { source, config: original.config, files });
  json('prepared.json', { source, config, replacements, changed, files: { ...files, ...replacements } });
  console.log(JSON.stringify({ source, changed, newFiles: Object.keys(replacements).filter(p => !files[p]), preserved: Object.keys(files).length - changed.length }));
} else if (process.argv.includes('--publish')) {
  const prepared = getJson('prepared.json');
  assert.equal(source, prepared.source, 'Live changed; prepare and validate again');
  const upload = new Map();
  for (const [path, hash] of Object.entries(prepared.replacements)) {
    const bytes = readFileSync(join(staged, path)); assert.equal(sha(bytes), hash);
    upload.set(hash, gzipSync(bytes, { level: 9 }));
  }
  const version = await api.createVersion('anw-games', { config: prepared.config, labels: { 'deployment-tool': 'ai-tier-maker-selective' } });
  const next = Object.entries(prepared.files);
  for (let i = 0; i < next.length; i += 1000) {
    const r = await client.post('/' + version + ':populateFiles', { files: Object.fromEntries(next.slice(i, i + 1000)) });
    const uploader = new Client({ urlPrefix: r.body.uploadUrl, auth: true });
    for (const hash of r.body.uploadRequiredHashes ?? []) {
      assert.ok(upload.has(hash), 'Existing live asset unexpectedly missing');
      const result = await uploader.post('/' + hash, Readable.from(upload.get(hash)), { headers: { 'Content-Type': 'application/octet-stream' }, responseType: 'stream', resolveOnHTTPError: true });
      assert.ok(result.status < 400, 'Hosting upload failed: ' + result.status);
    }
  }
  await api.updateVersion('anw-games', version.split('/').at(-1), { status: 'FINALIZED', config: prepared.config });
  assert.equal(await liveVersion(), source, 'Concurrent publication; new version not activated');
  const release = await api.createRelease('anw-games', 'live', version, { message: 'Add AI Tier Maker navigation tab and standalone tool' });
  const report = { source, version, release: release.name, changed: prepared.changed, preserved: Object.keys(files).length - prepared.changed.length };
  json('publication.json', report); console.log(JSON.stringify(report));
} else if (process.argv.includes('--verify')) {
  const publication = getJson('publication.json'), prepared = getJson('prepared.json'), baseline = getJson('baseline.json');
  assert.equal(source.split('/').at(-1), publication.version.split('/').at(-1));
  for (const [path, hash] of Object.entries(prepared.files)) assert.equal(files[path], hash, path);
  for (const path of Object.keys(prepared.replacements)) {
    const r = await fetch('https://anw-games.web.app' + path, { cache: 'no-store' }); assert.equal(r.status, 200);
    assert.deepEqual(Buffer.from(await r.arrayBuffer()), readFileSync(join(staged, path)), path);
  }
  const report = { version: source, allPublishedBytesMatch: true, preservedFiles: Object.keys(baseline.files).length - prepared.changed.length, modelsAndBundlesPreserved: true };
  json('verification.json', report); console.log(JSON.stringify(report));
} else throw new Error('Use --prepare, --publish or --verify');
