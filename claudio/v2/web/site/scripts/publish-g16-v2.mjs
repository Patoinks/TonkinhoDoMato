/** Preserve the current Hosting release; replace only the chess entry and add v2 assets. */
import { createRequire } from 'node:module';
import { readFileSync, writeFileSync, readdirSync, existsSync } from 'node:fs';
import { Readable } from 'node:stream';
import { gzipSync } from 'node:zlib';
import { createHash } from 'node:crypto';
import assert from 'node:assert/strict';

const require=createRequire(import.meta.url);
const firebaseRoot=process.env.FIREBASE_TOOLS_ROOT ?? 'C:/Users/Martim/AppData/Roaming/npm/node_modules/firebase-tools';
const auth=require(firebaseRoot+'/lib/auth.js');
const {Client}=require(firebaseRoot+'/lib/apiv2.js');
const api=require(firebaseRoot+'/lib/hosting/api.js');
const output=process.env.G16_WEB_OUTPUT;
if(!output)throw new Error('G16_WEB_OUTPUT is required');
const account=auth.getGlobalDefaultAccount();
if(!account)throw new Error('Firebase login unavailable');
auth.setActiveAccount({project:'anw-games'},account);
const client=new Client({urlPrefix:'https://firebasehosting.googleapis.com',apiVersion:'v1beta1',auth:true});
const channel=await api.getChannel('anw-games','anw-games','live');
const version=channel.release.version;
const source=version.name;
const original=(await client.get('/'+source)).body;
let files=[],pageToken='';
do {
  const response=await client.get('/'+source+'/files',{queryParams:{pageSize:1000,pageToken}});
  files.push(...(response.body.files??[]));pageToken=response.body.nextPageToken??'';
}while(pageToken);
const oldFiles=Object.fromEntries(files.map(f=>[f.path,f.hash]));
const rootHtml=await (await fetch('https://anw-games.web.app/',{cache:'no-store'})).text();
if(!process.argv.includes('--publish')) {
  writeFileSync(output+'/hosting-baseline.json',JSON.stringify({source,config:original.config,files:oldFiles},null,2));
  writeFileSync(output+'/public-index-before.html',rootHtml);
  console.log(JSON.stringify({source,files:files.length,rewrites:original.config.rewrites,headers:original.config.headers?.length}));
  process.exit(0);
}
const baseline=JSON.parse(readFileSync(output+'/hosting-baseline.json','utf8'));
assert.equal(baseline.source,source,'Live release changed since preparation; inspect again first');
assert.equal(readFileSync(output+'/public-index-before.html','utf8'),rootHtml);
const reference=JSON.parse(readFileSync(output+'/browser-reference.json','utf8'));
assert.equal(reference.different,0);assert.ok(reference.decisions>=100);
const smoke=JSON.parse(readFileSync(output+'/g16-v2-browser-smoke.json','utf8'));
assert.deepEqual(smoke.errors,[]);assert.equal(smoke.games.length,2);
const manifest=JSON.parse(readFileSync('public/tonkai/g16-v2.json','utf8'));
assert.equal(smoke.sha256,manifest.sha256);
// The existing SPA's links must load the new dedicated chess entry from Hosting.
const intoChess='<script id="g16-v2-navigation">document.addEventListener("click",e=>{const a=e.target.closest?.("a[href]");if(!a||e.button||e.ctrlKey||e.metaKey||e.shiftKey||e.altKey)return;const u=new URL(a.href,location.href);if(u.origin===location.origin&&(u.pathname==="/tonkai"||u.pathname.startsWith("/tonkai/"))){e.preventDefault();e.stopImmediatePropagation();location.assign(u.href)}},true)</script>';
const outOfChess='<script id="g16-v2-navigation">document.addEventListener("click",e=>{const a=e.target.closest?.("a[href]");if(!a||e.button||e.ctrlKey||e.metaKey||e.shiftKey||e.altKey)return;const u=new URL(a.href,location.href);if(u.origin===location.origin&&!(u.pathname==="/tonkai"||u.pathname.startsWith("/tonkai/"))){e.preventDefault();e.stopImmediatePropagation();location.assign(u.href)}},true)</script>';
const upload=new Map(),newFiles={...oldFiles};
const add=(path,bytes)=>{
  const zipped=gzipSync(bytes,{level:9}),hash=createHash('sha256').update(zipped).digest('hex');
  newFiles[path]=hash;upload.set(hash,zipped);
};
const cleanRoot=rootHtml.replace(/<script id="g16-v2-navigation">[\s\S]*?<\/script>\n?/g,'');
add('/index.html',Buffer.from(cleanRoot.replace('</head>',intoChess+'\n</head>')));
add('/tonkai/index.html',Buffer.from(readFileSync('dist/index.html','utf8').replace('</head>',outOfChess+'\n</head>')));
for(const file of readdirSync('dist/assets'))if(!oldFiles['/assets/'+file])add('/assets/'+file,readFileSync('dist/assets/'+file));
for(const file of ['g16-v2.json','g16-v2-version.json','g16-rules.mjs','g16-rules.wasm','native-provenance.json',manifest.binary]) add('/tonkai/'+file,readFileSync('public/tonkai/'+file));
const config=structuredClone(original.config);
config.rewrites=[{glob:'/tonkai{,/**}',path:'/tonkai/index.html'},...(config.rewrites??[]).filter(rule=>rule.glob!=='/tonkai{,/**}')];
config.headers=[{glob:'/tonkai{,/**}',headers:{'Cache-Control':'no-cache, must-revalidate'}},...(config.headers??[]).filter(rule=>rule.glob!=='/tonkai{,/**}')];
const changed=Object.keys(oldFiles).filter(path=>oldFiles[path]!==newFiles[path]);
assert.ok(changed.every(path=>path==='/index.html'||path==='/tonkai/index.html'||/^\/tonkai\/(g16-v2(?:-version)?\.json|g16-rules\.(mjs|wasm)|native-provenance\.json)$/.test(path)), 'Unexpected change to an existing Hosting file');
const prepared=existsSync(output+'/hosting-prepared.json') ? JSON.parse(readFileSync(output+'/hosting-prepared.json','utf8')) : null;
const created=prepared?.source===source ? prepared.created : await api.createVersion('anw-games',{config,labels:{'deployment-tool':'g16-v2-selective'}});
writeFileSync(output+'/hosting-prepared.json',JSON.stringify({source,created,files:newFiles,config},null,2));
const entries=Object.entries(newFiles);
let uploaded=0;
for(let i=0;i<entries.length;i+=1000){
  const response=await client.post('/'+created+':populateFiles',{files:Object.fromEntries(entries.slice(i,i+1000))});
  const uploader=new Client({urlPrefix:response.body.uploadUrl,auth:true});
  for(const hash of response.body.uploadRequiredHashes??[]) {
    const bytes=upload.get(hash);
    if(!bytes)throw new Error('Hosting requested an existing file not in the upload set');
    const response=await uploader.post('/'+hash,Readable.from(bytes),{headers:{'Content-Type':'application/octet-stream'},responseType:'stream',resolveOnHTTPError:true});
    if(response.status>=400) {
      const chunks=[];for await(const chunk of response.body)chunks.push(chunk);
      throw new Error(`Hosting upload failed (${response.status}): ${Buffer.concat(chunks).toString().slice(0,1000)}`);
    }
    uploaded++;
  }
}
const id=created.split('/').at(-1);
await api.updateVersion('anw-games',id,{status:'FINALIZED',config});
// Guard against concurrent publishers just before activating this release.
const latest=await api.getChannel('anw-games','anw-games','live');
assert.equal(latest.release.version.name,source,'Another release appeared; new version is preserved but not activated');
const release=await api.createRelease('anw-games','live',created,{message:'G16 v2 checkpoint 600, full C++ WebAssembly motor, no training'});
const report={source,version:created,release:release.name,uploaded,files:Object.keys(newFiles).length,preservedFiles:files.length-changed.length,
  changedExistingFiles:changed,sha256:manifest.sha256,source600:manifest.source_sha256};
writeFileSync(output+'/hosting-publication.json',JSON.stringify(report,null,2));
console.log(JSON.stringify(report));
