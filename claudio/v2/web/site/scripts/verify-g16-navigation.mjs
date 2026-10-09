import { chromium } from '@playwright/test';
import { readFileSync, writeFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import assert from 'node:assert/strict';

const out=process.env.G16_WEB_OUTPUT;
const require=createRequire(import.meta.url),firebase=process.env.FIREBASE_TOOLS_ROOT??'C:/Users/Martim/AppData/Roaming/npm/node_modules/firebase-tools';
const auth=require(firebase+'/lib/auth.js'),{Client}=require(firebase+'/lib/apiv2.js'),api=require(firebase+'/lib/hosting/api.js');
auth.setActiveAccount({},auth.getGlobalDefaultAccount());
const client=new Client({urlPrefix:'https://firebasehosting.googleapis.com',apiVersion:'v1beta1',auth:true});
const live=await api.getChannel('anw-games','anw-games','live'),version=live.release.version.name;
let files=[],pageToken='';do{const r=await client.get('/'+version+'/files',{queryParams:{pageSize:1000,pageToken}});files.push(...r.body.files);pageToken=r.body.nextPageToken??'';}while(pageToken);
const hashes=Object.fromEntries(files.map(f=>[f.path,f.hash]));
const initial=JSON.parse(readFileSync(out+'/hosting-initial-baseline.json','utf8'));
const publication=JSON.parse(readFileSync(out+'/hosting-publication.json','utf8'));
const intended=new Set(publication.changedExistingFiles);
for(const [path,hash]of Object.entries(initial.files))if(!intended.has(path))assert.equal(hashes[path],hash,path+' must retain its original content');
const original=readFileSync(out+'/public-index-before.html','utf8');
const browser=await chromium.launch({channel:'chrome',headless:true});
try {
  const page=await browser.newPage();const errors=[];page.on('pageerror',e=>errors.push(e.message));
  await page.goto('https://anw-games.web.app/');
  const rootScripts=await page.locator('script[src]').evaluateAll(nodes=>nodes.map(n=>new URL(n.src).pathname));
  for(const script of rootScripts)assert.ok(original.includes(script),'Homepage keeps its existing application bundle');
  await page.locator('a[href="/tonkai"]').first().click();
  await page.getByRole('heading',{name:'G16 v2',exact:true}).waitFor();
  await page.getByRole('button',{name:'Começar',exact:true}).waitFor();
  const chessScripts=await page.locator('script[src]').evaluateAll(nodes=>nodes.map(n=>new URL(n.src).pathname));
  assert.notDeepEqual(chessScripts,rootScripts);
  await page.locator('a[href="/"]').first().click();
  await page.waitForURL('https://anw-games.web.app/');
  assert.deepEqual(await page.locator('script[src]').evaluateAll(nodes=>nodes.map(n=>new URL(n.src).pathname)),rootScripts);
  assert.deepEqual(errors,[]);
  const report={version,preservedOriginalFiles:Object.keys(initial.files).length-intended.size,originalBundlePreserved:true,rootToG16Works:true,g16ToRootWorks:true,rootScripts,chessScripts,errors};
  writeFileSync(out+'/g16-v2-navigation.json',JSON.stringify(report,null,2));console.log(JSON.stringify(report));
}finally{await browser.close();}
