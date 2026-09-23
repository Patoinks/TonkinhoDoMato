// Explicitly authorized Hosting cleanup: preserve active channels and latest 3 versions.
const fs = require('node:fs');
const path = require('node:path');
const cli = path.join(process.env.APPDATA, 'npm/node_modules/firebase-tools/lib');
const auth = require(path.join(cli, 'auth.js'));
const { requireAuth } = require(path.join(cli, 'requireAuth.js'));
const api = require(path.join(cli, 'hosting/api.js'));
const { Client } = require(path.join(cli, 'apiv2.js'));
const { hostingApiOrigin } = require(path.join(cli, 'api.js'));
const site = 'anw-games';
const receiptPath = path.resolve(__dirname, '../perf_results/hosting-cleanup-20260923.json');
const id = v => v.name.split('/').pop();
(async () => {
  const options = { project: site, nonInteractive: true };
  const account = auth.getProjectDefaultAccount('C:/Users/Martim/Desktop/Github/ANW-TONKAI-RELEASE-20260922');
  if (account) auth.setActiveAccount(options, account);
  await requireAuth(options);
  const client = new Client({ urlPrefix: hostingApiOrigin(), apiVersion: 'v1beta1', auth: true });
  const channels = await api.listChannels(site, site);
  const active = new Set(channels.map(c => c.release?.version?.name?.split('/').pop()).filter(Boolean));
  if (!channels.some(c => c.name.endsWith('/live') && c.release?.version?.name)) throw Error('No live version found');
  let versions = [], token;
  do {
    const r = await client.get(`/sites/${site}/versions`, {queryParams: {pageSize: 100, ...(token ? {pageToken: token} : {})}});
    versions.push(...(r.body.versions || [])); token = r.body.nextPageToken;
  } while (token);
  const finalized = versions.filter(v => v.status === 'FINALIZED').sort((a,b) => b.createTime.localeCompare(a.createTime));
  const protectedIds = new Set([...active, ...finalized.slice(0,3).map(id)]);
  // A bounded first cleanup; retry deployment before deleting any more history.
  const candidates = finalized.filter(v => !protectedIds.has(id(v))).reverse().slice(0,12);
  const receipt = {site, startedAt: new Date().toISOString(), preserved: [...protectedIds],
    finalizedCount: finalized.length, candidates: candidates.map(v => ({id: id(v), created: v.createTime, bytes: v.versionBytes})), deleted: []};
  console.log(JSON.stringify(receipt));
  if (!process.argv.includes('--apply')) return;
  fs.writeFileSync(receiptPath, JSON.stringify(receipt, null, 2));
  for (const v of candidates) {
    const current = await api.listChannels(site, site);
    if (current.some(c => c.release?.version?.name?.split('/').pop() === id(v))) throw Error('Version became active: '+id(v));
    await client.delete(`/sites/${site}/versions/${id(v)}`);
    receipt.deleted.push({id: id(v), deletedAt: new Date().toISOString()});
    fs.writeFileSync(receiptPath, JSON.stringify(receipt, null, 2));
    console.log('Deleted inactive version '+id(v));
  }
  receipt.finishedAt = new Date().toISOString();
  fs.writeFileSync(receiptPath, JSON.stringify(receipt, null, 2));
})().catch(error => {console.error(error.message); process.exitCode = 1;});
