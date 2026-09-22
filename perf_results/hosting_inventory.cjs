const fs = require('node:fs');
const cli = 'C:/Users/Martim/AppData/Roaming/npm/node_modules/firebase-tools/lib/';
const auth = require(cli + 'auth.js');
const {requireAuth} = require(cli + 'requireAuth.js');
const api = require(cli + 'hosting/api.js');
(async () => {
  const options = {project:'anw-games', nonInteractive:true};
  const account = auth.getProjectDefaultAccount('C:/Users/Martim/Desktop/Github/ANW-TONKAI-RELEASE-20260922');
  if (account) auth.setActiveAccount(options,account);
  await requireAuth(options);
  const versions = await api.listVersions('anw-games');
  const channels = await api.listChannels('anw-games','anw-games');
  const report = {channels:channels.map(c=>({name:c.name,retainedReleaseCount:c.retainedReleaseCount,
    version:c.release?.version?.name,updated:c.updateTime})),
    versions:versions.map(v=>({name:v.name,status:v.status,createTime:v.createTime,finalizeTime:v.finalizeTime,
      fileCount:v.fileCount,versionBytes:v.versionBytes}))};
  fs.writeFileSync('perf_results/hosting_inventory.json',JSON.stringify(report,null,2));
  console.log(JSON.stringify({channels:report.channels,count:report.versions.length,
    totalBytes:report.versions.reduce((n,v)=>n+Number(v.versionBytes??0),0),
    recent:report.versions.slice(0,6),oldest:report.versions.slice(-2)},null,2));
})().catch(error=>{console.error(error.message);process.exitCode=1;});
