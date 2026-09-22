const fs = require('node:fs');
const cli = 'C:/Users/Martim/AppData/Roaming/npm/node_modules/firebase-tools/lib/';
const auth = require(cli+'auth.js');
const {requireAuth} = require(cli+'requireAuth.js');
const api = require(cli+'hosting/api.js');
const {Client} = require(cli+'apiv2.js');
const {hostingApiOrigin} = require(cli+'api.js');
(async () => {
  const options = {project:'anw-games',nonInteractive:true};
  const account = auth.getProjectDefaultAccount('C:/Users/Martim/Desktop/Github/ANW-TONKAI-RELEASE-20260922');
  if(account) auth.setActiveAccount(options,account);
  await requireAuth(options);
  const client = new Client({urlPrefix:hostingApiOrigin(),apiVersion:'v1beta1',auth:true});
  const channel = await api.getChannel('anw-games','anw-games','live');
  const live = channel.release.version.name.split('/').pop();
  const versions = await api.listVersions('anw-games');
  const candidates = versions.filter(v=>v.status==='FINALIZED' && v.createTime>='2026-09-22T19:25:00Z' && v.createTime<'2026-09-22T19:32:00Z');
  async function files(id) {
    let result=[], token;
    do {
      const res=await client.get(`/sites/anw-games/versions/${id}/files`,{queryParams:{pageSize:1000,...(token?{pageToken:token}:{})}});
      result.push(...(res.body.files??[])); token=res.body.nextPageToken;
    } while(token);
    return result;
  }
  const liveFiles=await files(live);
  const compare=[];
  for(const v of candidates) {
    const id=v.name.split('/').pop(), list=await files(id);
    const a=new Map(liveFiles.map(f=>[f.path,f.hash])),b=new Map(list.map(f=>[f.path,f.hash]));
    const different=[...new Set([...a.keys(),...b.keys()])].filter(p=>a.get(p)!==b.get(p));
    compare.push({id,createTime:v.createTime,bytes:v.versionBytes,different,files:list});
  }
  fs.writeFileSync('perf_results/hosting_compare.json',JSON.stringify({live,liveFiles,compare},null,2));
  console.log(JSON.stringify({live,liveFiles:liveFiles.length,candidates:compare.map(({files,...v})=>v)},null,2));
})().catch(e=>{console.error(e.message);process.exitCode=1;});
