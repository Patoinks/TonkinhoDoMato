import { spawn } from 'node:child_process';
import { chromium } from '@playwright/test';
import { readFileSync, writeFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import assert from 'node:assert/strict';
import { Chess } from 'chess.js';

const out=process.env.G16_WEB_OUTPUT ?? 'test-results';
const base=process.argv[2] ?? 'http://127.0.0.1:4176';
const expected=JSON.parse(readFileSync('public/tonkai/g16-v2.json','utf8'));
const server=process.argv[2] ? null : spawn(process.execPath,['node_modules/vite/bin/vite.js','preview','--host','127.0.0.1','--port','4176','--strictPort'],{stdio:'pipe',windowsHide:true});
let browser;
const evidence={url:base,sha256:expected.sha256,source:expected.source_sha256,revision:expected.revision,pieceValueMode:expected.piece_value_mode??'phase',errors:[],games:[],defaultHidden:false,mobileOverflow:null};
try {
  for(let i=0;i<60;i++){try{if((await fetch(base+'/tonkai')).ok)break;}catch{} await new Promise(r=>setTimeout(r,500));}
  const live=await (await fetch(base+'/tonkai/g16-v2.json',{cache:'no-store'})).json();
  assert.equal(live.revision,expected.revision); assert.equal(live.source_games,600);
  for(const [file,sha] of [[live.binary,live.sha256],['g16-rules.wasm',live.native.wasm_sha256],['g16-rules.mjs',live.native.loader_sha256]]) {
    const bytes=Buffer.from(await (await fetch(base+'/tonkai/'+file)).arrayBuffer());
    assert.equal(createHash('sha256').update(bytes).digest('hex'),sha);
  }
  browser=await chromium.launch({channel:'chrome',headless:true});
  for(const human of ['w','b']) {
    const page=await browser.newPage({viewport:{width:1440,height:1000}});
    page.on('pageerror',e=>evidence.errors.push(e.message));
    // Smoke games never enter the public game database.
    await page.route('**/firestore.googleapis.com/**',route=>route.abort());
    await page.goto(base+'/tonkai');
    await page.getByRole('heading',{name:'G16 v2',exact:true}).waitFor();
    await page.getByRole('button',{name:'Começar',exact:true}).waitFor({timeout:60000});
    await page.getByLabel('A tua cor').selectOption(human);
    await page.getByRole('button',{name:'Começar',exact:true}).click();
    await page.getByRole('status',{name:'Estado do jogo'}).filter({hasText:'A tua vez'}).waitFor({timeout:60000});
    evidence.defaultHidden=await page.locator('[data-value-square]').count()===0;
    assert.equal(evidence.defaultHidden,true);
    // Exercise four human/bot turns through the public board's normal controls.
    for(let i=0;i<4;i++) {
      const fen=await page.locator('.tk-board').getAttribute('data-fen'),board=new Chess(fen);
      assert.equal(board.turn(),human);
      const moves=board.moves({verbose:true});
      const move=moves.find(m=>m.piece==='p' && m.from[0]==='e' && m.to[1]===(human==='w'?'4':'5')) ?? moves[0];
      await page.locator(`.tk-board [data-square="${move.from}"]`).click();
      await page.locator(`.tk-board [data-square="${move.to}"]`).click();
      if(move.promotion) await page.getByRole('button',{name:/Dama/}).click();
      await page.waitForFunction(previous=>document.querySelector('.tk-board')?.getAttribute('data-fen')!==previous,fen);
      await page.getByRole('status',{name:'Estado do jogo'}).filter({hasText:'A tua vez'}).waitFor({timeout:60000});
    }
    const fen=await page.locator('.tk-board').getAttribute('data-fen');
    const moves=Number(fen.split(' ')[5])*2-2+Number(fen.split(' ')[1]==='b'); assert.ok(moves>=8);
    await page.getByRole('button',{name:'Mostrar avaliações'}).click();
    await page.locator('[data-value-square]').first().waitFor();
    assert.equal(await page.locator('[data-engine="TK"]').getAttribute('data-fen'),fen);
    const clocks=await page.locator('.tk-clock').allTextContents();
    await page.getByRole('button',{name:'Desistir',exact:true}).click();
    await page.getByRole('status',{name:'Estado do jogo'}).filter({hasText:'G16 v2 venceu'}).waitFor();
    const displayedValues=await page.locator('[data-value-square]').evaluateAll(nodes=>nodes.map(n=>({square:n.getAttribute('data-value-square'),value:n.getAttribute('data-value')})));
    const shownBoard=new Chess(fen),knights=displayedValues.filter(row=>shownBoard.get(row.square)?.type==='n');
    if(live.piece_value_mode==='contextual'){
      assert.equal(live.claudio_config.phase_values,'');
      assert.ok(knights.length>0 && knights.some(row=>Math.abs(Math.abs(Number(row.value))-1.5246)>1e-4));
    }
    evidence.games.push({human,plies:moves,ended:'resignation',clocks,fen,knights});
    await page.getByRole('button',{name:'Ver tabuleiro',exact:true}).click();
    if(human==='w') {
      await page.screenshot({path:out+'/g16-v2-desktop.png',fullPage:true});
      await page.setViewportSize({width:390,height:844});
      evidence.mobileOverflow=await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth);
      assert.equal(evidence.mobileOverflow,false);
      await page.screenshot({path:out+'/g16-v2-mobile.png',fullPage:true});
    }
    await page.close();
  }
  const versions=await browser.newPage();await versions.goto(base+'/tonkai/versoes');
  await versions.getByRole('heading',{name:'G16 v2',exact:true}).waitFor();
  assert.deepEqual(evidence.errors,[]);
  writeFileSync(out+'/g16-v2-browser-smoke.json',JSON.stringify(evidence,null,2));
  console.log(JSON.stringify(evidence));
} finally { await browser?.close(); server?.kill(); }
