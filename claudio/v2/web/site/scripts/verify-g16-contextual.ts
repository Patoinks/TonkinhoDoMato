import { readFileSync, writeFileSync } from 'node:fs';
import { pathToFileURL } from 'node:url';
import assert from 'node:assert/strict';
import { Chess, type Square } from 'chess.js';
import { Network } from '../src/tonkai/network';

const manifest=JSON.parse(readFileSync('public/tonkai/g16-v2.json','utf8'));
assert.equal(manifest.piece_value_mode,'contextual');assert.equal(manifest.claudio_config.phase_values,'');
const {default:create}=await import(pathToFileURL(process.cwd()+'/public/tonkai/g16-rules.mjs').href);
const rules=await create({wasmBinary:readFileSync('public/tonkai/g16-rules.wasm')});
const bytes=readFileSync('public/tonkai/'+manifest.binary);
const net=new Network(manifest,bytes.buffer.slice(bytes.byteOffset,bytes.byteOffset+bytes.byteLength),rules);
const ref=JSON.parse(readFileSync(process.argv[2],'utf8'));
assert.equal(manifest.config.kd_weight,ref.weight);
let maxNeuralError=0,maxMaterialError=0,searchCases=0;
const code=(uci:string)=>{
 const square=(s:string)=>(Number(s[1])-1)*8+s.charCodeAt(0)-97;
 return square(uci.slice(0,2))|(square(uci.slice(2,4))<<6)|(uci[4]?' pnbrqk'.indexOf(uci[4])<<12:0);
};
for(const frame of ref.frames){
 const board=new Chess(frame.root);for(const uci of frame.moves??[])board.move({from:uci.slice(0,2),to:uci.slice(2,4),promotion:uci[4]});
 const self=frame.side?'w':'b',value=net.temporal!.pieceValuer(board,self);
 for(let sq=0;sq<64;sq++){
  const square=(String.fromCharCode(97+(sq&7))+(1+(sq>>3))) as Square,piece=board.get(square);
  if(piece&&piece.type!=='k')maxNeuralError=Math.max(maxNeuralError,Math.abs((piece.color===self?1:-1)*value(piece,square)-frame.values[sq]));
 }
 if(board.turn()===self){
  net.rank(board,self);const material=rules.call('material',[]) as number[];
  for(let sq=0;sq<64;sq++)maxNeuralError=Math.max(maxNeuralError,Math.abs(material[sq]-frame.values[sq]));
 }
}
for(const test of ref.material){
 rules.call('init',[test.fen,[]]);rules.call('context',[test.values]);
 const codes=test.rows.map((row:{uci:string})=>code(row.uci));
 const fallback=[0,...Object.values(manifest.kd_values)];
 const losses=rules.call('exchange',[codes,fallback,2,2000000]) as number[];
 const deep=rules.call('deep',[codes,fallback,2,1,2000000]) as [number,number[]][];
 test.rows.forEach((row:{exchange:number;deep:number},i:number)=>{
  assert.ok(Number.isFinite(losses[i])&&Number.isFinite(deep[i][0]));
  maxMaterialError=Math.max(maxMaterialError,Math.abs(losses[i]-row.exchange),Math.abs(deep[i][0]-row.deep));searchCases++;
 });
}
// A moved piece carries its individual value; castling carries the rook too.
rules.call('init',['r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 20 21',[]]);
const material=Array.from({length:128},(_,i)=>.1+i*.071);rules.call('context',[material]);
rules.call('push',[code('e1g1')]);assert.equal((rules.call('material',[])as number[])[5],material[7]);
rules.call('pop',[]);assert.equal((rules.call('material',[])as number[])[7],material[7]);
// The fullmove number alone no longer changes the piece's material price.
const fen='4k3/8/8/8/8/2N2N2/PPPPPPPP/4K3 w - - 0 ',early=new Chess(fen+'1'),later=new Chess(fen+'30');
for(const sq of ['c3','f3']as Square[])assert.equal(net.temporal!.pieceValuer(early,'w')(early.get(sq)!,sq),net.temporal!.pieceValuer(later,'w')(later.get(sq)!,sq));
const first=net.temporal!.pieceValuer(early,'w'),knights={c3:first(early.get('c3')!,'c3'),f3:first(early.get('f3')!,'f3')};
assert.ok(Math.abs(knights.c3-knights.f3)>1e-6,'Different knight squares must retain contextual distinctions');
assert.ok(maxNeuralError<1e-3,`Neural reference mismatch ${maxNeuralError}`);
assert.ok(maxMaterialError<1e-7,`Material oracle mismatch ${maxMaterialError}`);
const report={mode:manifest.piece_value_mode,measuredWeight:ref.weight,networkWeight:1-ref.weight,frames:ref.frames.length,searchCases,maxNeuralError,maxMaterialError,knights,phaseJumpRemoved:true,castlingIdentityPreserved:true,weightsUnchanged:manifest.sha256};
writeFileSync(process.argv[3],JSON.stringify(report,null,2));console.log(JSON.stringify(report));
