import os,time,json,copy,statistics,concurrent.futures,multiprocessing
import numpy as np
import torch,chess,tonkai
from tonkai.config import Config
from tonkai.encoding import *
from tonkai.agent import TonkAIAgent,GameWorker
from tonkai.model import TonkAINet
from tonkai.metrics import Metrics

results={}
def emit(k,v):
    results[k]=v;print(k,json.dumps(v),flush=True)
    with open('investigation_codex_followup.json','w') as f:json.dump(results,f,indent=2)
def bench(fn,n=100):
    for _ in range(3):fn()
    ts=[]
    for _ in range(5):
        t=time.perf_counter()
        for _ in range(n):fn()
        ts.append((time.perf_counter()-t)*1000/n)
    return dict(ms=statistics.median(ts),range=[min(ts),max(ts)])
def process_worker(index,barrier):
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    cfg=Config(device='cpu',temperature=.1,seed=42+index)
    net=TonkAINet(cfg)
    net.load_state_dict(torch.load('checkpoints/latest.pt',map_location='cpu',weights_only=False)['model'])
    m=Metrics();w=GameWorker(TonkAIAgent(net,cfg,'cpu'),tonkai.make_opponent(cfg),cfg,m,np.random.default_rng(cfg.seed))
    w.play_game();p=m.plies
    barrier.wait();t=time.perf_counter()
    for _ in range(30):w.play_game()
    dt=time.perf_counter()-t
    return dict(seconds=dt,plies=m.plies-p,games=30)
def main():
    torch.set_num_threads(1)
    cfg=Config(device='cuda',temperature=.1)
    net,a,o,w,tr,m=tonkai.build(cfg,'cuda')
    net.load_state_dict(torch.load('checkpoints/latest.pt',map_location='cpu',weights_only=False)['model'])
    for _ in range(40):tr.add_game(w.play_game())
    idx=np.random.default_rng(77).integers(0,len(tr.predict_replay),64)
    packed=tr.predict_replay.cand[idx];mask=np.arange(48)[None,:]<tr.predict_replay.count[idx,None]
    def original():return unpack_many(packed.reshape(-1,99)).reshape(-1,782)[mask.reshape(-1)]
    def valid_only():return unpack_many(packed[mask])
    assert np.array_equal(original(),valid_only())
    emit('valid_first',dict(original=bench(original),valid_first=bench(valid_only),valid=int(mask.sum()),rows=3072,exact=True))
    x,y=tr.replay.sample(512);flat=x[:,:768].reshape(512,12,64).copy();flat[:,[5,11],:]=0;flat=flat.reshape(512,768)
    r1=np.random.default_rng(33);r2=np.random.default_rng(33)
    f64=np.empty((512,768),dtype=np.float64);f32=np.empty((512,768),dtype=np.float32)
    def orig_rng():return (r1.random((512,768)).astype(np.float32)*flat).argmax(1)
    def reuse_rng():
        r2.random(out=f64);np.copyto(f32,f64);np.multiply(f32,flat,out=f32)
        return f32.argmax(1)
    assert np.array_equal(orig_rng(),reuse_rng())
    emit('rng_reuse',dict(original=bench(orig_rng),reuse=bench(reuse_rng),exact_first_draw=True))
    board=chess.Board()
    for san in ['e4','e5','Nf3','Nc6','Bc4','Bc5','d3','d6']:board.push_san(san)
    moves=list(board.legal_moves);enc=encode_after_moves(board,moves,chess.BLACK)
    dst=np.empty((len(moves),99),dtype=np.uint8)
    def loop_pack():
        for i in range(len(moves)):pack_into(enc[i],dst[i])
        return dst.copy()
    def batch_pack():
        dst[:,:98]=np.packbits(enc[:,:781].astype(np.uint8),axis=1,bitorder='little')
        dst[:,98]=np.rint(enc[:,781].astype(np.float64)*100).astype(np.uint8)
        return dst.copy()
    assert np.array_equal(loop_pack(),batch_pack())
    emit('batch_pack',dict(loop=bench(loop_pack),batch=bench(batch_pack),moves=len(moves),exact=True))
    # Exact model and seed for CPU/CUDA comparison, frozen during all trials.
    cpu=copy.deepcopy(net).cpu()
    for device in ['cpu','cuda','cpu','cuda']:
        model=cpu if device=='cpu' else net
        mm=Metrics();ww=GameWorker(TonkAIAgent(model,cfg,device),tonkai.make_opponent(cfg),cfg,mm,np.random.default_rng(123))
        t=time.perf_counter()
        for _ in range(30):ww.play_game()
        dt=time.perf_counter()-t
        key='matched_'+device
        emit(key,results.get(key,[])+[dict(games_s=30/dt,ms_ply=1000*dt/mm.plies,plies=mm.plies,positions=mm.tonkai_positions,timers=dict(mm.times))])
    # CUDA events give stream elapsed, including host submission gaps, not kernel busy time.
    xt=torch.from_numpy(enc).cuda();beg=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
    with torch.inference_mode():
        for _ in range(20):net.value_logits(net.encode(xt))
        torch.cuda.synchronize();beg.record()
        for _ in range(200):net.value_logits(net.encode(xt))
        end.record();end.synchronize()
    emit('gpu_event_stream_ms_fwd',beg.elapsed_time(end)/200)
    del tr,w,a,net,cpu
    # Each trial excludes spawn/import/model-load/warm-up using a process barrier.
    for count in [1,2,4,6,2]:
        ctx=multiprocessing.get_context('spawn')
        with ctx.Manager() as manager:
            barrier=manager.Barrier(count)
            with concurrent.futures.ProcessPoolExecutor(count,mp_context=ctx) as pool:
                rows=list(pool.map(process_worker,range(count),[barrier]*count))
        elapsed=max(r['seconds'] for r in rows)
        key='workers_'+str(count)
        emit(key,results.get(key,[])+[dict(games_s=30*count/elapsed,plies_s=sum(r['plies'] for r in rows)/elapsed,seconds=elapsed,workers=rows)])
if __name__=='__main__':main()
