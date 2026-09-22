"""Read-only library investigation. Writes only investigation_codex_* artifacts."""
import os, json, time, statistics, platform, copy, contextlib
from collections import defaultdict
import numpy as np
import torch
import chess
import tonkai
import tonkai.agent as am
from tonkai.config import Config
from tonkai.encoding import INPUT_SIZE, unpack_many

OUT = {}
def emit(k,v):
    OUT[k]=v
    print(k, json.dumps(v), flush=True)
    with open('investigation_codex_results.json','w') as f: json.dump(OUT,f,indent=2)
def sync(): torch.cuda.synchronize()
def bench(fn,n=30,reps=5):
    for _ in range(3): fn()
    sync()
    out=[]
    for _ in range(reps):
        t=time.perf_counter()
        for _ in range(n): fn()
        sync(); out.append((time.perf_counter()-t)*1000/n)
    return dict(median_ms=statistics.median(out),min_ms=min(out),max_ms=max(out),blocks_ms=out)

def main():
    emit('system',dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,logical_cpus=os.cpu_count(),threads=torch.get_num_threads(),interop=torch.get_num_interop_threads(),gpu=torch.cuda.get_device_name(),env={k:os.environ.get(k) for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']}))
    cfg=Config(device='cuda',temperature=.1)
    net,agent,opponent,worker,tr,metrics=tonkai.build(cfg,'cuda')
    # Read weights only; never save a checkpoint or change checkpoint state.
    checkpoint=torch.load('checkpoints/latest.pt',map_location='cpu',weights_only=False)
    net.load_state_dict(checkpoint['model'])
    emit('checkpoint',dict(games=checkpoint.get('games'),steps=checkpoint.get('steps'),config=checkpoint.get('config')))
    del checkpoint
    captured=[]
    original_encode=am.encode_after_moves
    def capture(board,moves,color,out=None):
        if len(captured)<400: captured.append((board.copy(),list(moves),color))
        return original_encode(board,moves,color,out)
    am.encode_after_moves=capture
    for _ in range(32): tr.add_game(worker.play_game())
    am.encode_after_moves=original_encode
    emit('replay',dict(value=len(tr.replay),predict=len(tr.predict_replay),mean_candidates=float(tr.predict_replay.count[:len(tr.predict_replay)].mean())))
    # Fixed sample corpus, live worker wrappers only for timings, no source edits.
    stats=defaultdict(lambda:[0,0])
    def wrap(obj,name,label):
        orig=getattr(obj,name)
        def call(*a,**kw):
            t=time.perf_counter(); v=orig(*a,**kw)
            stats[label][0]+=time.perf_counter()-t; stats[label][1]+=1
            return v
        setattr(obj,name,call)
        return lambda:setattr(obj,name,orig)
    restores=[wrap(am,'encode_after_moves','encode_candidates'),wrap(am,'pack_into','pack_row'),wrap(worker,'_sacrifice_check','sacrifice'),wrap(am.env,'terminal','terminal'),wrap(tr,'add_game','ingest')]
    before=metrics.to_dict(); t=time.perf_counter()
    for _ in range(24): tr.add_game(worker.play_game())
    dt=time.perf_counter()-t
    after=metrics.to_dict(); plies=after['plies']-before['plies']
    emit('selfplay',dict(wall_s=dt,games=24,games_s=24/dt,plies=plies,ms_ply=dt*1000/plies,candidates=after['tonkai_positions']-before['tonkai_positions'],timer_s={k:after['times'].get(k,0)-before['times'].get(k,0) for k in after['times']},sub_s_calls=dict(stats)))
    for restore in restores:restore()
    emit('train_full',bench(tr.train_step))
    emit('value_sample',bench(lambda:tr.replay.sample(512),n=100))
    emit('predict_sample',bench(lambda:tr.predict_replay.sample(64),n=100))
    x,y=tr.replay.sample(512)
    px,mask,truth=tr.predict_replay.sample(64)
    emit('bytes',dict(value_float=x.nbytes,predict_float=px.nbytes,predict_valid_float=px.reshape(-1,782)[mask.reshape(-1)].nbytes,valid_rows=int(mask.sum()),packed_predict=64*48*99,unpack_bits=64*48*784))
    rng=np.random.default_rng(123)
    emit('rng_probe',bench(lambda:rng.random((512,768)).astype(np.float32),n=100))
    emit('predict_mask_copy',bench(lambda:px.reshape(-1,782)[mask.reshape(-1)],n=100))
    valid=px.reshape(-1,782)[mask.reshape(-1)]
    emit('h2d_valid_predict',bench(lambda:torch.from_numpy(valid).to('cuda'),n=100))
    emit('h2d_value',bench(lambda:torch.from_numpy(x).to('cuda'),n=100))
    # Train phases: synchronize both boundaries, so totals are attribution experiments,
    # not the baseline throughput. Nested scopes report exclusive time.
    stats.clear(); stack=[]; restores=[]
    def stage(obj,name,label):
        orig=getattr(obj,name)
        def call(*a,**kw):
            sync(); start=time.perf_counter(); stack.append(0)
            ret=orig(*a,**kw); sync(); elapsed=time.perf_counter()-start
            children=stack.pop(); stats[label][0]+=elapsed-children; stats[label][1]+=1
            if stack:stack[-1]+=elapsed
            return ret
        setattr(obj,name,call);restores.append(lambda:setattr(obj,name,orig))
    for obj,name,label in [(tr.replay,'sample','value_sample'),(tr.predict_replay,'sample','predict_sample'),(tr,'_piece_loss','piece_exclusive'),(tr,'_value_and_piece_loss','value_exclusive'),(tr,'_predict_loss','predict_exclusive'),(torch.Tensor,'backward','backward'),(torch.nn.utils,'clip_grad_norm_','clip'),(tr.opt,'step','optimizer'),(tr,'train_step','train_remaining')]:stage(obj,name,label)
    for _ in range(30):tr.train_step()
    emit('train_sync_phases_ms',{k:dict(ms_step=v[0]*1000/30,calls=v[1]) for k,v in stats.items()})
    for restore in reversed(restores):restore()
    # CPU/GPU same fixed boards and same weights, including required CPU result.
    cpu=copy.deepcopy(net).cpu().eval()
    board,moves,color=captured[30]; xb=original_encode(board,moves,color)
    emit('inference_candidates',len(moves))
    def infer(model,device):
        with torch.inference_mode():return torch.softmax(model.value_logits(model.encode(torch.from_numpy(xb).to(device))),-1).cpu().numpy()
    for threads in [1,2,4,6,1]:
        torch.set_num_threads(threads)
        emit('cpu_infer_threads_'+str(threads),bench(lambda:infer(cpu,'cpu'),n=200))
    emit('cuda_infer',bench(lambda:infer(net,'cuda'),n=200))
    xt=torch.from_numpy(xb).to('cuda')
    def resident():
        with torch.inference_mode():return torch.softmax(net.value_logits(net.encode(xt)),-1)
    emit('cuda_resident_fwd',bench(resident,n=200))
    for threads in [1,2,6,1]:
        torch.set_num_threads(threads)
        emit('train_threads_'+str(threads),bench(tr.train_step,n=20,reps=3))
    # Same deterministic selfplay seed per device, no learning.
    from tonkai.agent import TonkAIAgent,GameWorker
    from tonkai.metrics import Metrics
    cpu.load_state_dict(net.state_dict())
    for device in ['cuda','cpu','cuda','cpu']:
        model=net if device=='cuda' else cpu
        m=Metrics(); a=TonkAIAgent(model,cfg,device)
        w=GameWorker(a,tonkai.make_opponent(cfg),cfg,m,np.random.default_rng(867))
        start=time.perf_counter()
        for _ in range(20):w.play_game()
        elapsed=time.perf_counter()-start
        key='full_selfplay_'+device
        item=dict(games_s=20/elapsed,ms_ply=elapsed*1000/m.plies,plies=m.plies,timers=dict(m.times))
        emit(key,OUT.get(key,[])+[item])
    # Ready replay; explicitly verify updates actually execute.
    start=time.perf_counter(); b=tr.steps; p=metrics.plies; timers=dict(metrics.times)
    for _ in range(20):tr.add_game(worker.play_game());tr.train(4)
    elapsed=time.perf_counter()-start
    emit('combined',dict(games_s=20/elapsed,wall_s=elapsed,steps=tr.steps-b,plies=metrics.plies-p,timers_s={k:v-timers.get(k,0) for k,v in metrics.times.items()}))
    # Profiler is diagnostic only; heavily instrumented timings excluded above.
    try:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
            for _ in range(5):tr.train_step()
        with open('investigation_codex_profiler.txt','w') as f:
            f.write(prof.key_averages().table(sort_by='self_cpu_time_total',row_limit=40))
            f.write('\n\n'+prof.key_averages().table(sort_by='self_cuda_time_total',row_limit=40))
        prof.export_chrome_trace('investigation_codex_trace.json')
        emit('profiler','saved')
    except Exception as exc:emit('profiler_error',str(exc))
if __name__=='__main__':main()
