"""Isolate scratch reuse and measure steady-state weight refresh overhead."""
import json,time,statistics
from pathlib import Path
import numpy as np,torch,tonkai
from tonkai.config import Config
from tonkai.metrics import Metrics
from tonkai.parallel import ActorPool

def main():
    results={'probe':{},'refresh':{}}
    for b in [512,2048]:
        flat=np.ones((b,768),dtype=np.float32)
        r64=np.empty_like(flat,dtype=np.float64);s32=np.empty_like(flat)
        rng=np.random.default_rng(42)
        def old():return (rng.random((b,768)).astype(np.float32)*flat).argmax(axis=1)
        def new():
            rng.random(out=r64);np.copyto(s32,r64);np.multiply(s32,flat,out=s32)
            return s32.argmax(axis=1)
        row={}
        for label,fn in [('old',old),('scratch',new)]:
            blocks=[]
            for _ in range(3):
                start=time.perf_counter()
                for _ in range(100):fn()
                blocks.append((time.perf_counter()-start)*10)
            row[label+'_ms']=statistics.median(blocks)
        results['probe'][b]=row
    cfg=Config(device='cuda',inference_device='learner',temperature=.1)
    net,_,_,w,tr,m=tonkai.build(cfg)
    state=torch.load('checkpoints/deployed_5020games.pt',map_location='cpu',weights_only=False)['model']
    net.load_state_dict(state)
    corpus=[w.play_game() for _ in range(70)]
    for interval in [1,20,100]:
        cfg.num_workers=2;cfg.weight_refresh_games=interval
        net,_,_,_,tr,m=tonkai.build(cfg);net.load_state_dict(state)
        for r in corpus:tr.add_game(r)
        start=time.perf_counter()
        with ActorPool(cfg,net,m) as pool:
            for _ in range(20):
                tr.add_game(pool.next_game());tr.train(1);pool.refresh(net,tr.games)
            startup=time.perf_counter()-start
            before=dict(m.times); pos=m.tonkai_positions;version=pool.version.value
            start=time.perf_counter()
            for _ in range(180):
                tr.add_game(pool.next_game());tr.train(1);pool.refresh(net,tr.games)
            dt=time.perf_counter()-start
            delta=Metrics();delta.times.update({k:v-before.get(k,0) for k,v in m.times.items()})
            results['refresh'][interval]={'games_s':180/dt,'positions_s':(m.tonkai_positions-pos)/dt,
                'wall_s':dt,'warmup_and_start_s':startup,'publications':pool.version.value-version,
                'times':dict(delta.times),'timing_breakdown':delta.timing_breakdown()}
        Path('perf_results/details.json').write_text(json.dumps(results,indent=2))
    print(json.dumps(results,indent=2))
if __name__=='__main__':main()
