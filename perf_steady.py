"""Longer warmed single/four-worker runs; two-worker run is in details.json."""
import json,time
from pathlib import Path
import torch,tonkai
from tonkai.config import Config
from tonkai.metrics import Metrics
from tonkai.parallel import ActorPool

def main():
    cfg=Config(device='cuda',inference_device='learner',temperature=.1)
    net,_,_,w,_,_=tonkai.build(cfg)
    state=torch.load('checkpoints/deployed_5020games.pt',map_location='cpu',weights_only=False)['model']
    net.load_state_dict(state);corpus=[w.play_game() for _ in range(70)]
    result={}
    for workers in [1,4]:
        cfg.num_workers=workers;cfg.inference_device='cpu'
        net,_,_,w,tr,m=tonkai.build(cfg);net.load_state_dict(state);tr.sync_inference()
        for rec in corpus:tr.add_game(rec)
        start=time.perf_counter()
        pool=ActorPool(cfg,net,m) if workers>1 else None
        try:
            for _ in range(20):
                tr.add_game(pool.next_game() if pool else w.play_game());tr.train(1)
                if pool:pool.refresh(net,tr.games)
            startup=time.perf_counter()-start
            before=dict(m.times);pos=m.tonkai_positions;start=time.perf_counter()
            for _ in range(180):
                tr.add_game(pool.next_game() if pool else w.play_game());tr.train(1)
                if pool:pool.refresh(net,tr.games)
            dt=time.perf_counter()-start
            delta=Metrics();delta.times.update({k:v-before.get(k,0) for k,v in m.times.items()})
            result[workers]={'games_s':180/dt,'positions_s':(m.tonkai_positions-pos)/dt,
                'wall_s':dt,'warmup_and_start_s':startup,'times':dict(delta.times),
                'timing_breakdown':delta.timing_breakdown()}
        finally:
            if pool:pool.close()
    Path('perf_results/steady.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
