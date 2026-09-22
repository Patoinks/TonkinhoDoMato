import copy,json,time,statistics
from pathlib import Path
import numpy as np, torch, tonkai
from tonkai.config import Config
from tonkai.agent import TonkAIAgent,GameWorker
from tonkai.metrics import Metrics
from tonkai.encoding import unpack_many,pack_into,PACKED_SIZE

def main():
    cfg=Config(device='cuda',inference_device='learner',temperature=.1)
    net,_,_,worker,tr,m=tonkai.build(cfg)
    net.load_state_dict(torch.load('checkpoints/deployed_5020games.pt',map_location='cpu',weights_only=False)['model'])
    result={}
    for device in ['cuda','cpu','cuda','cpu']:
        model=copy.deepcopy(net).to(device)
        mm=Metrics(); w=GameWorker(TonkAIAgent(model,cfg,device),tonkai.make_opponent(cfg),cfg,mm,np.random.default_rng(867))
        start=time.perf_counter()
        for _ in range(40): w.play_game()
        dt=time.perf_counter()-start
        result.setdefault(device,[]).append({'games_s':40/dt,'positions_s':mm.tonkai_positions/dt,'plies':mm.plies,'times':dict(mm.times)})
    Path('perf_results/frozen_devices.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
if __name__=='__main__':main()
