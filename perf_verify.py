import sys, json, time, hashlib, pickle, subprocess
from pathlib import Path
import numpy as np
import torch, chess, tonkai
from tonkai.config import Config
from tonkai.encoding import encode, encode_after_moves, pack
from export_web import GOLDEN_FENS
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'perf_results'; OUT.mkdir(exist_ok=True)
def digest():
    h=hashlib.sha256()
    for _,fen,color in GOLDEN_FENS:
        b=chess.Board(fen)
        for c in [color,not color]:
            x=encode(b,c); h.update(x.tobytes()); h.update(pack(x).tobytes())
            for x in encode_after_moves(b,list(b.legal_moves),c): h.update(x.tobytes()); h.update(pack(x).tobytes())
    return h.hexdigest()
def run(label):
    cfg=Config(device='cuda',temperature=.1)
    net,a,o,w,tr,m=tonkai.build(cfg)
    state=torch.load('checkpoints/deployed_5020games.pt',map_location='cpu',weights_only=False)['model']; net.load_state_dict(state); tr.sync_inference()
    rec=w.play_game(); game=hashlib.sha256(rec.packed.tobytes()+b''.join(p.tobytes() for p,i in rec.predict_samples)).hexdigest()
    for _ in range(64): tr.add_game(w.play_game())
    while not tr.ready(): tr.add_game(w.play_game())
    result={'encoding':digest(),'game':game,'workers':{},'batch_size':cfg.batch_size,'updates':cfg.updates_per_game}
    for _ in range(3): tr.train(cfg.updates_per_game)
    start=time.perf_counter()
    for _ in range(30): tr.train(cfg.updates_per_game)
    result['learner_ms_game']=(time.perf_counter()-start)*1000/30
    for workers in ([1,2,4] if hasattr(cfg,'weight_refresh_games') else [1]):
        cfg.num_workers=workers
        if workers > 1: tr.inference_net = net
        times=dict(m.times); pos=m.tonkai_positions
        start=time.perf_counter()
        if workers==1:
            for _ in range(60): tr.add_game(w.play_game()); tr.train(cfg.updates_per_game)
        else:
            from tonkai.parallel import ActorPool
            with ActorPool(cfg,net,m) as pool:
                for _ in range(60):
                    tr.add_game(pool.next_game()); tr.train(cfg.updates_per_game); pool.refresh(net,tr.games)
        dt=time.perf_counter()-start
        from tonkai.metrics import Metrics
        delta=Metrics(); delta.times.update({k:v-times.get(k,0) for k,v in m.times.items()})
        result['workers'][workers]={'games_s':60/dt,'positions_s':(m.tonkai_positions-pos)/dt,'wall_s':dt,'times':{k:v-times.get(k,0) for k,v in m.times.items()},'timing_breakdown':delta.timing_breakdown(),'actor_times':dict(getattr(m,'actor_times',{}))}
    subprocess.run([sys.executable,'export_web.py','--checkpoint','checkpoints/deployed_5020games.pt','--out',str(OUT/'export')],check=True,stdout=subprocess.DEVNULL)
    assert json.loads((OUT/'export/golden.json').read_text())==json.loads(Path('web/golden.json').read_text())
    if label!='baseline':
        baseline=json.loads((OUT/'baseline.json').read_text()); assert result['encoding']==baseline['encoding']; assert result['game']==baseline['game']
    (OUT/(label+'.json')).write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
if __name__=='__main__': run(sys.argv[1])

