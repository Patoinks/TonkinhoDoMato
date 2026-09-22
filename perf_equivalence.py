"""Compare optimized training against the original trainer, bit for bit."""
import subprocess,types,json,hashlib
from pathlib import Path
import numpy as np,torch,tonkai
from tonkai.config import Config
source=subprocess.check_output(['git','show','1804bfc:tonkai/trainer.py'],text=True)
reference=types.ModuleType('tonkai._reference_trainer');reference.__package__='tonkai'
exec(compile(source,'1804bfc:tonkai/trainer.py','exec'),reference.__dict__)
results={}
for device in ['cpu','cuda']:
    hashes=[]
    states=[]
    for original in [True,False]:
        cfg=Config(device=device,inference_device='learner',samples_per_game=0,
                   updates_per_game=4,batch_size=128,min_replay=1,max_plies=80)
        net,_,_,w,tr,m=tonkai.build(cfg)
        if original:tr=reference.Trainer(net,cfg,device,m,w.rng)
        h=hashlib.sha256()
        for _ in range(5):
            r=w.play_game();h.update(r.packed.tobytes())
            for p,i in r.predict_samples:h.update(p.tobytes())
            tr.add_game(r);tr.train(4)
        hashes.append(h.hexdigest());states.append({k:v.detach().cpu().clone() for k,v in net.state_dict().items()})
    assert hashes[0]==hashes[1],(device,hashes)
    assert all(torch.equal(states[0][k],states[1][k]) for k in states[0]),device
    results[device]={'five_trained_games_sha256':hashes[0],'parameters_bit_identical':True}
Path('perf_results/training_equivalence.json').write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2))
