"""Same seeded replay corpus for before/after microbenchmarks."""
import sys,time,json,statistics
from pathlib import Path
import numpy as np,torch,tonkai
from tonkai.config import Config
from tonkai.encoding import unpack_many,PACKED_SIZE,pack_into

def main(label):
    cfg=Config(device='cuda',inference_device='learner',temperature=.1)
    net,_,_,w,tr,m=tonkai.build(cfg)
    net.load_state_dict(torch.load('checkpoints/deployed_5020games.pt',map_location='cpu',weights_only=False)['model'])
    for _ in range(70):tr.add_game(w.play_game())
    def bench(fn,n=30):
        for _ in range(3):fn()
        out=[]
        for _ in range(3):
            start=time.perf_counter()
            for _ in range(n):fn()
            torch.cuda.synchronize();out.append(1000*(time.perf_counter()-start)/n)
        return {'median_ms':statistics.median(out),'blocks_ms':out}
    results={}
    results['train_2048']=bench(lambda:tr.train_step(2048))
    results['predict_sample_256']=bench(lambda:tr.predict_replay.sample_valid(256) if hasattr(tr.predict_replay,'sample_valid') else tr.predict_replay.sample(256))
    x=unpack_many(tr.predict_replay.cand[0,:tr.predict_replay.count[0]])
    def packing():
        import tonkai.encoding as E
        if hasattr(E,'pack_many'):return E.pack_many(x)
        out=np.empty((len(x),PACKED_SIZE),dtype=np.uint8)
        for i in range(len(x)):pack_into(x[i],out[i])
        return out
    results['packing']=bench(packing,n=500)
    results['candidates']=len(x)
    Path('perf_results/'+label+'.json').write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2))
if __name__=='__main__':main(sys.argv[1])
