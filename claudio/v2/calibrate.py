"""Calibrate policy residuals against held-out own-search choices, not engine evaluations."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import paths
from claudio_load import load_claudio
from v2.agent import OwnPolicy


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--assets',required=True)
    args=parser.parse_args()
    folder=Path(args.assets)
    torch.set_num_threads(1)
    model=load_claudio(folder/'compact.pt')[0]
    policy=OwnPolicy(model.latent_size).eval()
    policy.load_state_dict(torch.load(folder/'policy.pt',weights_only=True),strict=False)
    archive=np.load(folder/'own-dataset.npz')
    arrays={key:archive[key] for key in ('x','memory','order_features','order_base')}
    groups=json.loads((folder/'own-dataset.json').read_text())['groups']
    valid=[g for g in groups if g['game']%5==0 and g['policy_target'] is not None and len(g['top'])>1]
    samples=[]
    with torch.inference_mode():
        for g in valid:
            ids=np.array(g['top'])+g['start']
            context,_=model.candidates(torch.from_numpy(arrays['x'][ids]),torch.from_numpy(arrays['memory'][ids]))
            features=torch.from_numpy(arrays['order_features'][ids]).clone()
            features[:,2]=(features[:,2]/policy.temperature).clamp(-10,0)
            base=torch.from_numpy(arrays['order_base'][ids])
            logits=policy(context,features,base)
            prior=base/policy.temperature
            samples.append((prior,logits-prior,g['policy_target']))
    blends=[0,0.05,0.1,0.2,0.35,0.5,0.75,1]
    rows=[]
    for blend in blends:
        correct=sum(int((base+blend*delta).argmax())==target for base,delta,target in samples)
        rows.append({'blend':blend,'correct':correct,'roots':len(samples),'accuracy':correct/max(1,len(samples))})
    # Smaller correction wins ties. Validation can explicitly keep the original ordering.
    best=max(rows,key=lambda row:(row['correct'],-row['blend']))
    policy.blend.fill_(best['blend'])
    threshold=1.01
    for level in np.arange(0.1,0.91,0.05):
        selected=[]
        for base,delta,target in samples:
            logits=base+best['blend']*delta
            if float(logits.softmax(0).max())>=level:
                selected.append((int(logits.argmax())==target,int(base.argmax())==target))
        if len(selected)>=20 and sum(a for a,_ in selected)/len(selected)>=0.9 and sum(a for a,_ in selected)>=sum(b for _,b in selected):
            threshold=float(level);break
    torch.save(policy.state_dict(),folder/'policy.pt')
    report=json.loads((folder/'learning.json').read_text())
    report.update(policy_blend=best['blend'],policy_confidence_threshold=threshold,
                  policy_validation_top1=best['accuracy'],policy_calibration_sweep=rows)
    (folder/'learning.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'best':best,'threshold':threshold,'sweep':rows},indent=2))


if __name__=='__main__':main()
