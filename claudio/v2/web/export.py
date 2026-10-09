"""Export the existing all-modules v2 bundle; no updates, teacher substitutions or training."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import paths  # frozen Claudio imports
import torch
from claudio_agent import ClaudioConfig


def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    data=torch.load(args.bundle,map_location='cpu',weights_only=False)
    if data['format']!='claudio-v2' or data['default_variant']!='all': raise ValueError('Expected tested all-modules v2')
    tensors=[];blob=bytearray()
    groups={'net':data['compact']['net'],'policy':data['policy'],'king_safety':data['teacher']['king']['state_dict']}
    groups['policy']['blend']=torch.tensor(1.)
    for prefix,weights in groups.items():
        for name,tensor in weights.items():
            tensor=tensor.detach().cpu().float().contiguous()
            if not torch.isfinite(tensor).all(): raise ValueError('Nonfinite weights')
            tensors.append({'name':f'{prefix}.{name}','shape':list(tensor.shape),'offset':len(blob),'count':tensor.numel()})
            blob.extend(tensor.numpy().astype('<f4').tobytes())
    digest=hashlib.sha256(blob).hexdigest();binary=f'tonkai-{digest}.bin'
    cfg=asdict(ClaudioConfig.with_settings([],base=data['teacher']['config']))
    cfg.update(draw_value=.5,draw_ahead=.5,endgame_mate=6,book='',book_weight=.03,book_min=4)
    if (cfg['value_scaling'] or not cfg['memory'] or cfg['reply_depth'] != 2 or cfg['leaf_weight']
            or not cfg['root_pvs'] or cfg['temperature'] or cfg['head'] != 'mean'
            or data['options']['shared_table'] or data['options']['quiet_promotions']):
        raise ValueError('This exporter is pinned to the tested checkpoint-600 all-modules configuration')
    native=json.loads((args.out/'native-provenance.json').read_text())
    manifest={'format':'float32-le','model_kind':'claudio-v2-web','model_name':'G16 v2',
        'bytes':len(blob),'sha256':digest,'binary':binary,'tensors':tensors,'draw_weight':.5,
        'source_sha256':data['source_sha256'],'checkpoint_sha256':hashlib.sha256(args.bundle.read_bytes()).hexdigest(),
        'source_games':600,'default_variant':'all','architecture':data['compact']['architecture'],
        'config':dict(data['compact']['architecture'],objective='victory',king_safety_weight=cfg['king_weight']),
        'claudio_config':cfg,'options':data['options'],'book':data['book']['positions'],
        'kd_values':{str(k):v for k,v in data['teacher']['values'].items()},'king_safety_model':True,'native':native}
    manifest['revision']=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    (args.out/binary).write_bytes(blob)
    (args.out/'g16-v2.json').write_text(json.dumps(manifest),encoding='utf-8')
    (args.out/'g16-v2-version.json').write_text(json.dumps({k:manifest[k] for k in ('sha256','revision','model_kind','model_name','source_sha256')}),encoding='utf-8')
    print(json.dumps({'bytes':len(blob),'sha256':digest,'revision':manifest['revision']}))


if __name__=='__main__':main()
