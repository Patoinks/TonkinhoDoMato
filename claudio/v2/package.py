"""Portable weights, learned policy, own book and G16 v2 module settings."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import torch


def save_bundle(source, folder, target, variant='full'):
    from .agent import V2Options
    folder=Path(folder)
    data={'format':'claudio-v2','name':'G16 v2 checkpoint600','source_games':600,
          'source_sha256':hashlib.sha256(Path(source).read_bytes()).hexdigest(),
          'teacher':torch.load(source,map_location='cpu',weights_only=False),
          'compact':torch.load(folder/'compact.pt',map_location='cpu',weights_only=False),
          'policy':torch.load(folder/'policy.pt',map_location='cpu',weights_only=True),
          'book':json.loads((folder/'opening-book.json').read_text(encoding='utf-8')),
          'options':asdict(V2Options()),'default_variant':variant,'draw_points':0.5,
          'learning':json.loads((folder/'learning.json').read_text(encoding='utf-8'))}
    torch.save(data,target)
    return data


def load_bundle(target, variant=None, rng=None):
    from .agent import V2Claudio,V2Options,OwnPolicy
    from .rules import draw_config
    from claudio_load import build_net
    from claudio_agent import ClaudioConfig
    data=torch.load(target,map_location='cpu',weights_only=False)
    if data.get('format')!='claudio-v2':raise ValueError('Not a G16 v2 bundle')
    variant=variant or data['default_variant']
    chosen=data['compact'] if variant in ('full','all') else data['teacher']
    net=build_net(chosen['architecture'])
    net.load_state_dict(chosen['net'])
    config=draw_config(dict(data['teacher']['config'],book='',book_weight=0.03,book_min=4,endgame_mate=6))
    king=None
    if data['teacher'].get('king'):
        from tonkai.king_safety_model import KingSafetyNet,KingSafetyModel
        king_net=KingSafetyNet(data['teacher']['king']['hidden'])
        king_net.load_state_dict(data['teacher']['king']['state_dict'])
        king=KingSafetyModel(king_net)
    policy=OwnPolicy(net.latent_size)
    policy.load_state_dict(data['policy'])
    options=V2Options(**data['options'])
    if variant=='exact':
        options.learned_policy=options.opening_book=options.opening_fast=options.finish_planner=False
    elif variant=='finish':
        options.learned_policy=options.opening_book=options.opening_fast=False
    player=V2Claudio(net,data['teacher']['values'],ClaudioConfig.with_settings([],base=config),
                     rng=rng,king_model=king,options=options,policy=policy)
    player.policy_threshold=data['learning'].get('policy_confidence_threshold',1.01)
    if variant=='all':
        player.policy_threshold=0.0
        player.policy.blend.fill_(1.0)
    if options.opening_book:
        player.book_moves={key:{uci:(int(n),int(good)) for uci,(n,good) in rows.items()}
                           for key,rows in data['book']['positions'].items()}
    return player
