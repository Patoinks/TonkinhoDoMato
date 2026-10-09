"""Matched history-preserving latency benchmark, ABBA and unchanged checkpoint 600."""
import argparse
import json
from pathlib import Path
import sys
import time
import statistics

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import paths
import chess
import numpy as np
import torch
from claudio_agent import Claudio,ClaudioConfig
from claudio_load import load_claudio
from v2.agent import load_v2
from v2.rules import adjudicate,draw_config


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--assets',required=True)
    parser.add_argument('--positions',type=int,default=96)
    args=parser.parse_args()
    import psutil_free_priority
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    source=load_claudio(args.checkpoint,with_config=True,with_king=True)
    experience=Path(args.assets)/'experience.json'
    if not experience.exists():experience=paths.ROOT/'claudio/results/v2-checkpoint600/experience.json'
    games=json.loads(experience.read_text())['games']
    games=games[::max(1,len(games)//12)][:12]
    eligible=[(g,p) for g,row in enumerate(games) for p in range(len(row['moves'].split()))]
    selected={eligible[round(i*(len(eligible)-1)/(args.positions-1))] for i in range(args.positions)}
    def actor(variant,side):
        rng=np.random.default_rng([9901,int(side)])
        if variant=='baseline':
            net,values,cfg,_,king=source
            return Claudio(net,values,ClaudioConfig.with_settings([],base=draw_config(cfg)),rng=rng,king_model=king)
        return load_v2(args.checkpoint,args.assets,variant=variant,rng=rng)
    def evaluate(variant):
        warm=actor(variant,True)
        if variant=='baseline':warm.choose(chess.Board())
        else:warm.choose(chess.Board(),terminal_checked=True)
        rows=[]
        for game,row in enumerate(games):
            players={side:actor(variant,side) for side in (False,True)}
            board=chess.Board()
            for ply,uci in enumerate(row['moves'].split()):
                player=players[board.turn]
                if (game,ply) in selected and adjudicate(board) is None:
                    began=time.perf_counter()
                    action=player.choose(board) if variant=='baseline' else player.choose(board,terminal_checked=True)
                    seconds=time.perf_counter()-began
                    assert action in board.legal_moves
                    rows.append({'game':game,'ply':ply,'move':action.uci(),'seconds':seconds})
                else:player._catch_up(board)
                board.push_uci(uci)
        return rows
    runs={}
    reference=None
    for variant in ('exact','full','all'):
        results=[]
        for name in ('baseline',variant,variant,'baseline'):
            rows=evaluate(name)
            if reference is None:reference=[(r['game'],r['ply'],r['move']) for r in rows]
            assert len(rows)==len(reference)
            changed=sum(r['move']!=ref[2] for r,ref in zip(rows,reference,strict=True))
            if name in ('baseline','exact') and changed:raise AssertionError(f'{name} changed {changed} reference decisions')
            results.append({'variant':name,'rows':rows,'changed_moves':changed})
            print(f"ABBA {name}: {len(rows)} decisions, {sum(r['seconds'] for r in rows)*1000/len(rows):.3f} ms",flush=True)
        metrics={}
        for name in ('baseline',variant):
            timings=[r['seconds']*1000 for run in results if run['variant']==name for r in run['rows']]
            metrics[name]={'mean_ms':statistics.mean(timings),'median_ms':statistics.median(timings),
                           'p95_ms':sorted(timings)[int(.95*(len(timings)-1))],
                           'different_moves':next(run['changed_moves'] for run in results if run['variant']==name)}
        runs[variant]={'metrics':metrics,'time_saved_percent':100*(1-metrics[variant]['mean_ms']/metrics['baseline']['mean_ms']),
                       'runs':results}
    report={'positions':len(reference),'method':'ABBA; full observed per-turn GRU history; 1 thread CPU; warmup excluded',
            'source_games':600,'draw_points':0.5,'variants':runs}
    (Path(args.assets)/'benchmark.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:{j:v for j,v in value.items() if j!='runs'} for k,value in runs.items()},indent=2))


if __name__=='__main__':main()
