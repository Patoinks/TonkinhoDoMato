"""Frozen checkpoint-600 comparisons: own v2 versus original, both with draws=0.5."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from multiprocessing import get_context
import os
from pathlib import Path
import sys
import threading
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import paths
from v2.rules import adjudicate, draw_config

_w = {}


def write(path, data):
    temp = Path(str(path)+'.tmp')
    temp.write_text(json.dumps(data, indent=2), encoding='utf-8')
    os.replace(temp,path)


def init(checkpoint, assets, engine, out):
    import torch
    import psutil_free_priority
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    _w.update(checkpoint=checkpoint,assets=assets,engine_path=engine,out=Path(out),engine=None)


def make(variant, seed):
    import numpy as np
    if variant=='baseline':
        from claudio_load import load_claudio
        from claudio_agent import Claudio,ClaudioConfig
        net,values,config,_,king=load_claudio(_w['checkpoint'],with_config=True,with_king=True)
        return Claudio(net,values,ClaudioConfig.with_settings([],base=draw_config(config)),
                       rng=np.random.default_rng(seed),king_model=king)
    from v2.agent import load_v2
    return load_v2(_w['checkpoint'],_w['assets'],variant=variant,rng=np.random.default_rng(seed))


def engine():
    if _w['engine'] is None:
        from tonkai.uci_opponent import UCIOpponent
        _w['engine']=UCIOpponent(_w['engine_path'],1800,0.002)
    _w['engine'].reset()
    return _w['engine']


def export_pgn(board, players, result, event):
    import chess.pgn
    game=chess.pgn.Game.from_board(board)
    game.headers.update(Event=event,Date='2026.10.09',White=players[True],Black=players[False],Result=result)
    return str(game)


def job(item):
    import chess
    import numpy as np
    mode,game_id=item['mode'],item['game']
    if mode=='head':
        players={'baseline':make('baseline',[7711,game_id,0]),'all':make('all',[7711,game_id,1])}
        ids={True:'baseline' if game_id%2 else 'all',False:'all' if game_id%2 else 'baseline'}
        board=chess.Board()
        # Different own-experience openings; same opening with swapped colours in each pair.
        experience=Path(_w['assets'])/'experience.json'
        if not experience.exists():experience=paths.ROOT/'claudio/results/v2-checkpoint600/experience.json'
        source=json.loads(experience.read_text())['games']
        opening=source[(game_id//2)%len(source)]['moves'].split()[:4]
        for uci in opening: board.push_uci(uci)
    else:
        variant=item['variant']
        me=game_id%2==0
        players={variant:make(variant,[7703,game_id,int(me)])}
        ids={me:variant,not me:'Stockfish1800'}
        board=chess.Board(item.get('fen',chess.STARTING_FEN))
        if mode=='stress':
            me=board.turn
            ids={me:variant,not me:'Stockfish1800'}
        engine()
    started=time.perf_counter()
    times={key:0.0 for key in players}
    calls={key:0 for key in players}
    final=adjudicate(board)
    while final is None and board.ply()-board.root().ply()<item.get('max_plies',300):
        key=ids[board.turn]
        if key=='Stockfish1800':
            move=_w['engine'].choose(board,np.random.default_rng([7721,game_id]))
        else:
            start=time.perf_counter()
            if hasattr(players[key],'options'):
                move=players[key].choose(board,terminal_checked=True)
            else:
                move=players[key].choose(board)
            times[key]+=time.perf_counter()-start
            calls[key]+=1
        if move not in board.legal_moves: raise AssertionError(f'{key} illegal move {move}')
        board.push(move)
        final=adjudicate(board)
    winner=None if final is None or final['winner'] is None else ids[final['winner']]
    row={'mode':mode,'game':game_id,'variant':item.get('variant','head'),'white':ids[True],'black':ids[False],
         'winner':winner,'end':final['end'] if final else 'max_plies','result':final['result'] if final else '*',
         'plies':len(board.move_stack),'start_fen':board.root().fen(),'moves':' '.join(m.uci() for m in board.move_stack),
         'time':times,'decisions':calls,'seconds':time.perf_counter()-started,
         'stats':{key:getattr(player,'stats',{}) for key,player in players.items()}}
    folder=_w['out']/mode
    folder.mkdir(exist_ok=True)
    name=f"{row['variant']}-{game_id:04d}"
    write(folder/(name+'.json'),row)
    (folder/(name+'.pgn')).write_text(export_pgn(board,ids,row['result'],f'G16 v2 {mode}'),encoding='utf-8')
    return row


def batch(items):
    try:
        return [job(item) for item in items]
    finally:
        if _w['engine'] is not None: _w['engine'].close()


def table(records, variants):
    result={}
    for variant in variants:
        selected=[r for r in records if variant in r['time']]
        n=len(selected)
        wins=sum(r['winner']==variant for r in selected)
        losses=sum(r['winner'] is not None and r['winner']!=variant for r in selected)
        unfinished=sum(r['end']=='max_plies' for r in selected)
        draws=n-wins-losses-unfinished
        seconds=sum(r['time'][variant] for r in selected)
        decisions=sum(r['decisions'][variant] for r in selected)
        points=wins+0.5*draws
        end_reasons={reason:sum(r['end']==reason for r in selected) for reason in sorted({r['end'] for r in selected})}
        stats={}
        for row in selected:
            for key,value in row['stats'].get(variant,{}).items():
                stats[key]=stats.get(key,0)+value
        result[variant]={'games':n,'wins':wins,'draws':draws,'losses':losses,'unfinished':unfinished,
                         'points':points,'score':points/n if n else 0,'finished_score':points/(n-unfinished) if n>unfinished else None,
                         'win_rate':wins/n if n else 0,'mean_plies':sum(r['plies'] for r in selected)/n if n else 0,
                         'seconds_per_game':seconds/n if n else 0,'ms_per_decision':1000*seconds/decisions if decisions else 0,
                         'end_reasons':end_reasons,'module_stats':stats}
    return result


def stress_positions(assets, count=20):
    import chess
    import random
    from v2.agent import native
    target=Path(assets)/'stress-positions.json'
    if target.exists(): return json.loads(target.read_text())['positions']
    rng=random.Random(8801)
    positions=[]
    candidates=['7k/1R6/7B/P7/8/8/4K3/8 w - - 53 143']
    attempts=0
    while len(positions)<count and attempts<200:
        attempts+=1
        if candidates:
            fen=candidates.pop()
            board=chess.Board(fen)
        else:
            board=chess.Board(None)
            wk,bk,piece=rng.sample(range(64),3)
            board.set_piece_at(wk,chess.Piece(chess.KING,chess.WHITE))
            board.set_piece_at(bk,chess.Piece(chess.KING,chess.BLACK))
            board.set_piece_at(piece,chess.Piece(chess.QUEEN if attempts%2 else chess.ROOK,chess.WHITE))
            board.turn=chess.WHITE
            if not board.is_valid() or adjudicate(board): continue
            fen=board.fen()
        proof=native.mating_plan(fen,[],8,2_000_000)
        if proof['moves']:
            positions.append({'fen':fen,'proved_mate_moves':proof['moves'],'nodes':proof['nodes']})
            print(f"conversion positions {len(positions)}/{count}: mate {proof['moves']}",flush=True)
    if len(positions)<count: raise RuntimeError('Not enough actually proved conversion positions')
    write(target,{'positions':positions,'rule':'Winning proven by our own full-legal-move solver, no engine score'})
    return positions


def run(args):
    out=Path(args.out)
    out.mkdir(parents=True,exist_ok=True)
    variants=args.variants.split(',')
    if args.mode=='stress':
        positions=stress_positions(args.assets,args.games)
        items=[{'mode':'stress','variant':variant,'game':i,'fen':row['fen'],'max_plies':150}
               for i,row in enumerate(positions) for variant in variants]
    elif args.mode=='head':
        variants=['baseline','all']
        items=[{'mode':'head','game':i,'max_plies':300} for i in range(args.games)]
    else:
        items=[{'mode':'sf','variant':variant,'game':i,'max_plies':300}
               for i in range(args.games) for variant in variants]
    folder=out/args.mode
    folder.mkdir(exist_ok=True)
    existing={p.stem for p in folder.glob('*.json')}
    items=[item for item in items if f"{item.get('variant','head')}-{item['game']:04d}" not in existing]
    begun=time.perf_counter()
    done=threading.Event()
    def status():
        rows=[json.loads(p.read_text(encoding='utf-8')) for p in folder.glob('*.json')]
        elapsed=time.perf_counter()-begun
        data={'state':'running','mode':args.mode,'games':len(rows),'target':args.games*(1 if args.mode=='head' else len(variants)),
              'table':table(rows,variants),'seconds':elapsed,'updated_at':datetime.now(timezone.utc).isoformat(),
              'draw_points':0.5,'source_games':600,'variants':variants}
        write(out/'evaluation-status.json',data)
        return rows,data
    def monitor():
        while not done.wait(1):
            try: status()
            except OSError: pass
    watcher=threading.Thread(target=monitor,daemon=True)
    watcher.start()
    try:
        with ProcessPoolExecutor(args.workers,mp_context=get_context('spawn'),initializer=init,
                                 initargs=(args.checkpoint,args.assets,args.engine,str(out))) as pool:
            parts=[items[i::args.workers] for i in range(min(args.workers,len(items)))]
            futures=[pool.submit(batch,part) for part in parts]
            for future in futures: future.result()
    finally:
        done.set()
        watcher.join(timeout=3)
    rows,data=status()
    data['state']='completed'
    data['list']=sorted(rows,key=lambda r:(r['game'],r['variant']))
    data['source_sha256']=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()
    data['weights_frozen']=True
    write(out/f'{args.mode}-results.json',data)
    write(out/'evaluation-status.json',{k:v for k,v in data.items() if k!='list'})
    print(json.dumps({k:v for k,v in data.items() if k not in ('list','table')},indent=2),flush=True)
    print(json.dumps(data['table'],indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('sf','head','stress'),default='sf')
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--assets',required=True)
    parser.add_argument('--out',required=True)
    parser.add_argument('--engine',default=str(paths.STOCKFISH))
    parser.add_argument('--games',type=int,default=100)
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--variants',default='baseline,full,teacher')
    run(parser.parse_args())
