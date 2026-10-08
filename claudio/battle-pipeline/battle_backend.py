"""Frozen round robin, BO20 per pair, checkpoint snapshots and per-colour results."""
import argparse
import csv
import hashlib
import json
import os
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from itertools import combinations
from multiprocessing import get_context
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'claudio'/'battle-pipeline'/'runs'
MANIFEST=OUT/'manifest.json'
REPO=ROOT
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='1'
sys.path.insert(0,str(REPO/'claudio'))
import paths

_manifest=None
_loaded={}
_tim=None

def plan(ids,games):
    if games < 2 or games % 2:
        raise ValueError('Games per pair must be a positive even number')
    pairs=list(combinations(ids,2))
    return [{'id':f'{a}__{b}__{g:02d}','a':a,'b':b,'game':g,'opening':g//2,
             'white':a if g%2==0 else b,'black':b if g%2==0 else a}
            for opening in range(games//2) for a,b in pairs for g in (2*opening,2*opening+1)]

def init_worker(manifest_path):
    global _manifest
    import torch
    import psutil_free_priority
    os.chdir(REPO)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    _manifest=json.loads(Path(manifest_path).read_text(encoding='utf-8'))

def loaded_player(spec,seed):
    import numpy as np
    from claudio_load import load_players,make_player
    if spec['kind']=='claudio':
        if spec['id'] not in _loaded:
            _loaded[spec['id']]=load_players('+'.join(item['path'] for item in spec['checkpoints']))
        return make_player(_loaded[spec['id']],np.random.default_rng(seed))
    if spec['kind']=='timdgei':
        global _tim
        from match_checkpoints import _load
        from tonkai.cognitive_agent import CognitiveAgent
        from tonkai.opponents import CognitiveOpponent
        from copy import copy
        if _tim is None:
            _,net,knowledge,cfg,kd=_load(spec['checkpoint']['path'])
            genes=json.loads(Path(spec['genes']['path']).read_text(encoding='utf-8')).get('base_genes',{})
            cfg=copy(cfg)
            for key,value in genes.items():
                setattr(cfg,key,value)
            cfg.tactic_search='exact'
            cfg.threat_explore=cfg.trade_explore=0.0
            if 'king_model' in spec:
                cfg.king_safety_model=spec['king_model']['path']
            cfg.validate()
            _tim=(net,knowledge,cfg,kd)
        net,knowledge,cfg,kd=_tim
        return CognitiveOpponent(CognitiveAgent(net,copy(cfg),knowledge,kd_values=kd),copy(cfg),np.random.default_rng(seed))
    raise ValueError(spec['kind'])

def engine_player(spec):
    import chess.engine
    import subprocess
    launch={'creationflags':subprocess.CREATE_NO_WINDOW} if os.name=='nt' else {}
    engine=chess.engine.SimpleEngine.popen_uci(spec['engine']['path'],timeout=10,**launch)
    try:
        if os.name=='nt':
            import ctypes
            kernel=ctypes.windll.kernel32
            handle=kernel.OpenProcess(0x0200,False,engine.transport.get_pid())
            if handle:
                try: kernel.SetPriorityClass(handle,0x4000)
                finally: kernel.CloseHandle(handle)
        options={'Threads':_manifest['engine_threads'],'Hash':_manifest['engine_hash_mb'],
                 'UCI_LimitStrength':spec['elo'] is not None}
        if spec['elo'] is not None:
            options['UCI_Elo']=spec['elo']
        if 'Skill Level' in engine.options:
            options['Skill Level']=engine.options['Skill Level'].max
        for key in options:
            if key not in engine.options:
                raise ValueError(f'Missing engine option {key}')
        engine.configure(options)
        engine.ping()
        return engine,options
    except BaseException:
        engine.close()
        raise

def play(job):
    import chess
    import chess.engine
    import arena
    specs={spec['id']:spec for spec in _manifest['players']}
    ids=(job['white'],job['black'])
    players,engines,options={},{},{}
    try:
        for index,pid in enumerate(ids):
            spec=specs[pid]
            if spec['kind']=='engine':
                engines[pid],options[pid]=engine_player(spec)
            else:
                players[pid]=loaded_player(spec,[_manifest['seed'],job['opening'],sum(map(ord,pid)),index])
        board=arena.opening(_manifest['seed'],job['opening'],_manifest['random_plies'])
        spent={pid:0.0 for pid in ids}
        decisions={pid:0 for pid in ids}
        while not board.is_game_over(claim_draw=True) and len(board.move_stack)<_manifest['max_plies']:
            pid=job['white'] if board.turn else job['black']
            begun=time.perf_counter()
            if pid in engines:
                move=engines[pid].play(board,chess.engine.Limit(time=_manifest['engine_time_seconds'])).move
            else:
                move=players[pid].choose(board)
            spent[pid]+=time.perf_counter()-begun
            decisions[pid]+=1
            if move not in board.legal_moves:
                raise ValueError(f'Illegal move from {pid}: {move}, {board.fen()}')
            board.push(move)
        outcome=board.outcome(claim_draw=True)
        winner=None if outcome is None or outcome.winner is None else job['white'] if outcome.winner else job['black']
        return {**job,'winner':winner,'end':outcome.termination.name if outcome else 'max_plies',
                'result':outcome.result() if outcome else '*','plies':len(board.move_stack),
                'moves':' '.join(move.uci() for move in board.move_stack),
                'time':spent,'decisions':decisions,'engine_options':options}
    finally:
        for engine in engines.values():
            engine.close()

def standings(players,records):
    table={p['id']:{'name':p['name'],'games':0,'wins':0,'draws':0,'losses':0,'unfinished':0,
                   'seconds':0.0,'decisions':0,'white':{'games':0,'wins':0,'draws':0,'losses':0,'unfinished':0},
                   'black':{'games':0,'wins':0,'draws':0,'losses':0,'unfinished':0}} for p in players}
    for row in records:
        for pid in (row['a'],row['b']):
            target=table[pid]
            colour=target['white' if row['white']==pid else 'black']
            key=('unfinished' if row['end']=='max_plies' else 'wins' if row['winner']==pid
                 else 'draws' if row['winner'] is None else 'losses')
            for item in (target,colour):
                item['games']+=1
                item[key]+=1
            target['seconds']+=row['time'][pid]
            target['decisions']+=row['decisions'][pid]
    for row in table.values():
        for item in (row,row['white'],row['black']):
            item['win_rate']=item['wins']/item['games'] if item['games'] else 0.0
        row['seconds_per_game']=row['seconds']/row['games'] if row['games'] else 0.0
        row['ms_per_decision']=1000*row['seconds']/row['decisions'] if row['decisions'] else 0.0
    return dict(sorted(table.items(),key=lambda kv:(-kv[1]['win_rate'],kv[1]['losses'],kv[1]['seconds_per_game'])))

def write_json(path,data):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data,indent=2),encoding='utf-8')
    tmp.replace(path)

def append_pgn(row,manifest):
    import chess
    import chess.pgn
    names={spec['id']:spec['name'] for spec in manifest['players']}
    game=chess.pgn.Game()
    game.headers.update(Event=f"Batalha BO{manifest['games_per_pair']}",Date=manifest.get('date','2026.10.08'),Round=str(row['game']+1),
                        White=names[row['white']],Black=names[row['black']],
                        Result=row['result'],Termination=row['end'],PlyCount=str(row['plies']))
    node=game
    for uci in row['moves'].split():
        node=node.add_variation(chess.Move.from_uci(uci))
    folder=OUT/'partidas'
    folder.mkdir(exist_ok=True)
    with (folder/f"{row['a']}--{row['b']}.pgn").open('a',encoding='utf-8') as stream:
        stream.write(str(game)+'\n\n')

def publish(manifest,records,state,started,error=None):
    table=standings(manifest['players'],records)
    elapsed=time.perf_counter()-started
    info={'pid':os.getpid(),'state':state,'completed_games':len(records),'total_games':manifest['total_games'],
          'participants':len(manifest['players']),'pairs':manifest['pairs'],'workers':manifest['workers'],
          'elapsed_this_process_seconds':elapsed,'games_per_second_this_process':len(records)/max(.001,elapsed),
          'table':table,'training':False,'error':error}
    write_json(OUT/'status.json',info)
    write_json(OUT/'standings.json',table)
    headers=['rank','name','games','wins','draws','losses','unfinished','win_rate','white_win_rate','black_win_rate','seconds_per_game']
    with (OUT/'standings.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f,delimiter=';')
        writer.writerow(headers)
        for rank,row in enumerate(table.values(),1):
            writer.writerow([rank,row['name'],row['games'],row['wins'],row['draws'],row['losses'],row['unfinished'],
                             row['win_rate'],row['white']['win_rate'],row['black']['win_rate'],row['seconds_per_game']])
    text=['# Fase de grupos BO20','',f"Estado: {state}. Jogos: {len(records)}/{manifest['total_games']}. 2 workers, prioridade baixa.",'',
          '| AI | Jogos | V | E | D | Inacabados | WR | Brancas WR | Pretas WR | s/jogo |',
          '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in table.values():
        text.append(f"| {row['name']} | {row['games']} | {row['wins']} | {row['draws']} | {row['losses']} | {row['unfinished']} | {100*row['win_rate']:.1f}% | {100*row['white']['win_rate']:.1f}% | {100*row['black']['win_rate']:.1f}% | {row['seconds_per_game']:.3f} |")
    text+=['','WR = vitorias / todos os jogos. Inacabados aos 300 meios-lances nao contam como empates.',
           'Cada par joga 10 aberturas, com as cores trocadas. A classificacao parcial tem quantidades diferentes de jogos.',
           'Stockfish Max = Stockfish 11, UCI_LimitStrength=false, Skill Level=20, 1 thread e 2 ms/lance.',
           'As tres equipas de especialistas usam a rede da respetiva cor. Nenhum checkpoint e treinado.']
    (OUT/'standings.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    print(json.dumps({key:info[key] for key in ('state','completed_games','total_games','elapsed_this_process_seconds')}),flush=True)

def main():
    import psutil_free_priority
    parser=argparse.ArgumentParser()
    parser.add_argument('--probe',action='store_true')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    manifest=json.loads(MANIFEST.read_text(encoding='utf-8'))
    # Check immutable input snapshots before any game.
    for spec in manifest['players']:
        sources=spec.get('checkpoints',[])+[spec[key] for key in ('checkpoint','genes','king_model','engine') if key in spec]
        for source in sources:
            assert hashlib.sha256(Path(source['path']).read_bytes()).hexdigest()==source['sha256']
    if args.probe:
        init_worker(str(MANIFEST))
        import chess
        proof=[]
        for spec in manifest['players']:
            if spec['kind']=='engine':
                engine,options=engine_player(spec)
                try:
                    import chess.engine
                    move=engine.play(chess.Board(),chess.engine.Limit(time=manifest['engine_time_seconds'])).move
                    assert move in chess.Board().legal_moves
                    proof.append({'id':spec['id'],'name':engine.id,'options':options,'legal_move':move.uci()})
                finally:
                    engine.close()
            else:
                for colour in (True,False):
                    board=chess.Board()
                    if not colour: board.push_uci('e2e4')
                    player=loaded_player(spec,[2027,0,1])
                    move=player.choose(board)
                    assert move in board.legal_moves
                proof.append({'id':spec['id'],'both_colours_legal':True})
            print(f"probe OK: {spec['name']}",flush=True)
        write_json(OUT/'probe.json',proof)
        return
    history=OUT/'games.jsonl'
    if history.exists() and not args.resume:
        raise RuntimeError('Existing results: use --resume to preserve them')
    records=[json.loads(line) for line in history.read_text(encoding='utf-8').splitlines()] if history.exists() else []
    finished={row['id'] for row in records}
    assert len(finished)==len(records)
    jobs=[job for job in plan([spec['id'] for spec in manifest['players']],manifest['games_per_pair']) if job['id'] not in finished]
    started=time.perf_counter()
    publish(manifest,records,'running',started)
    state='completed'
    pool=ProcessPoolExecutor(manifest['workers'],mp_context=get_context('spawn'),initializer=init_worker,initargs=(str(MANIFEST),))
    todo=iter(jobs)
    pending={}
    def submit():
        job=next(todo,None)
        if job is not None: pending[pool.submit(play,job)]=job
    for _ in range(2*manifest['workers']): submit()
    try:
        with history.open('a',encoding='utf-8') as stream:
            while pending:
                done,_=wait(pending,timeout=5,return_when=FIRST_COMPLETED)
                for future in done:
                    job=pending.pop(future)
                    row=future.result()
                    assert row['id']==job['id'] and row['id'] not in finished
                    finished.add(row['id'])
                    records.append(row)
                    stream.write(json.dumps(row)+'\n')
                    stream.flush()
                    append_pgn(row,manifest)
                    if len(records)%10==0 or len(records)==manifest['total_games']:
                        publish(manifest,records,'running',started)
                    if not (OUT/'STOP').exists(): submit()
                if (OUT/'STOP').exists(): state='stopped'
        pool.shutdown(wait=True)
        assert state!='completed' or len(records)==manifest['total_games']
        # Verify checkpoint snapshots again before publishing final results.
        for spec in manifest['players']:
            for source in spec.get('checkpoints',[])+([spec['checkpoint']] if 'checkpoint' in spec else []):
                assert hashlib.sha256(Path(source['path']).read_bytes()).hexdigest()==source['sha256']
        write_json(OUT/'results.json',{'manifest':manifest,'table':standings(manifest['players'],records),'list':records,'state':state})
        publish(manifest,records,state,started)
    except BaseException as error:
        publish(manifest,records,'failed',started,repr(error))
        # Only evaluation workers belonging to this run are cleaned up on failure.
        import subprocess
        for process in list(getattr(pool,'_processes',{}).values()):
            if os.name=='nt': subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],capture_output=True)
            else: process.terminate()
        pool.shutdown(wait=False,cancel_futures=True)
        raise

if __name__=='__main__': main()
