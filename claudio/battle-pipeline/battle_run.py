"""Prepare and run repeatable frozen battles from the validated local registry."""
import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime, timezone
import hashlib
from itertools import combinations
import json
from multiprocessing import get_context
import os
from pathlib import Path
import sys
import time
import unicodedata
import uuid

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
import battle_backend as backend

def normalize(text):
    text=unicodedata.normalize('NFKD',text).encode('ascii','ignore').decode().lower()
    return ''.join(c for c in text if c.isalnum())

def select_models(registry,names):
    lookup={normalize(p['id']):p for p in registry['players']}
    by_id={p['id']:p for p in registry['players']}
    for alias,pid in registry['aliases'].items(): lookup[normalize(alias)]=by_id[pid]
    for p in registry['players']: lookup[normalize(p['name'])]=p
    if names.strip().lower()=='all': return list(registry['players'])
    selected=[]
    for name in names.split(','):
        if normalize(name) not in lookup: raise ValueError(f'Unknown model: {name}. Use -List.')
        model=lookup[normalize(name)]
        if model not in selected: selected.append(model)
    if len(selected)<2: raise ValueError('Select at least two different models')
    return selected

def jobs_for(ids,games):
    if games<1: raise ValueError('Games must be positive')
    pairs=list(combinations(ids,2))
    return [{'id':f'{a}__{b}__{g:03d}','a':a,'b':b,'game':g,'opening':g//2,
             'white':a if g%2==0 else b,'black':b if g%2==0 else a}
            for opening in range((games+1)//2) for a,b in pairs
            for g in (2*opening,2*opening+1) if g<games]

def sources_of(spec):
    return spec.get('checkpoints',[])+[spec[key] for key in ('checkpoint','genes','king_model','engine') if key in spec]

def prepare(args):
    begun=time.perf_counter()
    registry=json.loads((HERE/'registry.json').read_text(encoding='utf-8'))
    players=select_models(registry,args.players)
    players=json.loads(json.dumps(players))
    # Only the selected cached files are checked. No copying, building, training,
    # probing or repository lookup takes place before an ordinary battle.
    for spec in players:
        for source in sources_of(spec):
            path=(ROOT/source['path']).resolve()
            if hashlib.sha256(path.read_bytes()).hexdigest()!=source['sha256']:
                raise ValueError(f'Cached input changed: {path}')
            source['path']=str(path)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    run=HERE/'runs'/f'{stamp}-bo{args.games}-{uuid.uuid4().hex[:6]}'
    run.mkdir(parents=True)
    manifest={**registry,'players':players,'repository':str(ROOT),'games_per_pair':args.games,
              'workers':args.workers,'seed':args.seed,'random_plies':args.random_plies,'max_plies':args.max_plies,
              'pairs':len(players)*(len(players)-1)//2,'created_utc':datetime.now(timezone.utc).isoformat(),
              'client_timezone':'Europe/Lisbon','label':args.label,'training':False}
    manifest['date']=args.date
    manifest['total_games']=manifest['pairs']*args.games
    manifest['prepare_seconds']=time.perf_counter()-begun
    backend.write_json(run/'manifest.json',manifest)
    backend.write_json(run/'status.json',{'state':'prepared','completed_games':0,'total_games':manifest['total_games']})
    (HERE/'LATEST.txt').write_text(str(run),encoding='utf-8')
    print(json.dumps({'run':str(run),'total_games':manifest['total_games'],'workers':args.workers,
                      'models':[p['name'] for p in players],'prepare_seconds':manifest['prepare_seconds']}))

def report(run,manifest,records,state,started,error=None):
    backend.OUT=run
    backend.publish(manifest,records,state,started,error)
    status=json.loads((run/'status.json').read_text(encoding='utf-8'))
    status['games_per_pair']=manifest['games_per_pair']
    status['models']=[p['name'] for p in manifest['players']]
    backend.write_json(run/'status.json',status)
    text=(run/'standings.md').read_text(encoding='utf-8').replace('# Fase de grupos BO20',f"# Batalha BO{manifest['games_per_pair']}")
    text=text.replace('2 workers, prioridade baixa.',f"{manifest['workers']} workers, prioridade baixa.")
    text=text.replace('Cada par joga 10 aberturas, com as cores trocadas.',
                      f"Cada par joga {manifest['games_per_pair']} jogos. Cores alternadas e aberturas emparelhadas; BO impar termina com um jogo sem par.")
    (run/'standings.md').write_text(text,encoding='utf-8')

def execute(args):
    import psutil_free_priority
    run=Path(args.execute).resolve()
    run.relative_to((HERE/'runs').resolve())
    manifest=json.loads((run/'manifest.json').read_text(encoding='utf-8'))
    history=run/'games.jsonl'
    if history.exists() and not args.resume: raise ValueError('Existing games: use -Resume')
    records=[json.loads(line) for line in history.read_text(encoding='utf-8').splitlines()] if history.exists() else []
    done_ids={r['id'] for r in records}
    if len(done_ids)!=len(records): raise ValueError('Duplicate saved game IDs')
    if args.resume: (run/'STOP').unlink(missing_ok=True)
    todo=iter([j for j in jobs_for([p['id'] for p in manifest['players']],manifest['games_per_pair']) if j['id'] not in done_ids])
    started=time.perf_counter()
    report(run,manifest,records,'running',started)
    pool=ProcessPoolExecutor(manifest['workers'],mp_context=get_context('spawn'),initializer=backend.init_worker,
                             initargs=(str(run/'manifest.json'),))
    pending={}
    def submit():
        job=next(todo,None)
        if job is not None: pending[pool.submit(backend.play,job)]=job
    for _ in range(manifest['workers']): submit()
    state='completed'
    try:
        with history.open('a',encoding='utf-8') as stream:
            while pending:
                finished,_=wait(pending,timeout=1,return_when=FIRST_COMPLETED)
                if (run/'STOP').exists(): state='stopped'
                for future in finished:
                    job=pending.pop(future)
                    row=future.result()
                    if row['id']!=job['id'] or row['id'] in done_ids: raise ValueError('Invalid game ID')
                    done_ids.add(row['id'])
                    records.append(row)
                    stream.write(json.dumps(row)+'\n'); stream.flush()
                    backend.OUT=run
                    backend.append_pgn(row,manifest)
                    report(run,manifest,records,'running' if state!='stopped' else 'stopping',started)
                    if state!='stopped': submit()
        pool.shutdown(wait=True)
        if state=='completed' and len(records)!=manifest['total_games']: raise ValueError('Incomplete battle')
        for spec in manifest['players']:
            for source in sources_of(spec):
                if hashlib.sha256(Path(source['path']).read_bytes()).hexdigest()!=source['sha256']:
                    raise ValueError('Battle changed an input file')
        backend.write_json(run/'results.json',{'state':state,'manifest':manifest,
            'table':backend.standings(manifest['players'],records),'list':records,
            'seconds':time.perf_counter()-started,'weights_unchanged':True})
        report(run,manifest,records,state,started)
    except BaseException as error:
        report(run,manifest,records,'failed',started,repr(error))
        import subprocess
        for process in list(getattr(pool,'_processes',{}).values()):
            if os.name=='nt': subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],capture_output=True)
            else: process.terminate()
        pool.shutdown(wait=False,cancel_futures=True)
        raise

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--execute')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--players',default='G16,G15Treinado')
    parser.add_argument('--games',type=int,default=15)
    parser.add_argument('--workers',type=int,choices=range(1,7),default=2)
    parser.add_argument('--seed',type=int,default=2027)
    parser.add_argument('--random-plies',type=int,default=4)
    parser.add_argument('--max-plies',type=int,default=300)
    parser.add_argument('--label',default='battle')
    parser.add_argument('--date',default=datetime.now(timezone.utc).strftime('%Y.%m.%d'))
    args=parser.parse_args()
    if args.execute: execute(args)
    elif args.prepare: prepare(args)
    else: parser.error('Use --prepare or --execute')

if __name__=='__main__': main()
