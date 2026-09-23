"""Isolated fixed-checkpoint execution benchmark; never trains/saves a campaign."""
import argparse
import copy
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
WORK = ROOT / 'perf_results/execution-fastpaths-20260923'


def pool_trial(args, state, cfg):
    import hashlib
    import numpy as np
    import torch
    from tonkai.cognitive_trainer import CognitiveTrainer
    from tonkai.cognitive_pool import CognitivePool
    from tonkai.opponents import choose_kind
    from tonkai.uci_opponent import UCIOpponent
    from tonkai.captures import CaptureStats
    cfg.num_workers = 2 if args.mode == 'pool-broker' else 8
    cfg.device = 'cuda' if args.mode == 'pool-fast' else 'cpu'
    cfg.batch_search = args.mode in ('pool-fast', 'pool-batch')
    cfg.inference_broker = args.mode == 'pool-broker'
    cfg.mix_self, cfg.mix_tiny, cfg.mix_fairy = .75, .20, .04
    cfg.opponent_path = str(ROOT/'engines/stockfish-11/stockfish-11-win/Windows/stockfish_20011801_x64_bmi2.exe')
    cfg.opponent_elo, cfg.opponent_move_time = 1500, .002
    cfg.fairy_path = str(ROOT/'engines/fairy-stockfish/fairy-stockfish_x86-64-bmi2.exe')
    cfg.fairy_elo = 1000
    trainer = CognitiveTrainer(cfg); trainer.load_state_dict(state)
    if args.mode == 'pool-fast':
        from tonkai.cuda_learner import CudaUnroll
        trainer._unroll = CudaUnroll(trainer)
    for _ in range(3): trainer.update()
    # Fixed actor snapshots make the paired workload reproducible despite
    # different learner speed/completion order. This is not a live campaign.
    frozen = copy.deepcopy(trainer.net).cpu().eval()
    frozen.load_state_dict(state['model'])
    knowledge = copy.deepcopy(trainer.knowledge)
    captures = CaptureStats(cfg.hot_piece_kills)
    if state.get('capture_stats'): captures.load_state_dict(state['capture_stats'])
    kd_values = captures.piece_values()
    actions, counts, update_seconds, learned = {}, {}, 0., 0
    timings = dict(submit=0., receive=0., ingest=0., investigate=0., actors=0.)
    games = 8 if args.mode == 'pool-broker' else 64
    rng = np.random.default_rng(812)
    kinds = [choose_kind(cfg,rng) for _ in range(games + cfg.num_workers)]
    external = UCIOpponent(cfg.opponent_path,cfg.opponent_elo,cfg.opponent_move_time)
    try:
        with CognitivePool(cfg,cfg.opponent_path) as pool:
            for i in range(cfg.num_workers):
                pool.submit(frozen,knowledge,7000+i,i,kind=kinds[i],kd_values=kd_values)
            for _ in range(cfg.num_workers): pool.receive()
            started = time.perf_counter(); submitted = completed = 0
            while completed < games:
                while submitted < games and len(pool.pending) < pool.capacity:
                    phase_start=time.perf_counter()
                    pool.submit(frozen,knowledge,9000+submitted,submitted,
                                kind=kinds[cfg.num_workers+submitted],kd_values=kd_values)
                    timings['submit'] += time.perf_counter()-phase_start
                    submitted += 1
                phase_start=time.perf_counter()
                episode, board, colour, stats, trace = pool.receive()
                timings['receive'] += time.perf_counter()-phase_start
                timings['actors'] += stats['actor_seconds']
                completed += 1
                actions[stats['game_number']] = list(episode.actions)
                kind = stats['opponent_kind']; counts[kind] = counts.get(kind,0)+1
                phase_start=time.perf_counter()
                trainer.ingest(episode,trace,colour)
                timings['ingest'] += time.perf_counter()-phase_start
                phase_start=time.perf_counter()
                trainer.investigate(episode,trace,colour,external)
                timings['investigate'] += time.perf_counter()-phase_start
                update_started=time.perf_counter()
                for _ in range(cfg.updates_per_game):
                    report=trainer.update(); learned += report['learned_positions']
                update_seconds += time.perf_counter()-update_started
            elapsed=time.perf_counter()-started
            stats = dict(pool.broker.stats) if pool.broker else {}
    finally:
        external.close()
    plies=sum(len(x) for x in actions.values())
    reference=WORK/'pool-actions.json'
    ordered={str(k):v for k,v in sorted(actions.items())}
    if args.mode=='pool-base' and not reference.exists(): reference.write_text(json.dumps(ordered))
    expected=json.loads(reference.read_text()) if reference.exists() else {}
    return dict(mode=args.mode,repetition=args.repetition,seconds=elapsed,games=games,
                games_per_second=games/elapsed,plies=plies,plies_per_second=plies/elapsed,
                learner_seconds=update_seconds,learned_positions=learned,updates=games*cfg.updates_per_game,
                opponents=counts,broker=stats,timings=timings,
                changed_games=sum(expected.get(k)!=v for k,v in ordered.items()) if expected else None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', required=True, choices=('baseline','compact','cuda','cuda-graphs',
        'actor-base','incremental','batch','combined','broker','pool-base','pool-fast','pool-broker','pool-batch'))
    parser.add_argument('--updates', type=int, default=100)
    parser.add_argument('--repetition', type=int, default=0)
    args = parser.parse_args()
    import numpy as np
    import torch
    from tonkai.cognitive_config import CognitiveConfig
    from tonkai.cognitive_trainer import CognitiveTrainer, CognitiveKnowledge
    torch.set_num_threads(1)
    state = torch.load(WORK/'input.pt', map_location='cpu', weights_only=False)
    cfg = CognitiveConfig.from_dict(state['config'])
    for flag in ('compact_sequences','incremental_encoder','batch_search','inference_broker','cuda_graphs'):
        setattr(cfg, flag, False)
    if args.mode.startswith('pool-'):
        out = pool_trial(args,state,cfg)
        print(json.dumps(out),flush=True)
        (WORK/f'{args.mode}-{args.repetition}.json').write_text(json.dumps(out,indent=2))
        return
    cfg.device = 'cuda' if args.mode.startswith('cuda') else 'cpu'
    cfg.compact_sequences = args.mode == 'compact'
    trainer = CognitiveTrainer(cfg)
    trainer.load_state_dict(state)
    out = dict(mode=args.mode, repetition=args.repetition, source_games=state['games'])
    if args.mode in ('baseline', 'compact', 'cuda', 'cuda-graphs'):
        if args.mode == 'cuda-graphs':
            from tonkai.cuda_learner import CudaUnroll
            trainer._unroll = CudaUnroll(trainer)
        for _ in range(10): trainer.update()
        if cfg.device == 'cuda': torch.cuda.synchronize()
        start = time.perf_counter()
        losses = [trainer.update() for _ in range(args.updates)]
        if cfg.device == 'cuda': torch.cuda.synchronize()
        out['seconds'] = time.perf_counter()-start
        out['updates'] = args.updates
        out['learned_positions'] = sum(row['learned_positions'] for row in losses)
        out['last_loss'] = losses[-1]
        weights = {k:v.detach().cpu() for k,v in trainer.net.state_dict().items()}
        reference = WORK / f'reference-{args.updates}.pt'
        if args.mode == 'baseline' and not reference.exists(): torch.save(weights, reference)
        expected = torch.load(reference, weights_only=True, map_location='cpu')
        out['max_parameter_difference'] = max(float((weights[k]-v).abs().max()) for k,v in expected.items())
    elif args.mode == 'broker':
        from concurrent.futures import ThreadPoolExecutor
        from multiprocessing import get_context
        from tonkai.inference_broker import InferenceBroker, EncoderClient
        net = trainer.net.cpu().eval().requires_grad_(False)
        broker = InferenceBroker(get_context('spawn'), 8)
        x = torch.randn(30, 782)
        # Best-case same-version and realistic different-version groups.
        out['cases'] = []
        try:
            for distinct in (False, True):
                versions=[]
                for i in range(8):
                    if distinct:
                        with torch.no_grad(): net.encoder[0].bias.add_(.0001)
                    versions.append(broker.register(net.encoder))
                clients=[EncoderClient(broker.requests, broker.responses[i],i,v) for i,v in enumerate(versions)]
                with ThreadPoolExecutor(8) as pool:
                    list(pool.map(lambda c:c.encode(x), clients))
                    start=time.perf_counter()
                    for _ in range(100): list(pool.map(lambda c:c.encode(x), clients))
                    elapsed=time.perf_counter()-start
                start=time.perf_counter()
                with torch.inference_mode():
                    for _ in range(800): net.encode(x)
                out['cases'].append(dict(distinct_versions=distinct, broker_seconds=elapsed,
                                         local_serial_seconds=time.perf_counter()-start))
                for v in versions: broker.release(v)
            out['stats']=broker.stats
        finally: broker.close()
    else:
        import chess
        from tonkai.cognitive_agent import CognitiveAgent
        cfg.incremental_encoder = args.mode in ('incremental','combined')
        cfg.batch_search = args.mode in ('batch','combined')
        net=trainer.net.cpu().eval().requires_grad_(False)
        agent=CognitiveAgent(net,cfg,trainer.knowledge)
        positions=[]
        for episode in list(trainer.replay._episodes.values())[:12]:
            board=chess.Board(episode.initial_fen or chess.STARTING_FEN)
            for action in episode.actions[:24]:
                if not board.is_game_over(): positions.append(board.copy(stack=True))
                board.push_uci(action)
        for board in positions[:12]: agent.evaluate(board,True,explore=False)
        start=time.perf_counter()
        scores=[agent.evaluate(board,True,explore=False).scores.copy() for board in positions]
        out['seconds']=time.perf_counter()-start
        out['positions']=len(positions)
        reference=WORK/'actor-reference.pt'
        if args.mode=='actor-base' and not reference.exists(): torch.save(scores,reference)
        expected=torch.load(reference,weights_only=False)
        out['max_score_difference']=max(float(np.max(np.abs(a-b))) for a,b in zip(scores,expected))
        out['argmax_changes']=sum(int(a.argmax()!=b.argmax()) for a,b in zip(scores,expected))
        out['diagnostics']=dict(agent.diagnostics)
    print(json.dumps(out),flush=True)
    if 'scripts/benchmark_execution_paths.py' not in (ROOT/'WORKING.md').read_text(encoding='utf-8'):
        raise RuntimeError('reservation missing')
    (WORK/f'{args.mode}-{args.repetition}.json').write_text(json.dumps(out,indent=2),encoding='utf-8')


if __name__=='__main__': main()
