"""Bounded, in-memory training benchmark with real Windows actors and Rodent.

Reads a checkpoint and optional launch resignation settings. Does not save
weights, publish, log campaign games, or change production files.
"""
import argparse
import hashlib
from io import BytesIO
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--launch', type=Path, required=True)
    parser.add_argument('--source-root', type=Path, default=ROOT)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--games', type=int, default=64)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.games <= args.workers or args.workers < 1:
        parser.error('Use more games than workers, and at least one worker')
    if args.output.resolve() in (args.checkpoint.resolve(), args.launch.resolve()):
        parser.error('Output cannot overwrite an input')
    sys.path.append(str(ROOT))
    sys.path.insert(0, str(args.source_root.resolve()))
    import numpy as np
    import torch
    from tonkai import fast_rules
    from tonkai.captures import CaptureStats
    from tonkai.cognitive_config import CognitiveConfig
    from tonkai.cognitive_pool import CognitivePool
    from tonkai.cognitive_trainer import CognitiveTrainer
    from tonkai.resign import ResignConfig, ResignController, hope_threshold
    from tonkai.uci_opponent import UCIOpponent

    torch.set_num_threads(1)
    blob = args.checkpoint.read_bytes()
    digest = hashlib.sha256(blob).hexdigest()
    state = torch.load(BytesIO(blob), weights_only=False, map_location='cpu')
    cfg = CognitiveConfig.from_dict(state['config'])
    cfg.num_workers = args.workers
    cfg.device = cfg.inference_device = 'cpu'
    trainer = CognitiveTrainer(cfg)
    trainer.load_state_dict(state)
    captures = CaptureStats(cfg.hot_piece_kills)
    if state.get('capture_stats'):
        captures.load_state_dict(state['capture_stats'])
    launch = json.loads(args.launch.read_text())
    rc = ResignConfig(**(launch.get('resign') or {}))
    resign = ResignController(rc, hope_threshold(rc, cfg.objective, cfg.survival_gamma, cfg.victory_weight))
    if state.get('resign'):
        resign.load_state_dict(state['resign'])
    rng = np.random.default_rng(741)
    submitted = completed = plies = updates = experiments = 0
    learner_seconds = submit_seconds = 0.
    warmed_at = None
    counts = dict(win=0, draw=0, loss=0, resigned=0, unfinished=0)
    actor_seconds = []
    started = time.perf_counter()
    with UCIOpponent(cfg.opponent_path, cfg.opponent_elo, cfg.opponent_move_time) as external:
        with CognitivePool(cfg, cfg.opponent_path) as pool:
            while completed < args.games:
                while len(pool.pending) < pool.capacity and submitted < args.games:
                    begin = time.perf_counter()
                    pool.submit(trainer.net, trainer.knowledge, rng.integers(0, 2**31 - 1), submitted + 1,
                                resign_plan=resign.plan(), kd_values=captures.piece_values() if cfg.kd_weight else None)
                    submit_seconds += time.perf_counter() - begin
                    submitted += 1
                episode, board, colour, stats, trace = pool.receive()
                begin = time.perf_counter()
                trainer.ingest(episode, trace, colour)
                report = stats.get('resign')
                outcome = ('resigned' if report and report['resigned'] else 'unfinished' if not episode.terminated
                           else 'loss' if episode.lost else 'draw' if stats['result'] == '1/2-1/2' else 'win')
                captures.add(board.move_stack, colour, outcome)
                resign.update(report, lost=episode.lost)
                experiments += len(trainer.investigate(episode, trace, colour, external))
                for _ in range(cfg.updates_per_game):
                    trainer.update()
                    updates += 1
                learner_seconds += time.perf_counter() - begin
                completed += 1
                plies += stats['plies']
                counts[outcome] += 1
                actor_seconds.append(stats['actor_seconds'])
                if completed == args.workers:
                    warmed_at = time.perf_counter()
                if completed % 16 == 0:
                    print(f'{completed}/{args.games} games', flush=True)
            finished = time.perf_counter()
    report = dict(source=str(args.source_root.resolve()), backend=fast_rules.BACKEND, workers=args.workers,
                  games=completed, seconds=finished-started, games_per_second=completed/(finished-started),
                  after_first_wave_games_per_second=(completed-args.workers)/(finished-warmed_at),
                  plies=plies, plies_per_second=plies/(finished-started), mean_plies=plies/completed,
                  mean_actor_seconds=float(np.mean(actor_seconds)), learner_seconds=learner_seconds,
                  submit_seconds=submit_seconds, updates=updates, experiments=experiments, results=counts,
                  checkpoint_sha256=digest, checkpoint_unchanged=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()==digest,
                  opponent=external.info, config=cfg.to_dict(), resign_config=launch.get('resign'),
                  conditions='CPU actors and learner; original search/rewards; real Rodent timing and asynchronous completion are nondeterministic. Includes learning/captures/experiments/resignation; excludes disk checkpoints, publishing and campaign logging. In-memory only.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('config', 'opponent', 'resign_config')}), flush=True)


if __name__ == '__main__':
    main()
