"""Short real Rodent training runs in memory; never save/publish production weights."""
import argparse
import json
from pathlib import Path
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument('--source-root')
parser.add_argument('--workers', type=int, required=True)
parser.add_argument('--games', type=int, default=24)
parser.add_argument('--output', required=True)
args = parser.parse_args()
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(args.source_root).resolve() if args.source_root else ROOT))
import numpy as np
import torch
from tonkai.cognitive_config import CognitiveConfig
from tonkai.cognitive_trainer import CognitiveTrainer
from tonkai.cognitive_pool import CognitivePool
from tonkai.uci_opponent import UCIOpponent

def main():
    torch.set_num_threads(1)
    state = torch.load(ROOT / 'checkpoints/cognitive-1000-500/latest.pt', weights_only=False, map_location='cpu')
    cfg = CognitiveConfig.from_dict(state['config']); cfg.num_workers = args.workers; cfg.validate()
    trainer = CognitiveTrainer(cfg); trainer.load_state_dict(state)
    rng = np.random.default_rng(741)
    submitted = completed = plies = 0
    learner_seconds = 0.; warmed_at = None
    counts = {'wins': 0, 'draws': 0, 'losses': 0, 'truncated': 0}
    start = time.perf_counter()
    with UCIOpponent(cfg.opponent_path, cfg.opponent_elo, cfg.opponent_move_time) as external:
        with CognitivePool(cfg, cfg.opponent_path) as pool:
            while completed < args.games:
                while len(pool.pending) < pool.capacity and submitted < args.games:
                    pool.submit(trainer.net, trainer.knowledge, rng.integers(0, 2**31 - 1), submitted + 1)
                    submitted += 1
                episode, board, colour, stats, trace = pool.receive()
                learner_start = time.perf_counter()
                trainer.ingest(episode, trace, colour)
                trainer.investigate(episode, trace, colour, external)
                for _ in range(cfg.updates_per_game): trainer.update()
                learner_seconds += time.perf_counter() - learner_start
                completed += 1; plies += stats['plies']
                counts['truncated' if not episode.terminated else 'losses' if episode.lost
                       else 'draws' if stats['result'] == '1/2-1/2' else 'wins'] += 1
                if completed == args.workers: warmed_at = time.perf_counter()
                if completed % 8 == 0: print(f'{completed}/{args.games} games', flush=True)
            finished = time.perf_counter()
    report = dict(source=args.source_root or 'working-tree', workers=args.workers, games=completed,
        seconds=finished-start, games_per_second=completed/(finished-start),
        after_first_wave_games_per_second=(completed-args.workers)/(finished-warmed_at),
        positions_per_second=plies/(finished-start), mean_plies=plies/completed,
        learner_seconds=learner_seconds, results=counts, opponent=external.info, config=cfg.to_dict())
    Path(args.output).write_text(json.dumps(report, indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in ('config','opponent')}), flush=True)

if __name__ == '__main__': main()
