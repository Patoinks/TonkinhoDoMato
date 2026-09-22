"""Frozen-model actor benchmark and decision record, including actual history."""
import argparse
import cProfile
import io
import json
from pathlib import Path
import pstats
import statistics
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import chess
import torch
from tonkai.cognitive_agent import CognitiveAgent
from tonkai.cognitive_config import CognitiveConfig
from tonkai.cognitive_trainer import CognitiveTrainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--profile', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    checkpoint = Path('checkpoints/cognitive-1000-500/latest.pt')
    state = torch.load(checkpoint, weights_only=False, map_location='cpu')
    cfg = CognitiveConfig.from_dict(state['config'])
    trainer = CognitiveTrainer(cfg); trainer.load_state_dict(state)
    boards = []
    for episode in state['replay']['episodes'][-16:]:
        for plies in (12, 40, 80):
            board = chess.Board(episode['initial_fen'] or chess.STARTING_FEN)
            for action in episode['actions'][:plies]: board.push_uci(action)
            if not board.is_game_over(claim_draw=True): boards.append(board)
    times, decisions = [], []
    profile = cProfile.Profile()
    for repeat in range(3):
        for board in boards:
            agent = CognitiveAgent(trainer.net, cfg, trainer.knowledge)
            agent.observe(board, board.turn)
            if args.profile and repeat == 2: profile.enable()
            start = time.perf_counter()
            choice = agent.evaluate(board, board.turn, explore=False)
            elapsed = (time.perf_counter() - start) * 1000
            if args.profile and repeat == 2: profile.disable()
            if repeat == 1: times.append(elapsed)
            if repeat == 2:
                decisions.append(dict(fen=board.fen(), moves=[m.uci() for m in choice.moves],
                    scores=choice.scores.tolist(), best=choice.moves[int(choice.scores.argmax())].uci()))
    report = dict(checkpoint=str(checkpoint), positions=len(boards), config=cfg.to_dict(),
        latency_ms=dict(mean=statistics.mean(times), median=statistics.median(times)), decisions=decisions)
    Path(args.output).write_text(json.dumps(report, indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in ('config', 'decisions')}))
    if args.profile:
        stream = io.StringIO(); pstats.Stats(profile, stream=stream).sort_stats('cumtime').print_stats(35)
        Path(args.output).with_suffix('.profile.txt').write_text(stream.getvalue())
        print(stream.getvalue())

if __name__ == '__main__': main()
