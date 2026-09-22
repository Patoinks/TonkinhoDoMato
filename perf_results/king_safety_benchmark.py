"""Matched checkpoint/positions: added king-prior latency, not playing strength."""
import json
from pathlib import Path
import statistics
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import chess
import torch
from tonkai.cognitive_agent import CognitiveAgent
from tonkai.cognitive_config import CognitiveConfig
from tonkai.cognitive_trainer import CognitiveTrainer

torch.set_num_threads(1)
source = Path('checkpoints/cognitive-1000-500/latest.pt')
state = torch.load(source, weights_only=False, map_location='cpu')
cfg = CognitiveConfig.from_dict(state['config'])
trainer = CognitiveTrainer(cfg); trainer.load_state_dict(state)
boards = []
for episode in state['replay']['episodes'][-40:]:
    board = chess.Board(episode['initial_fen'] or chess.STARTING_FEN)
    for move in episode['actions'][:12]: board.push_uci(move)
    if not board.is_game_over(claim_draw=True): boards.append(board)
    if len(boards) == 16: break
times = {0: [], .06: []}
for repeat in range(3):
    for board in boards:
        for weight in ((0, .06) if repeat % 2 == 0 else (.06, 0)):
            cfg.king_safety_weight = weight
            agent = CognitiveAgent(trainer.net, cfg, trainer.knowledge)
            agent.observe(board, board.turn)
            started = time.perf_counter()
            agent.evaluate(board, board.turn, explore=False)
            if repeat: times[weight].append((time.perf_counter() - started) * 1000)
report = dict(positions=len(boards), measured_repeats=2, checkpoint=str(source), threads=1,
              scope='Full candidate evaluation; piece safety stays enabled; no training or strength test',
              latency_ms={str(k): dict(mean=statistics.mean(v), median=statistics.median(v), maximum=max(v)) for k, v in times.items()})
Path(__file__).with_suffix('.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report))
