"""Compare optimized mechanics against the pre-optimization source snapshot."""
import importlib.util
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import chess
import numpy as np
from tonkai.piece_safety import piece_senses
from tonkai.king_safety import king_senses

def previous(name):
    source = Path(__file__).parent / 'cognitive-speed-baseline/tonkai' / (name + '.py')
    spec = importlib.util.spec_from_file_location('baseline_' + name, source)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module

old_pieces, old_kings = previous('piece_safety'), previous('king_safety')
rng = np.random.default_rng(348)
value = lambda piece, square: (piece.piece_type * .71 + (square % 7) * .12 - .9)
board = chess.Board()
for index in range(500):
    if board.is_game_over(claim_draw=True) or len(board.move_stack) >= 180: board = chess.Board()
    board.push(list(board.legal_moves)[int(rng.integers(board.legal_moves.count()))])
    assert king_senses(board) == old_kings.king_senses(board), board.fen()
    depth = (1, 2, 6)[index % 3]
    assert piece_senses(board, value, depth=depth) == old_pieces.piece_senses(board, value, depth=depth), board.fen()
before = json.loads(Path('perf_results/cognitive-speed-before.json').read_text())
after = json.loads(Path('perf_results/cognitive-speed-after.json').read_text())
assert before['decisions'] == after['decisions']
report = dict(mechanical_positions=500, unchanged_decisions=len(before['decisions']), max_score_difference=0)
Path(__file__).with_suffix('.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report))
