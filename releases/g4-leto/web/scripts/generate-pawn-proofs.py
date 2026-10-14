"""Pack KPvK WDL/DTZ facts for browser use; this does not train the G4 network.

Requires python-chess and local Syzygy KPvK .rtbw/.rtbz tables.
python scripts/generate-pawn-proofs.py --syzygy PATH --output public/tonkai/pawn-kpk-v3.bin
"""
import argparse
import hashlib
import json
from pathlib import Path
import chess
import chess.syzygy

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--syzygy', required=True, type=Path)
parser.add_argument('--output', type=Path, default=Path('public/tonkai/pawn-kpk-v3.bin'))
args = parser.parse_args()
data = bytearray([255]) * (24 * 64 * 64 * 2)
valid = wins = maximum = 0
board = chess.Board(None)
with chess.syzygy.open_tablebase(args.syzygy) as oracle:
    for slot in range(24):
        pawn = chess.square(slot % 4, 1 + slot // 4)
        for white in range(64):
            for black in range(64):
                if white == black or pawn in (white, black) or chess.square_distance(white, black) <= 1:
                    continue
                board.clear()
                board.set_piece_at(pawn, chess.Piece(chess.PAWN, chess.WHITE))
                board.set_piece_at(white, chess.Piece(chess.KING, chess.WHITE))
                board.set_piece_at(black, chess.Piece(chess.KING, chess.BLACK))
                for turn in [chess.WHITE, chess.BLACK]:
                    board.turn = turn
                    if not board.is_valid():
                        continue
                    valid += 1
                    wdl = oracle.probe_wdl(board) * (1 if turn else -1)
                    index = ((slot * 64 + white) * 64 + black) * 2 + (0 if turn else 1)
                    if wdl == 0:
                        data[index] = 0
                    else:
                        if wdl != 2:
                            raise ValueError((board.fen(), wdl))
                        distance = abs(oracle.probe_dtz(board))
                        if distance >= 254:
                            raise ValueError((board.fen(), distance))
                        data[index] = distance + 1
                        wins += 1
                        maximum = max(maximum, distance)
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_bytes(data)
print(json.dumps({'file': args.output.name, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest(),
                  'positions': valid, 'winning': wins, 'maximum_reset_distance': maximum}, indent=2))
