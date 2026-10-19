"""Checks every repertoire line of repertorios_deuses.py: legality (first illegal move reported), conflicts
(the same position with two different own moves) and counts."""
import sys
from pathlib import Path

import chess

sys.path.insert(0, str(Path(__file__).resolve().parent))
from repertorios_deuses import OPENINGS  # noqa: E402

bad = 0
for name, text in OPENINGS.items():
    own, conflicts, lines = {}, 0, 0
    for raw in text.strip().splitlines():
        side, _, line = raw.strip().partition(":")
        colour = chess.WHITE if side.strip() == "W" else chess.BLACK
        board = chess.Board()
        lines += 1
        for i, san in enumerate(line.split()):
            try:
                move = board.parse_san(san)
            except ValueError as error:
                bad += 1
                print(f"{name} {side}: ilegal no lance {i + 1} '{san}' em: {' '.join(line.split()[:i + 1])}  ({type(error).__name__})")
                break
            if board.turn == colour:
                key = (colour, board._transposition_key())
                if key in own and own[key] != move:
                    conflicts += 1
                    print(f"{name} {side}: conflito em {' '.join(line.split()[:i])}: {board.san(own[key])} vs {san}")
                own.setdefault(key, move)
            board.push(move)
    print(f"== {name}: {lines} linhas, {len(own)} posições próprias, {conflicts} conflitos")
print("ilegais:", bad)
