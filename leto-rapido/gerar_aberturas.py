"""EXPERIMENT (not adopted, 2026-10-19): opening repertoires where every own move is the style's choice.

The per-move style signals are too crude for the first moves: every personality chose 1.Nf3, and the
repertoires came out nearly identical. Kept as a tool.


At each of its turns Leto (the fast champion) scores every legal move; among the sound ones - within the
personality's window of the best and losing no material by Leto's exchange count - the personality plays the
one its style likes most (personalidades.features x its weights; ties by Leto's score). Where the theory
(abertura.LINES) has moves, only those are candidates (Leto vetoes only clear mistakes). The opponent's replies
are the theory's (abertura.LINES, most played first), or Leto's best where the theory ends: 3 to the first
move, 2 to the second, then 1. Lines are 14 half-moves long.

    python leto-rapido/gerar_aberturas.py            # prints the OPENINGS text and writes aberturas_geradas.json
"""
import collections
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import chess
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402,F401
import abertura  # noqa: E402
import personalidades as P  # noqa: E402
import torneio  # noqa: E402
from claudio_agent import Claudio, ClaudioConfig  # noqa: E402
from claudio_load import load_claudio  # noqa: E402

PLIES = 14
BRANCH = (3, 2)          # opponent replies at its first and second turns, then 1
FIRST_FOR_BLACK = ["e4", "d4", "c4", "Nf3"]

record = {}
P.apply_style = lambda personality, board, moves, scores, known, nets: record.update(
    last=(list(moves), np.array(scores, dtype=float), None if nets is None else np.array(nets, dtype=float))) or False
real_opening_move = P.opening_move
P.opening_move = lambda name, board: None

net, values, config, _, king = load_claudio(str(torneio.CHECKPOINT), with_config=True, with_king=True)
cfg = ClaudioConfig.with_settings(["syzygy="] + torneio.RAPIDO + ["opening_book=False"], base=config)


def leto_scores(board):
    """Leto's moves of this position, best first, with their scores and exchange counts."""
    player = Claudio(net, values, cfg, rng=np.random.default_rng(1), king_model=king)
    player._owner = 1
    player.personality = SimpleNamespace(name="gerador", stats={})
    record.clear()
    best = player.choose(board.copy())
    if "last" not in record:                     # forced or decided before the scores (a mate, one move)
        return [(best, 1.0, 0.0)]
    moves, scores, nets = record["last"]
    order = np.argsort(-scores)
    return [(moves[i], float(scores[i]), 0.0 if nets is None else float(nets[i])) for i in order]


THEORY = collections.defaultdict(collections.Counter)
for text in abertura.LINES.strip().splitlines():
    b = chess.Board()
    for san in text.split():
        m = b.parse_san(san)
        THEORY[b._transposition_key()][m] += 1
        b.push(m)


def style(name, board, move):
    f = P.features(board, move)
    return sum(w * f.get(k, 0.0) for k, w in P.PERSONALITIES[name].items())


THEORY_SLACK = 0.25     # a theory move stays a candidate unless Leto rates it this far below its best


def own_move(name, board, window):
    """In theory: the style's favourite among the theory moves (Leto vetoes only clear mistakes). Out of
    theory: among Leto's moves within the personality's window, as it plays in a game."""
    scored = leto_scores(board)
    top = scored[0][1]
    counts = THEORY[board._transposition_key()]
    theory = {m for m, c in counts.items() if c >= 2}   # one-off lines in abertura.LINES (2.Rg1...) are not theory
    known = [(m, s) for m, s, n in scored if m in theory and n >= -0.25 and s >= top - THEORY_SLACK]
    sound = known or [(m, s) for m, s, n in scored if s >= top - window and n >= -0.25] or [scored[0][:2]]
    return max(sound, key=lambda ms: (style(name, board, ms[0]), ms[1]))[0], len(sound)


def replies(board, k):
    theory = [m for m, _ in THEORY[board._transposition_key()].most_common(k)]
    if len(theory) < k:
        theory += [m for m, _, _ in leto_scores(board) if m not in theory][:k - len(theory)]
    return theory[:k]


def lines_for(name, colour):
    window = P.WINDOWS.get(name, 0.06)
    out, choices = [], []

    def walk(board, opp_turns):
        if len(board.move_stack) >= PLIES or board.is_game_over():
            out.append(board.copy())
            return
        if board.turn == colour:
            move, n = own_move(name, board, window)
            choices.append(n)
            board.push(move)
            walk(board, opp_turns)
            board.pop()
        else:
            k = BRANCH[opp_turns] if opp_turns < len(BRANCH) else 1
            for m in replies(board, k):
                board.push(m)
                walk(board, opp_turns + 1)
                board.pop()

    if colour == chess.WHITE:
        walk(chess.Board(), 0)
    else:
        for first in FIRST_FOR_BLACK:
            b = chess.Board()
            b.push_san(first)
            walk(b, 1)
    texts = []
    for b in out:
        sans, r = [], chess.Board()
        for m in b.move_stack:
            sans.append(r.san(m))
            r.push(m)
        texts.append(("W: " if colour == chess.WHITE else "B: ") + " ".join(sans))
    return texts, choices


def main():
    result = {}
    for name in P.PERSONALITIES:
        white, cw = lines_for(name, chess.WHITE)
        black, cb = lines_for(name, chess.BLACK)
        result[name] = white + black
        print(f'    "{name}": """   # {len(white)} W + {len(black)} B lines; sound choices per move: '
              f'{np.mean(cw + cb):.1f}')
        for line in white + black:
            print("        " + line)
        print('    """,')
    (Path(__file__).resolve().parent / "aberturas_geradas.json").write_text(json.dumps(result, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
