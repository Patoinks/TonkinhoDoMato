"""Do the personalities' openings play their own style? Every move of each repertoire is ranked among all the
legal moves of its position by each personality's style (personalidades.features x weights): 1.0 = the move
that style likes most. A repertoire matches its personality when its own style ranks its moves highest.

    python leto-rapido/aberturas_estilo.py
"""
import statistics as st
import sys
from pathlib import Path

import chess

sys.path.insert(0, str(Path(__file__).resolve().parent))
import personalidades as P  # noqa: E402


def style(name, board, move):
    f = P.features(board, move)
    return sum(w * f.get(k, 0.0) for k, w in P.PERSONALITIES[name].items())


def percentile(name, board, move):
    scores = [(style(name, board, m), m) for m in board.legal_moves]
    mine = style(name, board, move)
    worse = sum(1 for s, _ in scores if s < mine)
    ties = sum(1 for s, _ in scores if s == mine) - 1
    return (worse + 0.5 * ties) / max(1, len(scores) - 1)


def book_positions(name):
    out = []
    for text in P.OPENINGS[name].strip().splitlines():
        side, _, line = text.strip().partition(":")
        colour = chess.WHITE if side.strip() == "W" else chess.BLACK
        board = chess.Board()
        for san in line.split():
            move = board.parse_san(san)
            if board.turn == colour:
                out.append((board.copy(), move, f"{side.strip()}: {' '.join(line.split()[:line.split().index(san) + 1])}"))
            board.push(move)
    return out


def main():
    names = list(P.PERSONALITIES)
    print("| repertório \ estilo | " + " | ".join(names) + " |\n| --- |" + " ---: |" * len(names))
    weak = {}
    for book in names:
        positions = book_positions(book)
        row = []
        for s in names:
            ps = [percentile(s, b, m) for b, m, _ in positions]
            row.append(st.mean(ps))
            if s == book:
                weak[book] = sorted(((p, label) for p, (_, _, label) in zip(ps, positions)), key=lambda x: x[0])
        best = max(row)
        print(f"| {book} | " + " | ".join(f"**{v:.2f}**" if v == best else f"{v:.2f}" for v in row) + " |")
    print()
    for book in names:
        bad = [(p, label) for p, label in weak[book] if p < 0.5]
        print(f"{book}: {len(bad)}/{len(weak[book])} lances abaixo do meio do seu próprio estilo")
        for p, label in bad:
            print(f"   {p:.2f}  {label}")


if __name__ == "__main__":
    main()
