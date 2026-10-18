"""How often a move's scan value is the same two plies later (the idea: rescan only what the last moves touched)."""
import json, sys
sys.path.insert(0, "."); import paths  # noqa
import chess
from claudio_agent import native_board
values = [0.0, 1.0, 2.1305672037676793, 2.7274115583819394, 3.981036451607642, 9.061081716147656]
def scan(b):
    f = native_board(b); codes = list(f.legal_moves())
    out = f.deep_after_moves(codes, values, 3, 5, 559745, False, False)
    return {c: round(v, 6) for c, (v, _) in zip(codes, out)}
def dist(a, b): return max(abs(a % 8 - b % 8), abs(a // 8 - b // 8))
games = json.load(open(sys.argv[1], encoding="utf-8"))["games"][:int(sys.argv[2])]
tot = same = far_tot = far_same = changed_pos = all_same_pos = 0
for g in games:
    moves = [chess.Move.from_uci(u) for u in g["moves"].split()]
    b = chess.Board(); last = {}
    for i, m in enumerate(moves):
        if i >= 8 and not b.is_game_over():
            now = scan(b); prev = last.get(b.turn)
            if prev is not None:
                touched = {x for mv in moves[i - 2:i] for x in (mv.from_square, mv.to_square)}
                n = s = 0
                for c, v in now.items():
                    if c in prev:
                        tot += 1; n += 1; ok = prev[c] == v; same += ok; s += ok
                        fr, to = c & 63, (c >> 6) & 63
                        if min(dist(fr, t) for t in touched) > 2 and min(dist(to, t) for t in touched) > 2:
                            far_tot += 1; far_same += ok
                changed_pos += 1; all_same_pos += (n == s)
            last[b.turn] = now
        b.push(m)
print(f"lances comparados {tot}: valor igual {same / tot:.1%}")
print(f"lances longe das peças mexidas (>2 casas) {far_tot}: valor igual {far_same / far_tot:.1%}")
print(f"posições em que TODOS os lances ficaram iguais: {all_same_pos}/{changed_pos}")
