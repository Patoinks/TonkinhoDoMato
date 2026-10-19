"""Style metrics of the Pantheon (and neutral Leto) from their games, from the positions alone (no engine).

Five families, each the mean of its signals' z-scores across the players, shown 0-100 (lowest to highest player),
from the middlegame moves (16 pieces or more on the board):

- Agressividade: checks, new threats on enemy pieces, pressure on the enemy king's zone, pieces in the enemy half.
- Posicional: space (enemy-half squares it controls that enemy pawns cannot contest), mobility, centre, outposts.
- Defensivo: healthy pawns (no isolated or doubled), king shelter, defended pieces, retreats, draws.
- Truques: forks, new pins, discovered attacks, offers (a piece left where a cheaper one takes it), forcing moves.
- Caos: tension (captures available to both sides), imbalance (different piece types), hanging pieces, the
  opponent's number of moves, kings on opposite wings.

Extras: materialismo (takes when it can), simplificação (pieces off per 10 moves), iniciativa (forcing moves),
rei no final (centralised king with few pieces), game length, draws, score.

    python leto-rapido/estilo.py ../estilos
"""
import json
import math
import statistics as st
import sys
from pathlib import Path

import chess

V = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 100}
CENTRE = [chess.D4, chess.E4, chess.D5, chess.E5]
NAMES = {"leto": "Leto (neutro)", "nezha": "Nezha", "loki": "Loki", "anubis": "Anúbis", "hades": "Hades"}
FAMILIES = {
    "Agressividade": ["a_check", "a_threat", "a_kingzone", "a_advance"],
    "Posicional": ["p_space", "p_mobility", "p_centre", "p_outpost"],
    "Defensivo": ["d_structure", "d_shelter", "d_defended", "d_retreat", "d_draws"],
    "Truques": ["t_fork", "t_pin", "t_discovered", "t_offer", "t_forcing"],
    "Caos": ["c_tension", "c_imbalance", "c_hanging", "c_branching", "c_opposite"],
}


def zone(board, colour):
    k = board.king(colour)
    return chess.BB_KING_ATTACKS[k] | chess.BB_SQUARES[k] if k is not None else 0


def enemy_half(colour):
    return chess.BB_RANK_5 | chess.BB_RANK_6 | chess.BB_RANK_7 | chess.BB_RANK_8 if colour else \
        chess.BB_RANK_1 | chess.BB_RANK_2 | chess.BB_RANK_3 | chess.BB_RANK_4


def pawn_attacks(board, colour):
    out = 0
    for s in board.pieces(chess.PAWN, colour):
        out |= chess.BB_PAWN_ATTACKS[colour][s]
    return out


def cheapest_attacker(board, colour, square):
    a = [V[board.piece_type_at(s)] for s in board.attackers(colour, square)]
    return min(a) if a else None


def threatened(board, me, square):
    """An enemy piece on square that `me` attacks and that is undefended or worth more than its cheapest attacker."""
    p = board.piece_at(square)
    if p is None or p.color == me or p.piece_type == chess.KING:
        return False
    c = cheapest_attacker(board, me, square)
    return c is not None and (not board.attackers(not me, square) or V[p.piece_type] > c)


def threats(board, me):
    return {s for s in chess.scan_forward(board.occupied_co[not me]) if threatened(board, me, s)}


def pinned_count(board, colour):
    return sum(1 for s in chess.scan_forward(board.occupied_co[colour] & ~board.kings) if board.is_pinned(colour, s))


def mobility(board, colour):
    b = board.copy(stack=False)
    b.turn = colour
    b.ep_square = None
    return sum(1 for _ in b.generate_pseudo_legal_moves())


def captures_available(board, colour):
    b = board.copy(stack=False)
    b.turn = colour
    return sum(1 for m in b.generate_pseudo_legal_captures())


def move_signals(board, move, book):
    me = board.turn
    piece = board.piece_at(move.from_square)
    capture = board.is_capture(move)
    could_capture = any(True for _ in board.generate_legal_captures())
    check = board.gives_check(move)
    before_threats = threats(board, me)
    before_pins = pinned_count(board, not me)
    before_pieces = len(board.piece_map())
    after = board.copy(stack=False)
    after.push(move)
    after_threats = threats(after, me)
    new_threats = after_threats - before_threats
    to = move.to_square
    moved_value = V[(after.piece_type_at(to) or chess.PAWN)]
    # fork: the moved piece threatens two or more (or checks and threatens one)
    own_targets = {s for s in after_threats if to in after.attackers(me, s)}
    fork = len(own_targets) >= 2 or (check and len(own_targets) >= 1)
    # discovered: a new threat (or check) made by a piece other than the one that moved
    discovered = any(to not in after.attackers(me, s) for s in new_threats) or (
        check and to not in after.attackers(me, after.king(not me)) if after.king(not me) is not None else False)
    # offer: the moved piece can be taken by a cheaper piece, or taken undefended (not a capture of more)
    c = cheapest_attacker(after, not me, to)
    offer = c is not None and (c < moved_value or not after.attackers(me, to)) and piece.piece_type != chess.KING and \
        not (capture and V[board.piece_type_at(to) or chess.PAWN] >= moved_value)
    # pawn structure, shelter
    own_pawns = after.pieces(chess.PAWN, me)
    files = [chess.square_file(s) for s in own_pawns]
    isolated = sum(1 for f in files if (f - 1) not in files and (f + 1) not in files)
    doubled = len(files) - len(set(files))
    k = after.king(me)
    shelter = 0
    if k is not None:
        kf, kr = chess.square_file(k), chess.square_rank(k)
        for s in own_pawns:
            dr = chess.square_rank(s) - kr if me else kr - chess.square_rank(s)
            if abs(chess.square_file(s) - kf) <= 1 and 1 <= dr <= 2:
                shelter += 1
    own = [s for s in chess.scan_forward(after.occupied_co[me] & ~after.kings)]
    defended = sum(1 for s in own if after.attackers(me, s)) / max(1, len(own))
    rank_delta = chess.square_rank(to) - chess.square_rank(move.from_square)
    retreat = piece.piece_type not in (chess.KING, chess.PAWN) and (rank_delta < 0 if me else rank_delta > 0)
    enemy_pawn_cover = pawn_attacks(after, not me)
    controlled = 0
    for s in chess.scan_forward(enemy_half(me) & ~enemy_pawn_cover):
        if after.attackers(me, s):
            controlled += 1
    outposts = sum(1 for s in chess.scan_forward((after.pieces_mask(chess.KNIGHT, me) | after.pieces_mask(chess.BISHOP, me))
                                                 & enemy_half(me)) if not (chess.BB_SQUARES[s] & enemy_pawn_cover))
    centre = sum(1 for s in CENTRE if after.attackers(me, s) or (after.color_at(s) == me))
    imbalance = sum(abs(len(after.pieces(t, chess.WHITE)) - len(after.pieces(t, chess.BLACK)))
                    for t in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN))
    hanging = sum(1 for colour in (chess.WHITE, chess.BLACK)
                  for s in chess.scan_forward(after.occupied_co[colour] & ~after.kings)
                  if after.attackers(not colour, s) and not after.attackers(colour, s))
    ek = after.king(not me)
    few = len(after.piece_map()) <= 10
    king_centre = (3.5 - max(abs(chess.square_file(k) - 3.5), abs(chess.square_rank(k) - 3.5))) if (few and k is not None) else None
    return {
        "a_check": float(check), "a_threat": float(len(new_threats)),
        "a_kingzone": float(sum(1 for s in chess.scan_forward(zone(after, not me)) if after.attackers(me, s))),
        "a_advance": float(sum(1 for s in chess.scan_forward(after.occupied_co[me] & ~after.pawns & ~after.kings & enemy_half(me)))),
        "p_space": float(controlled), "p_mobility": float(mobility(after, me)), "p_centre": float(centre),
        "p_outpost": float(outposts),
        "d_structure": -float(isolated + doubled), "d_shelter": float(shelter), "d_defended": defended,
        "d_retreat": float(retreat),
        "t_fork": float(fork), "t_pin": float(max(0, pinned_count(after, not me) - before_pins)),
        "t_discovered": float(discovered), "t_offer": float(offer),
        "t_forcing": float(check or capture or bool(new_threats)),
        "c_tension": float(captures_available(after, me) + captures_available(after, not me)),
        "c_imbalance": float(imbalance), "c_hanging": float(hanging),
        "c_branching": float(after.legal_moves.count()),
        "x_greed": float(capture) if could_capture else None,
        "x_traded": float(before_pieces - len(after.piece_map())),
        "x_king_centre": king_centre,
        "_opposite": (k is not None and ek is not None and abs(chess.square_file(k) - chess.square_file(ek)) >= 4),
    }


PHASE = None   # (first, last) half-move of the game to count, e.g. (0, 19) for the opening; None: middlegame


def player(folder, name):
    data = json.loads((folder / f"{name}.json").read_text(encoding="utf-8"))
    games = data["games"]
    acc, results, lengths, opposite = {}, [], [], []
    for g in games:
        white = g["leto_white"]
        board = chess.Board()
        moves = [chess.Move.from_uci(u) for u in g["moves"].split()]
        opp = []
        in_book = True
        for i, m in enumerate(moves):
            if (board.turn == chess.WHITE) == white:
                sig = move_signals(board, m, in_book and i < 30)
                middle = (len(board.piece_map()) >= 16 if PHASE is None   # the families compare middlegame moves:
                          else PHASE[0] <= i <= PHASE[1])                 # game
                opposite.append(sig.pop("_opposite"))      # lengths differ, and the endgame has less of everything
                for k, v in sig.items():
                    if v is not None and (middle or k == "x_king_centre"):
                        acc.setdefault(k, []).append(v)
            board.push(m)
        results.append(g["score"])
        lengths.append(len(moves))
        acc.setdefault("d_draws", []).append(float(g["score"] == 0.5))
    out = {k: st.mean(v) for k, v in acc.items()}
    out["c_opposite"] = sum(opposite) / max(1, len(opposite))
    out["score"], out["plies"], out["games"] = st.mean(results), st.mean(lengths), len(games)
    out["x_traded"] *= 10
    return out


def main():
    folder = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "estilos")
    names = [n for n in NAMES if (folder / f"{n}.json").exists()]
    stats = {n: player(folder, n) for n in names}
    signals = [s for fam in FAMILIES.values() for s in fam]
    z = {}
    for s in signals:
        vals = [stats[n][s] for n in names]
        mu, sd = st.mean(vals), (st.pstdev(vals) or 1.0)
        for n in names:
            z[(n, s)] = (stats[n][s] - mu) / sd
    fam_raw = {(n, f): st.mean(z[(n, s)] for s in sig) for n in names for f, sig in FAMILIES.items()}
    lines = ["# Estilo dos deuses\n", f"\n{len(names)} jogadores, jogos contra o Stockfish 2250 (2 ms). Índice 0-100: o mais baixo "
             "e o mais alto dos jogadores em cada família (média dos z-scores dos sinais).\n\n"]
    head = "| jogador | score | " + " | ".join(FAMILIES) + " |\n| --- | ---: |" + " ---: |" * len(FAMILIES) + "\n"
    lines.append(head)
    for n in names:
        row = []
        for f in FAMILIES:
            vals = [fam_raw[(m, f)] for m in names]
            lo, hi = min(vals), max(vals)
            row.append(f"{100 * (fam_raw[(n, f)] - lo) / ((hi - lo) or 1):.0f}")
        lines.append(f"| {NAMES[n]} | {stats[n]['score']:.2f} | " + " | ".join(row) + " |\n")
    lines.append("\n## Extras\n\n| jogador | materialismo | simplificação (peças/10 lances) | iniciativa | rei no final | "
                 "meios-lances | empates |\n| --- | ---: | ---: | ---: | ---: | ---: | ---: |\n")
    for n in names:
        s = stats[n]
        lines.append(f"| {NAMES[n]} | {s.get('x_greed', 0):.0%} | {s['x_traded']:.2f} | {s['t_forcing']:.0%} | "
                     f"{s.get('x_king_centre', float('nan')):.2f} | {s['plies']:.0f} | {s['d_draws']:.0%} |\n")
    lines.append("\n## Sinais (médias por lance)\n\n| jogador | " + " | ".join(signals) + " |\n| --- |" + " ---: |" * len(signals) + "\n")
    for n in names:
        lines.append(f"| {NAMES[n]} | " + " | ".join(f"{stats[n][s]:.3g}" for s in signals) + " |\n")
    text = "".join(lines)
    (folder / "ESTILO.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
