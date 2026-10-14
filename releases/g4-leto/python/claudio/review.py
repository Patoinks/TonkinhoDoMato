"""Game review with Claudio's own calculation, like a player going over a game.

Every move gets a note from its material consequence:
    delta = forced material balance after the move - before it (mover's side),
where the forced balance = material on the board (Claudio's own piece values)
+ what the side to move wins by force within 5 half-moves (exact search,
captures settled; 3 half-moves found a third as many of the real blunders, 7 no
more than 5 - review_lab.py). A forced mate counts as +-MATE_PAWNS. delta >= +GOOD: good;
delta <= -GOOD: bad; otherwise neutral (nothing to learn from that move).

Consequences (delayed blame): a positional mistake usually costs material LATER.
Every change of the forced balance is also shared among the side's own earlier
moves within --horizon plies (more to the nearer ones), from that side's view. A
move's score = its immediate delta + its share of what followed; the label uses
the score, and its size is how much the move weighs in learning. Off by default
(horizon 0): it found a few more blunders but credited 14% of 'good' moves that were
really mistakes (review_lab.py).

The decisive moment: the move after which the loser's balance never came back
above -GOOD (and the winner's move after which it stayed ahead). Nothing here
knows anything about chess beyond the rules and Claudio's own values.

    python claudio/review.py --games claudio/results/groups-grupos-4.json --player "Este (treino 1550)" --limit 2
"""
import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import ROOT  # noqa: E402

import chess  # noqa: E402

MATE_PAWNS = 20.0
GOOD = 1.5
HORIZON = 20            # plies after a move in which later material changes are partly its doing
DECAY = 0.85            # nearer earlier moves take a larger share of a later change
VALUES = [0.0, 1.0, 2.13, 2.73, 3.98, 9.06]      # Claudio's own measured values (pawn = 1)


def forced_balance(board, fast, values, plies, qdepth, nodes):
    """Material from White's side after what the side to move wins by force (None if the search aborts)."""
    gain = fast.forced_gain(values, plies, qdepth, nodes)
    if gain != gain:
        return None
    if abs(gain) >= 900:
        gain = MATE_PAWNS if gain > 0 else -MATE_PAWNS
    material = sum(values[p.piece_type] * (1 if p.color == chess.WHITE else -1)
                   for p in board.piece_map().values() if p.piece_type != chess.KING)
    return material + (gain if board.turn == chess.WHITE else -gain)


def consequences(balances, sides, horizon=HORIZON, decay=DECAY):
    """Per ply: the share of later balance changes (within `horizon` plies) owed to that move,
    from its mover's side. A change is split among the side's own earlier moves, the nearer
    ones taking more (weights decay**distance, normalised)."""
    filled, last = [], 0.0
    for balance in balances:
        last = last if balance is None else balance
        filled.append(last)
    delayed = [0.0] * len(sides)
    for j in range(len(sides)):
        change = filled[j + 1] - filled[j]                         # White's view
        if abs(change) < 1e-9:
            continue
        for side, view in (("white", change), ("black", -change)):
            earlier = [t for t in range(max(0, j - horizon), j) if sides[t] == side]
            if not earlier:
                continue
            weights = [decay ** (j - t) for t in earlier]
            total = sum(weights)
            for t, weight in zip(earlier, weights):
                delayed[t] += view * weight / total
    return delayed


def net_balances(moves, net, start_fen=None):
    """Claudio's own intuition along the game, per position (index = ply): the network's value
    for White minus its value for Black, each with its own memory of the game exactly as when
    it plays that colour. In [-1, 1]; nothing here comes from outside Claudio."""
    import numpy as np
    import torch
    from claudio_agent import native_board
    from tonkai import fast_rules

    def value(rows, memory):
        contexts, _ = net.candidates(torch.from_numpy(np.asarray(rows, dtype=np.float32)), memory)
        return float(net.survival(contexts)[0].mean(-1)[0])

    sides = []
    with torch.inference_mode():
        for colour in (chess.WHITE, chess.BLACK):
            fast = native_board(chess.Board(start_fen) if start_fen else chess.Board())
            here = np.asarray(fast.encode_here(colour), dtype=np.float32)[None]
            memory = net.initial_memory(1)
            out = [value(here, memory)]
            memory = net.memory_cell(net.encode(torch.from_numpy(here)), memory)
            for uci in moves:
                code = fast_rules.code(chess.Move.from_uci(uci))
                out.append(value(fast_rules.encode_after_moves(fast, [code], colour), memory))
                fast.push(code)
                here = np.asarray(fast.encode_here(colour), dtype=np.float32)[None]
                memory = net.memory_cell(net.encode(torch.from_numpy(here)), memory)
            sides.append(out)
    return [white - black for white, black in zip(*sides)]


def review(moves, values=VALUES, plies=5, qdepth=5, nodes=2_000_000, good=GOOD, start_fen=None,
           horizon=0, decay=DECAY, extra=None):
    """Per move: side, san, balance before/after and delta (mover's side), label; plus the decisive moments.
    `extra` (optional, per position, White's side, in pawns) is added to the forced material balance,
    e.g. Claudio's own intuition from net_balances() times a weight."""
    from claudio_agent import native_board
    from tonkai import fast_rules
    board = chess.Board(start_fen) if start_fen else chess.Board()
    fast = native_board(board)

    def balance_now():
        balance = forced_balance(board, fast, values, plies, qdepth, nodes)
        return balance if balance is None or extra is None else balance + extra[len(balances)]

    balances = []
    balances.append(balance_now())
    notes = []
    for ply, uci in enumerate(moves):
        move = chess.Move.from_uci(uci)
        side, san = board.turn, board.san(move)
        board.push(move)
        fast.push(fast_rules.code(move))
        balances.append(balance_now())
        before, after = balances[-2], balances[-1]
        sign = 1 if side == chess.WHITE else -1
        delta = None if before is None or after is None else sign * (after - before)
        label = "neutral" if delta is None or abs(delta) < good else "good" if delta > 0 else "bad"
        if board.is_checkmate():
            delta, label = MATE_PAWNS, "good"                      # the mating move itself
        notes.append({"ply": ply, "move": (ply // 2) + 1, "side": "white" if side else "black", "san": san,
                      "uci": uci, "before": None if before is None else sign * before,
                      "after": None if after is None else sign * after, "delta": delta, "label": label})
    # What followed: later material changes, partly owed to the moves before them.
    delayed = consequences(balances, [n["side"] for n in notes], horizon, decay)
    for note, later in zip(notes, delayed):
        note["immediate_label"] = note["label"]
        note["delayed"] = later
        note["score"] = (note["delta"] or 0.0) + later
        if note["label"] != "good" or note["delta"] != MATE_PAWNS:
            note["label"] = "neutral" if abs(note["score"]) < good else "good" if note["score"] > 0 else "bad"
    outcome = board.outcome(claim_draw=True)
    winner = None if outcome is None or outcome.winner is None else ("white" if outcome.winner else "black")
    decisive = {}
    if winner is not None:
        loser = "black" if winner == "white" else "white"
        for who, sign_ok in ((loser, lambda b: b <= -good), (winner, lambda b: b >= good)):
            sign = 1 if who == "white" else -1
            seq = [None if b is None else sign * b for b in balances]
            start = len(seq) - 1
            while start > 0 and seq[start] is not None and sign_ok(seq[start]):
                start -= 1
            # The side's own move that turned the balance for good (or its most extreme move).
            own = [n for n in notes if n["side"] == who]
            later = [n for n in own if n["ply"] >= start - 1 and n["label"] == ("bad" if who == loser else "good")]
            pick = later[0] if later else (min(own, key=lambda n: n["delta"] or 0) if who == loser
                                           else max(own, key=lambda n: n["delta"] or 0)) if own else None
            decisive[who] = None if pick is None else {k: pick[k] for k in ("ply", "move", "san", "delta")}
    return {"notes": notes, "balances": balances, "winner": winner,
            "end": outcome.termination.name if outcome else "unfinished", "decisive": decisive}


def report(result, player_side=None, top=3):
    lines = [f"Resultado: {result['winner'] or 'empate'} ({result['end']}), {len(result['notes'])} meios-lances"]
    for side in ("white", "black"):
        own = [n for n in result["notes"] if n["side"] == side and n["delta"] is not None]
        mark = " (o jogador)" if side == player_side else ""
        best = sorted(own, key=lambda n: -n["score"])[:top]
        worst = sorted(own, key=lambda n: n["score"])[:top]
        lines.append(f"  {('Brancas' if side == 'white' else 'Pretas')}{mark}:")
        lines.append("    melhores: " + ", ".join(f"{n['move']}.{n['san']} ({n['score']:+.1f}: ja {n['delta']:+.1f}, depois {n['delayed']:+.1f})"
                                             for n in best if n["score"] > 0))
        lines.append("    piores:   " + ", ".join(f"{n['move']}.{n['san']} ({n['score']:+.1f}: ja {n['delta']:+.1f}, depois {n['delayed']:+.1f})"
                                             for n in worst if n["score"] < 0))
        counts = {k: sum(n["label"] == k for n in own) for k in ("good", "bad", "neutral")}
        lines.append(f"    bons {counts['good']}, maus {counts['bad']}, neutros {counts['neutral']}")
    for side, pick in result["decisive"].items():
        if pick:
            what = "o lance que ganhou" if side == result["winner"] else "o lance que perdeu"
            lines.append(f"  Decisivo ({'brancas' if side == 'white' else 'pretas'}): {what} foi "
                         f"{pick['move']}.{pick['san']} ({pick['delta']:+.1f})")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--games", type=Path, required=True, help="a results JSON (groups, engine or arena)")
    parser.add_argument("--player", help="only this player's games (groups JSON)")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--decided", action="store_true", help="only decided games")
    args = parser.parse_args()
    data = json.loads(args.games.read_text(encoding="utf-8"))
    games = data.get("list", data.get("games", []))
    shown = 0
    for g in games:
        moves = (g.get("moves") or g.get("uci") or "").split()
        if args.player and args.player not in (g.get("a"), g.get("b")):
            continue
        if args.decided and not g.get("winner"):
            continue
        side = None
        if args.player and "white" in g:
            side = "white" if g["white"] == args.player else "black"
        title = f"{g.get('a', '?')} vs {g.get('b', '?')}" if "a" in g else f"jogo {g.get('game')}"
        print(f"\n=== {title}, brancas: {g.get('white')}")
        print(report(review(moves), side))
        shown += 1
        if shown >= args.limit:
            break


if __name__ == "__main__":
    main()
