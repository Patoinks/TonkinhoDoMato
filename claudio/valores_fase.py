"""Claudio's piece values by phase of the game, measured in its own games - the owner's rule
(2026-09-23) applied per phase: a piece's score is 3 x its K/D in self-play games (both sides),
where a kill counts the victim's level in pawn units, the levels settled by repetition (the
same computation as tonkai.captures.CaptureStats.piece_scores), relative to the pawn.

Phase of a capture (the position it is played in): opening up to --opening-plies half-moves of
the game, ending at --ending-pieces or fewer pieces on the board (kings and pawns included),
middlegame otherwise. Games: Claudio against Claudio only (mutation tournaments, group stages).

    python claudio/valores_fase.py --games claudio/results/torneio-p3.jsonl claudio/results/grupos-*.json --label p3

Prints the values and writes results/valores-fase-<label>.json, with the setting for Claudio:
phase_values="P,N,B,R,Q;P,N,B,R,Q;P,N,B,R,Q" (opening; middlegame; ending).
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import glob
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import ROOT  # noqa: E402

PHASES = ("opening", "middlegame", "ending")
NAMES = ("P", "N", "B", "R", "Q")


def captures(job):
    """Counter of (phase, capturer type, victim type) for one game's moves."""
    import chess
    moves, opening_plies, ending_pieces = job
    board, out = chess.Board(), Counter()
    for ply, uci in enumerate(moves.split()):
        move = chess.Move.from_uci(uci)
        if board.is_capture(move):
            victim = chess.PAWN if board.is_en_passant(move) else board.piece_type_at(move.to_square)
            capturer = board.piece_type_at(move.from_square)
            phase = ("opening" if ply < opening_plies else
                     "ending" if chess.popcount(board.occupied) <= ending_pieces else "middlegame")
            if victim != chess.KING:
                out[(phase, capturer, victim)] += 1
        board.push(move)
    return out


def scores(pairs, min_deaths):
    """{type: 3 x level-weighted K/D} (types 1..5), or None without enough deaths. pairs[(killer, victim)]."""
    deaths = {k: sum(c for (a, v), c in pairs.items() if v == k) for k in range(1, 6)}
    if min(deaths.values()) < min_deaths:
        return None, deaths
    kills = {k: sum(c for (a, v), c in pairs.items() if a == k) for k in range(1, 6)}
    if not kills[1]:
        return None, deaths

    def level_kills(killer, level):
        return sum(c * level[v] for (a, v), c in pairs.items() if a == killer)
    level = {k: (kills[k] / deaths[k]) / (kills[1] / deaths[1]) for k in range(1, 6)}
    for _ in range(200):
        weighted = {k: level_kills(k, level) / deaths[k] for k in range(1, 6)}
        new = {k: weighted[k] / weighted[1] for k in range(1, 6)}
        settled = max(abs(new[k] - level[k]) for k in level) < 1e-12
        level = new
        if settled:
            break
    return {k: 3.0 * level_kills(k, level) / deaths[k] for k in range(1, 6)}, deaths


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--games", nargs="+", required=True, help="jsonl/json game logs (each game has 'moves')")
    parser.add_argument("--opening-plies", type=int, default=20)
    parser.add_argument("--ending-pieces", type=int, default=12)
    parser.add_argument("--min-deaths", type=int, default=50)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()

    games = []
    for pattern in args.games:
        for path in sorted(glob.glob(pattern)):
            text = Path(path).read_text(encoding="utf-8")
            rows = ([json.loads(line) for line in text.splitlines() if line.strip()] if path.endswith(".jsonl")
                    else json.loads(text).get("list", []))
            games += [row["moves"] for row in rows if isinstance(row.get("moves"), str) and row["moves"]]
    total = Counter()
    with ProcessPoolExecutor(args.workers) as pool:
        for counter in pool.map(captures, [(m, args.opening_plies, args.ending_pieces) for m in games], chunksize=200):
            total.update(counter)
    report = {"games": len(games), "opening_plies": args.opening_plies, "ending_pieces": args.ending_pieces,
              "phases": {}}
    print(f"{len(games)} jogos Claudio contra Claudio")
    rows = []
    for phase in PHASES + ("all",):
        pairs = Counter()
        for (p, a, v), c in total.items():
            if phase in ("all", p):
                pairs[(a, v)] += c
        found, deaths = scores(pairs, args.min_deaths)
        values = None if found is None else {k: found[k] / found[1] for k in range(1, 6)}
        report["phases"][phase] = {"values": values, "deaths": deaths, "captures": sum(pairs.values())}
        shown = "sem mortes suficientes" if values is None else "  ".join(f"{NAMES[k - 1]} {values[k]:.2f}"
                                                                      for k in range(1, 6))
        print(f"  {phase:10s} {shown}   (capturas {sum(pairs.values())}, mortes {[deaths[k] for k in range(1, 6)]})")
        if phase in PHASES and values:
            rows.append(",".join(f"{values[k]:.4f}" for k in range(1, 6)))
    if len(rows) == 3:
        report["setting"] = "phase_values=" + ";".join(rows)
        print(report["setting"])
    out = ROOT / "claudio/results" / f"valores-fase-{args.label}.json"
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
