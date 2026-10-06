"""Which version of Claudio's own review finds the real mistakes? Stockfish only JUDGES
(once, cached); every variant is Claudio's own:
    pN   its forced material search N half-moves deep
    hN   consequences over the next N half-moves (h0 = the immediate effect only)
    nW   plus its own intuition along the game (the network's value for White minus for
         Black, net_balances) times W pawns
Nothing is learned from Stockfish. The judge compares the played move with the best one
from the same position, and also says whether the difference shows up as material in
its lines (10 half-moves, 1-3-3-5-9: judge's side only) - the ceiling of a material review.

    python claudio/review_lab.py --games claudio/results/groups-grupos-4.json --count 30
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
from multiprocessing import get_context
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import ROOT, STOCKFISH  # noqa: E402

JUDGE_PIECES = {1: 1, 2: 3, 3: 3, 4: 5, 5: 9}
VARIANT = re.compile(r"p(\d+)h(\d+)(?:n(\d+(?:\.\d+)?))?$")
DEFAULT = ["p3h0", "p3h20", "p3h40", "p5h0", "p5h20", "p7h20", "p3h20n10", "p3h20n30", "p3h20n60", "p5h20n30"]
_w = {}


def _below_normal():
    if sys.platform == "win32":
        import ctypes
        kernel = ctypes.windll.kernel32
        kernel.SetPriorityClass(kernel.GetCurrentProcess(), 0x4000)


def win_chance(cp):
    return 1.0 / (1.0 + math.exp(-0.00368208 * cp))


# ---- the judge (cached) ----------------------------------------------------------------
def _init_judge(depth):
    _below_normal()
    import chess.engine
    _w["engine"] = chess.engine.SimpleEngine.popen_uci(str(STOCKFISH))
    _w["engine"].configure({"Threads": 1, "Hash": 64})
    _w["depth"] = depth


def _cp(score, colour):
    return max(-1500, min(1500, score.pov(colour).score(mate_score=1500)))


def _line_material(board, line, me, plies=10):
    board = board.copy(stack=False)
    for move in line[:plies]:
        board.push(move)
    if board.is_checkmate():
        return -100 if board.turn == me else 100
    return sum(JUDGE_PIECES.get(p.piece_type, 0) * (1 if p.color == me else -1) for p in board.piece_map().values())


def judge_game(moves):
    import chess
    import chess.engine
    board, rows, limit = chess.Board(), [], chess.engine.Limit(depth=_w["depth"])
    for uci in moves:
        move, me = chess.Move.from_uci(uci), board.turn
        info = _w["engine"].analyse(board, limit)
        best, best_line = _cp(info["score"], me), info.get("pv", [])
        if best_line and best_line[0] == move:
            played, played_line = best, best_line
        else:
            other = _w["engine"].analyse(board, limit, root_moves=[move])
            played, played_line = _cp(other["score"], me), other.get("pv", [move])
        rows.append({"best": best, "played": played, "loss": max(0, best - played),
                     "chance_loss": max(0.0, win_chance(best) - win_chance(played)),
                     "material_gap": _line_material(board, best_line, me) - _line_material(board, played_line, me)})
        board.push(move)
    return rows


# ---- Claudio's review variants ---------------------------------------------------------
def _init_review(checkpoint):
    _below_normal()
    import torch
    torch.set_num_threads(1)
    from claudio_load import load_claudio
    _w["net"] = load_claudio(checkpoint)[0]


def review_game(moves, variants):
    from review import review, net_balances
    intuition, out = None, {}
    for name in variants:
        plies, horizon, weight = VARIANT.match(name).groups()
        extra = None
        if weight:
            intuition = intuition if intuition is not None else net_balances(moves, _w["net"])
            extra = [float(weight) * x for x in intuition]
        rv = review(moves, plies=int(plies), horizon=int(horizon), extra=extra)
        out[name] = [n["label"] for n in rv["notes"]]
    return out


def _pool(workers, init, args):
    return ProcessPoolExecutor(workers, mp_context=get_context("spawn"), initializer=init, initargs=args)


def _close(pool):
    for process in list(getattr(pool, "_processes", {}).values()):
        process.kill()                        # the engines' threads would keep the workers alive
    pool.shutdown(wait=False, cancel_futures=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=Path, required=True)
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--depth", type=int, default=10)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--checkpoint", default=str(ROOT / "claudio/claudio-grande-g0.pt"))
    parser.add_argument("--variants", nargs="+", default=DEFAULT)
    args = parser.parse_args()
    _below_normal()
    games = [g["moves"].split() for g in json.loads(args.games.read_text(encoding="utf-8"))["list"]][:args.count]
    cache = ROOT / "claudio/results" / f"judge-cache-{args.games.stem}-d{args.depth}.json"
    judged = json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else []
    if len(judged) < len(games):
        pool = _pool(args.workers, _init_judge, (args.depth,))
        judged += list(pool.map(judge_game, games[len(judged):]))
        _close(pool)
        cache.write_text(json.dumps(judged), encoding="utf-8")
    judged = judged[:len(games)]
    pool = _pool(args.workers, _init_review, (args.checkpoint,))
    reviews = list(pool.map(review_game, games, [args.variants] * len(games)))
    _close(pool)

    rows = [dict(j, **{v: r[v][i] for v in args.variants}) for js, r in zip(judged, reviews) for i, j in enumerate(js)]
    groups = {
        "asneiras (>=2 peoes)": [r for r in rows if r["loss"] >= 200],
        "  - que se veem no material": [r for r in rows if r["loss"] >= 200 and r["material_gap"] >= 2],
        "  - posicionais": [r for r in rows if r["loss"] >= 200 and r["material_gap"] < 2],
        "mudam o jogo (>=20% hipotese)": [r for r in rows if r["chance_loss"] >= 0.20],
        "erros (1-2 peoes)": [r for r in rows if 100 <= r["loss"] < 200],
    }
    fine = [r for r in rows if r["loss"] < 50]
    print(f"{len(games)} jogos, {len(rows)} lances (juiz: mesma posicao, profundidade {args.depth})")
    for name, part in groups.items():
        print(f"  {name:32s} {len(part):5d}")
    header = "".join(f"{v:>10s}" for v in args.variants)
    print(f"\n{'apanha (marcados maus)':34s}{header}")
    table = {}
    for name, part in groups.items():
        cells = [100 * sum(r[v] == "bad" for r in part) / max(1, len(part)) for v in args.variants]
        table[name] = cells
        print(f"{name:34s}" + "".join(f"{c:9.0f}%" for c in cells))
    extra = {
        "falsos alarmes (bons marcados maus)": [100 * sum(r[v] == "bad" for r in fine) / max(1, len(fine)) for v in args.variants],
        "maus que sao mesmo erros (>=1 peao)": [100 * sum(r[v] == "bad" and r["loss"] >= 100 for r in rows)
                                                / max(1, sum(r[v] == "bad" for r in rows)) for v in args.variants],
        "bons que sao mesmo bons (<0.5 peao)": [100 * sum(r[v] == "good" and r["loss"] < 50 for r in rows)
                                                / max(1, sum(r[v] == "good" for r in rows)) for v in args.variants],
        "lances marcados (bons+maus)": [100 * sum(r[v] != "neutral" for r in rows) / len(rows) for v in args.variants],
    }
    print()
    for name, cells in extra.items():
        table[name] = cells
        print(f"{name:34s}" + "".join(f"{c:9.1f}%" for c in cells))
    out = ROOT / "claudio/results" / f"review-lab-{args.games.stem}.json"
    out.write_text(json.dumps({"variants": args.variants, "table": table, "games": len(games), "moves": len(rows)},
                              indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
