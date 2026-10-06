"""Stockfish as a JUDGE only (Claudio never learns from it): on a fixed set of real
positions (each with its game history, so Claudio's memory is exact), which move
does each Claudio version choose, and how much does that move lose against the
best one (centipawns, Stockfish depth --depth)? Same positions for every version,
so differences are not luck of the games.

    python claudio/judge_positions.py --models g0=claudio/claudio-grande-g0.pt b1000=claudio/train-brancas/claudio-grande-1000.pt
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from multiprocessing import get_context
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import ROOT, STOCKFISH  # noqa: E402

SOURCES = [ROOT / "claudio/results/groups-grupos-4.json", ROOT / "claudio/results/engine-grande-claudio-grande-g0-sf1800.json"]
_j = {}


def positions(count, seed):
    """(history of uci moves, ply) pairs from real games, spread over openings, middlegames and endings."""
    import numpy as np
    rng = np.random.default_rng(seed)
    games = []
    for source in SOURCES:
        data = json.loads(source.read_text(encoding="utf-8"))
        games += [g["moves"].split() for g in data["list"] if len(g["moves"].split()) > 20]
    picks = []
    while len(picks) < count:
        moves = games[int(rng.integers(len(games)))]
        ply = int(rng.integers(6, min(len(moves), 140)))
        picks.append(moves[:ply])
    return picks


def _init(models, depth):
    import torch
    torch.set_num_threads(1)
    import chess.engine
    from claudio_load import load_players, make_player
    _j["players"] = {name: load_players(spec) for name, spec in models.items()}
    _j["make"] = make_player
    _j["engine"] = chess.engine.SimpleEngine.popen_uci(str(STOCKFISH))
    _j["engine"].configure({"Threads": 1, "Hash": 64})
    _j["depth"] = depth


def _score(board, colour, move=None):
    """Worth of the position for `colour`; with `move`, of that move searched from this same position
    (comparing with the position after the move would count the side-to-move bonus as a loss)."""
    import chess.engine
    info = _j["engine"].analyse(board, chess.engine.Limit(depth=_j["depth"]), root_moves=None if move is None else [move])
    return max(-1500, min(1500, info["score"].pov(colour).score(mate_score=1500)))


def judge(history):
    import chess
    import numpy as np
    board = chess.Board()
    for uci in history:
        board.push_uci(uci)
    if board.is_game_over():
        return None
    me = board.turn
    best = _score(board, me)
    out = {"ply": len(history), "pieces": chess.popcount(board.occupied), "best": best, "moves": {}, "loss": {}}
    seen = {}
    for name, loaded in _j["players"].items():
        player = _j["make"](loaded, np.random.default_rng(0))
        move = player.choose(board)
        if move not in seen:
            seen[move] = _score(board, me, move)
        out["moves"][name] = move.uci()
        out["loss"][name] = max(0, best - seen[move])
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", required=True, help="name=checkpoint (or white.pt+black.pt)")
    parser.add_argument("--positions", type=int, default=300)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    models = dict(item.split("=", 1) for item in args.models)
    picks = positions(args.positions, args.seed)
    pool = ProcessPoolExecutor(args.workers, mp_context=get_context("spawn"), initializer=_init,
                               initargs=(models, args.depth))
    rows = [r for r in pool.map(judge, picks, chunksize=4) if r is not None]
    for process in list(getattr(pool, "_processes", {}).values()):
        process.kill()                        # the engines' threads would keep the workers alive
    pool.shutdown(wait=False, cancel_futures=True)
    first = next(iter(models))
    summary = {}
    for name in models:
        losses = [r["loss"][name] for r in rows]
        summary[name] = {"mean_loss_cp": sum(losses) / len(losses),
                         "blunders_200": sum(x >= 200 for x in losses) / len(losses),
                         "mistakes_100": sum(x >= 100 for x in losses) / len(losses),
                         "same_move_as_" + first: sum(r["moves"][name] == r["moves"][first] for r in rows) / len(rows)}
    for phase, test in (("abertura (<=20 meios-lances)", lambda r: r["ply"] <= 20),
                        ("meio-jogo", lambda r: r["ply"] > 20 and r["pieces"] > 12),
                        ("final (<=12 pecas)", lambda r: r["ply"] > 20 and r["pieces"] <= 12)):
        part = [r for r in rows if test(r)]
        for name in models:
            summary[name][phase] = sum(r["loss"][name] for r in part) / max(1, len(part))
    out = ROOT / "claudio/results" / f"judge-{args.label}.json"
    out.write_text(json.dumps({"models": models, "summary": summary, "rows": rows}, indent=1), encoding="utf-8")
    print(f"{len(rows)} posicoes (Stockfish so como juiz, profundidade {args.depth})")
    for name, s in summary.items():
        print(f"  {name:10s} perda media {s['mean_loss_cp']:5.0f} cp | erros>=100cp {100 * s['mistakes_100']:4.1f}% | "
              f"asneiras>=200cp {100 * s['blunders_200']:4.1f}% | igual a {first}: {100 * s['same_move_as_' + first]:4.1f}% | "
              f"abertura {s['abertura (<=20 meios-lances)']:.0f} meio {s['meio-jogo']:.0f} final {s['final (<=12 pecas)']:.0f}")


if __name__ == "__main__":
    main()
