"""Does Claudio's own review find the real blunders? Stockfish only JUDGES the moves
actually played (depth --depth); the review (immediate material only, and immediate +
consequences) is compared with it. Nothing is learned from Stockfish.

    python claudio/review_vs_judge.py --games claudio/results/groups-grupos-4.json --count 24
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from multiprocessing import get_context
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import ROOT, STOCKFISH  # noqa: E402

_e = {}


def _init(depth):
    import chess.engine
    _e["engine"] = chess.engine.SimpleEngine.popen_uci(str(STOCKFISH))
    _e["engine"].configure({"Threads": 1, "Hash": 64})
    _e["depth"] = depth


def one_game(moves):
    import chess
    import chess.engine
    from review import review
    board, losses, limit = chess.Board(), [], chess.engine.Limit(depth=_e["depth"])
    for uci in moves:
        # best move vs the played move, both searched from THIS position (same side to move, same depth):
        # comparing with the position after the move would count the side-to-move bonus as a loss
        move, me = chess.Move.from_uci(uci), board.turn
        info = _e["engine"].analyse(board, limit)
        best = max(-1500, min(1500, info["score"].pov(me).score(mate_score=1500)))
        if info["pv"][0] == move:
            losses.append(0)
        else:
            played = _e["engine"].analyse(board, limit, root_moves=[move])["score"].pov(me).score(mate_score=1500)
            losses.append(best - max(-1500, min(1500, played)))
        board.push(move)
    rv = review(moves)
    return [{"loss": max(0, loss), "immediate": n["immediate_label"], "label": n["label"], "score": n["score"],
             "delta": n["delta"]} for n, loss in zip(rv["notes"], losses)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=Path, required=True)
    parser.add_argument("--count", type=int, default=24)
    parser.add_argument("--depth", type=int, default=10)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    data = json.loads(args.games.read_text(encoding="utf-8"))
    games = [g["moves"].split() for g in data.get("list", [])][:args.count]
    pool = ProcessPoolExecutor(args.workers, mp_context=get_context("spawn"), initializer=_init, initargs=(args.depth,))
    rows = [row for game in pool.map(one_game, games) for row in game]
    for process in list(getattr(pool, "_processes", {}).values()):
        process.kill()
    pool.shutdown(wait=False, cancel_futures=True)
    blunders = [r for r in rows if r["loss"] >= 200]
    mistakes = [r for r in rows if 100 <= r["loss"] < 200]
    fine = [r for r in rows if r["loss"] < 50]
    print(f"{len(games)} jogos, {len(rows)} lances; asneiras reais (>=2 peoes) {len(blunders)}, erros (1-2) {len(mistakes)}")
    for name in ("immediate", "label"):
        what = "so o efeito imediato" if name == "immediate" else "imediato + consequencias"
        print(f"  {what:28s}: apanha {100 * sum(r[name] == 'bad' for r in blunders) / max(1, len(blunders)):4.0f}% das asneiras, "
              f"{100 * sum(r[name] == 'bad' for r in mistakes) / max(1, len(mistakes)):4.0f}% dos erros; "
              f"falsos alarmes {100 * sum(r[name] == 'bad' for r in fine) / max(1, len(fine)):4.1f}% dos lances bons; "
              f"lances marcados maus que sao mesmo erros: "
              f"{100 * sum(r[name] == 'bad' and r['loss'] >= 100 for r in rows) / max(1, sum(r[name] == 'bad' for r in rows)):4.0f}%")


if __name__ == "__main__":
    main()
