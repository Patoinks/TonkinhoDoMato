"""Where does Claudio lose? Stockfish only as a JUDGE (nothing is learned from it):
every Claudio move of a finished match is scored by the evaluation drop it causes
(before vs after, from Claudio's side, depth --depth), then grouped by phase and kind.

    python claudio/blunders.py --games claudio/results/engine-trained-duelcfg-sf1800.json --label trained-sf1800
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from multiprocessing import get_context
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import ROOT, STOCKFISH  # noqa: E402

VALUES = {1: 1, 2: 3, 3: 3, 4: 5, 5: 9}          # the judge's material count only, never used by Claudio
_e = {}


def _score(info, colour):
    import chess.engine
    score = info["score"].pov(colour)
    return max(-1000, min(1000, score.score(mate_score=1000)))


def judge(game, depth):
    import chess
    import chess.engine
    if "engine" not in _e:
        _e["engine"] = chess.engine.SimpleEngine.popen_uci(str(STOCKFISH))
        _e["engine"].configure({"Threads": 1, "Hash": 32})
    engine = _e["engine"]
    board = chess.Board()
    moves = game["moves"].split()
    me = chess.WHITE if game["white"] else chess.BLACK
    rows = []
    for ply, uci in enumerate(moves):
        move = chess.Move.from_uci(uci)
        if ply >= 4 and board.turn == me:
            before = _score(engine.analyse(board, chess.engine.Limit(depth=depth)), me)
            material = sum(VALUES.get(p.piece_type, 0) * (1 if p.color == me else -1) for p in board.piece_map().values())
            capture, check = board.is_capture(move), board.gives_check(move)
            board.push(move)
            after = _score(engine.analyse(board, chess.engine.Limit(depth=depth)), me)
            rows.append({"ply": ply, "move": uci, "before": before, "after": after, "loss": before - after,
                         "material": material, "pieces": chess.popcount(board.occupied), "capture": capture,
                         "check": check})
        else:
            board.push(move)
    return {"game": game["game"], "result": game["result"], "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=Path, required=True)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    data = json.loads(args.games.read_text(encoding="utf-8"))
    games = data["list"]
    with ProcessPoolExecutor(args.workers, mp_context=get_context("spawn")) as pool:
        judged = list(pool.map(judge, games, [args.depth] * len(games)))
    out = ROOT / "claudio/results" / f"blunders-{args.label}.json"
    out.write_text(json.dumps(judged), encoding="utf-8")
    rows = [r for g in judged for r in g["rows"]]
    big = [r for r in rows if r["loss"] >= 200]
    print(f"{len(judged)} jogos, {len(rows)} lances do Claudio, {len(big)} erros >= 2 peoes "
          f"({len(big) / len(judged):.1f} por jogo)")
    for name, test in (("abertura (lance < 15)", lambda r: r["ply"] < 30),
                       ("meio-jogo", lambda r: r["ply"] >= 30 and r["pieces"] > 12),
                       ("final (<= 12 pecas)", lambda r: r["ply"] >= 30 and r["pieces"] <= 12)):
        part = [r for r in rows if test(r)]
        part_big = [r for r in part if r["loss"] >= 200]
        mean = sum(max(0, r["loss"]) for r in part) / max(1, len(part))
        print(f"  {name:24s} {len(part):5d} lances, perda media {mean:5.0f} cp, erros grandes {len(part_big)}")
    first = []
    for g in judged:
        if g["result"] != "loss":
            continue
        lost = next((r for r in g["rows"] if r["after"] <= -300), None)
        if lost:
            first.append(lost["ply"] // 2 + 1)
    first.sort()
    if first:
        print(f"derrotas: a posicao fica perdida (<= -3 peoes) no lance mediano {first[len(first) // 2]} "
              f"(quartis {first[len(first) // 4]} / {first[3 * len(first) // 4]})")
    decided = [g for g in judged if g["result"] == "loss"]
    quiet = sum(1 for g in decided if next((r for r in g["rows"] if r["after"] <= -300), {}).get("material", 0) >= 0)
    print(f"das {len(decided)} derrotas, {quiet} ficaram perdidas SEM o Claudio estar abaixo em material antes do erro")


if __name__ == "__main__":
    main()
