"""Calculation depth against quality: the same real positions decided at every
depth, each choice scored by a UCI engine used only as a judge.

Per depth: mean loss against the engine's best move (centipawns, capped),
blunders (loss >= 200) and milliseconds per decision.

    python scripts/depth_ratio.py --checkpoint checkpoints/obj-sf1800-max/latest.pt \\
        --games history/matches/eval-sf1800-max-before.games.json
"""
import argparse
from copy import copy
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
ENGINE = ROOT / "engines/stockfish-11/stockfish-11-win/Windows/stockfish_20011801_x64_bmi2.exe"
# tactic_depth 0 = one-move tactics only; N = our move + N half-moves.
DEPTHS = {"1 lance (táticas)": 0, "3 meios-lances": 2, "4 meios-lances": 3, "5 meios-lances": 4, "6 meios-lances": 5}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--games", type=Path, required=True)
    parser.add_argument("--positions", type=int, default=150)
    parser.add_argument("--every", type=int, default=3, help="decide at every Nth own move")
    parser.add_argument("--judge-depth", type=int, default=12)
    parser.add_argument("--cap", type=int, default=1000)
    args = parser.parse_args()
    import chess
    import chess.engine
    import torch
    torch.set_num_threads(1)
    from match_checkpoints import _load
    from tonkai.cognitive_agent import CognitiveAgent
    name, net, knowledge, base, kd = _load(str(args.checkpoint))
    agents = {}
    for label, depth in DEPTHS.items():
        cfg = copy(base)
        for key, value in dict(tactic_candidates=4, tactic_depth=depth, avoid_draws_ahead=3.0, endgame_pieces=10,
                               mate_attack_ks=0.0, mate_defense_ks=0.0, threat_explore=0.0, trade_explore=0.0).items():
            setattr(cfg, key, value)
        agents[label] = CognitiveAgent(net, cfg, knowledge, kd_values=kd)
    judge = chess.engine.SimpleEngine.popen_uci(str(ENGINE))
    judge.configure({"Threads": 1, "Hash": 64})
    limit = chess.engine.Limit(depth=args.judge_depth)

    def value(board, colour):
        return judge.analyse(board, limit)["score"].pov(colour).score(mate_score=10_000)

    stats = {label: {"loss": [], "ms": []} for label in DEPTHS}
    games = json.loads(args.games.read_text(encoding="utf-8"))
    done = 0
    for game in games:
        if done >= args.positions:
            break
        colour = chess.WHITE if game["tonkai"] == "white" else chess.BLACK
        board = chess.Board()
        for agent in agents.values():
            agent.reset()
            agent.observe(board, colour)
        own = 0
        for uci in game["moves"]:
            if board.turn == colour and not board.is_game_over():
                own += 1
                if own % args.every == 0 and done < args.positions:
                    best = value(board, colour)
                    for label, agent in agents.items():
                        started = time.perf_counter()
                        with torch.inference_mode():
                            move = agent.choose(board, colour, None, explore=False).move
                        stats[label]["ms"].append(1000 * (time.perf_counter() - started))
                        after = board.copy()
                        after.push(move)
                        got = value(after, colour) if not after.is_game_over() else (
                            10_000 if after.is_checkmate() else 0)
                        stats[label]["loss"].append(min(args.cap, max(0, best - got)))
                    done += 1
            board.push_uci(uci)
            for agent in agents.values():
                agent.observe(board, colour)
    judge.quit()
    report = {}
    for label, row in stats.items():
        losses, ms = row["loss"], row["ms"]
        report[label] = {"positions": len(losses), "mean_loss_cp": sum(losses) / len(losses),
                         "blunders": sum(loss >= 200 for loss in losses), "ms_per_decision": sum(ms) / len(ms)}
        print(f"{label:18s} perda média {report[label]['mean_loss_cp']:6.1f} cp   erros graves {report[label]['blunders']:3d}"
              f"   {report[label]['ms_per_decision']:7.1f} ms/decisão", flush=True)
    (ROOT / "history/depth-ratio.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
