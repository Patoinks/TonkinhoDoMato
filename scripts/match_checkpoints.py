"""Evaluation match: one TonkAI against one frozen Tonkinho, in parallel.

Neither side learns and nothing is written into a campaign: the result goes
to history/matches/ (or --out). Colours alternate; both sides play their
normal exploring style (else every game with the same colours would repeat);
no resignation, so every game is played to its end. Each game has its own
seed, so a match is reproducible whatever the number of workers.

    python scripts/match_checkpoints.py --current benchmarks/tonkinho-v3.1-jogo65963.pt \\
        --rival benchmarks/tonkinho-checkpoint1-jogo52895.pt --games 500

--current may be a campaign checkpoint (latest.pt) or a snapshot file.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
import math
from multiprocessing import get_context
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_state = {}


def _load(path, cfg=None):
    """(name, net, knowledge, cfg, kd_values) from a campaign checkpoint or a snapshot."""
    import torch
    from tonkai.actor_snapshot import unpack_snapshot
    from tonkai.captures import CaptureStats, parse_weights, replay_games
    from tonkai.cognitive_config import CognitiveConfig
    from tonkai.cognitive_trainer import CognitiveKnowledge, CognitiveTrainer
    from tonkai.temporal_model import SurvivalNet

    data = torch.load(path, map_location="cpu", weights_only=False)
    if data.get("format") == "tonkinho-snapshot-v1":
        cfg = cfg or CognitiveConfig.from_dict(data["config"])
        cfg.device = "cpu"
        weights, knowledge_state = unpack_snapshot(data["payload"])
        net = SurvivalNet(cfg).eval().requires_grad_(False)
        net.load_state_dict(weights)
        knowledge = CognitiveKnowledge(cfg, net.encoder)
        knowledge.load_state_dict(knowledge_state)
        name = f"{data.get('model_name') or 'Tonkinho congelado'} (jogo {data['version']})"
        return name, net, knowledge, cfg, data.get("kd_values")
    cfg = CognitiveConfig.from_dict(data["config"])
    cfg.device = "cpu"
    trainer = CognitiveTrainer(cfg)
    trainer.load_state_dict(data)
    stats = CaptureStats(cfg.hot_piece_kills, opponent_weights=parse_weights(cfg.value_opponents))
    if data.get("capture_stats"):
        stats.load_state_dict(data["capture_stats"])
    if stats.needs_seed() and data.get("replay"):
        stats.seed_new_measures(replay_games(data["replay"]["episodes"]))
    kd = stats.piece_values() if cfg.kd_weight > 0 else None
    name = f"{data.get('model_name', 'TonkAI')} (jogo {data['games']})"
    return name, trainer.net.eval(), trainer.knowledge, cfg, kd


def _init(current, rival, rival_name, mates=None, piece_values=None):
    import copy
    import torch
    torch.set_num_threads(1)
    name, net, knowledge, cfg, kd = _load(current)
    other, rnet, rknowledge, _, _ = _load(rival, cfg)
    ccfg = copy.copy(cfg)                  # mate calculation for the current side only
    for key, value in (mates or {}).items():
        setattr(ccfg, key, value)
    if piece_values:                       # both sides play with these piece values
        kd = piece_values
    _state.update(name=name, net=net, knowledge=knowledge, cfg=cfg, ccfg=ccfg, kd=kd,
                  rival_name=rival_name or other, rnet=rnet, rknowledge=rknowledge)


def _game(index, seed):
    import chess
    import chess.pgn
    import numpy as np
    from tonkai.cognitive_agent import CognitiveAgent
    from tonkai.opponents import CognitiveOpponent
    from tonkai.survival_agent import play_episode

    s = _state
    colour = chess.WHITE if index % 2 == 0 else chess.BLACK
    agent = CognitiveAgent(s["net"], s["ccfg"], s["knowledge"], kd_values=s["kd"])
    rival = CognitiveOpponent(CognitiveAgent(s["rnet"], s["cfg"], s["rknowledge"], kd_values=s["kd"]), s["cfg"],
                              np.random.default_rng([seed, index, 1]))
    rival.description = s["rival_name"]
    episode, board, colour, stats = play_episode(agent, rival, s["cfg"], np.random.default_rng([seed, index]),
                                                 self_color=colour, opponent_name=s["rival_name"])
    outcome = board.outcome(claim_draw=True)
    if not episode.terminated or outcome is None:
        key = "unfinished"
    elif outcome.winner is None:
        key = "draw"
    else:
        key = "win" if outcome.winner == colour else "loss"
    pgn = chess.pgn.Game.from_board(board)
    series = stats.get("king_safety")          # v4: KS of both kings after every ply
    if series:
        node = pgn
        pgn.comment = f"[%ks w={series[0][0]:.3f} b={series[0][1]:.3f}]"
        for white, black in series[1:]:
            node = node.variations[0]
            node.comment = f"[%ks w={white:.3f} b={black:.3f}]"
    pgn.headers.update(Event="TonkAI evaluation match", Round=str(index + 1),
                       White=s["name"] if colour else s["rival_name"], Black=s["rival_name"] if colour else s["name"],
                       Result=outcome.result() if outcome else "*",
                       Termination=outcome.termination.name.lower() if outcome else "max_plies")
    return index, colour, key, len(board.move_stack), str(pgn), s["name"], s["rival_name"]


def summarise(rows):
    def score(part):
        done = [r for r in part if r[2] != "unfinished"]
        points = sum(1 if r[2] == "win" else .5 if r[2] == "draw" else 0 for r in done)
        return {"games": len(part), "win": sum(r[2] == "win" for r in part), "draw": sum(r[2] == "draw" for r in part),
                "loss": sum(r[2] == "loss" for r in part), "unfinished": sum(r[2] == "unfinished" for r in part),
                "score": points / len(done) if done else None}
    total = score(rows)
    p = total["score"]
    elo = None if p in (None, 0, 1) else -400 * math.log10(1 / p - 1)
    # Standard error of the per-game score (draws included), 95% interval.
    values = [1 if r[2] == "win" else .5 if r[2] == "draw" else 0 for r in rows if r[2] != "unfinished"]
    mean = sum(values) / len(values)
    se = math.sqrt(sum((v - mean) ** 2 for v in values) / max(1, len(values) - 1) / len(values))
    return {**total, "elo_difference": elo, "score_95": [max(0, mean - 1.96 * se), min(1, mean + 1.96 * se)],
            "as_white": score([r for r in rows if r[1]]), "as_black": score([r for r in rows if not r[1]]),
            "mean_plies": sum(r[3] for r in rows) / len(rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current", type=Path, default=ROOT / "checkpoints/cognitive-mix-20260923/latest.pt")
    parser.add_argument("--rival", type=Path, required=True, help="a snapshot file (tonkinho-jogo-N.pt)")
    parser.add_argument("--name", help="the rival's display name")
    parser.add_argument("--games", type=int, default=50)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out", type=Path, default=ROOT / "history/matches")
    parser.add_argument("--label", default=None, help="file name stem for the result")
    parser.add_argument("--mate-search-moves", type=int, default=0,
                        help="current side plays a forced mate found within N own moves")
    parser.add_argument("--mate-defense-moves", type=int, default=0,
                        help="current side avoids moves after which the rival forces mate within N moves")
    parser.add_argument("--mate-node-limit", type=int, default=20_000, help="nodes per mate search")
    parser.add_argument("--piece-values", help="both sides use these values, e.g. 1:1,2:1.03,3:1.22,4:1.70,5:4.02")
    args = parser.parse_args()
    values = ({int(k): float(v) for k, v in (part.split(":") for part in args.piece_values.split(","))}
              if args.piece_values else None)
    mates = {"mate_search_moves": args.mate_search_moves, "mate_defense_moves": args.mate_defense_moves,
             "mate_node_limit": args.mate_node_limit}

    rows = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn"), initializer=_init,
                             initargs=(str(args.current), str(args.rival), args.name, mates, values)) as pool:
        for row in pool.map(_game, range(args.games), [args.seed] * args.games):
            rows.append(row)
            if len(rows) % 50 == 0:
                print(f"{len(rows)}/{args.games} jogos", flush=True)
    rows.sort()
    report = {"current": rows[0][5], "current_file": str(args.current), "rival": rows[0][6],
              "rival_file": str(args.rival), "seed": args.seed, "mate_calculation": mates, "piece_values": values,
              "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), **summarise(rows)}
    stem = args.label or f"match-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{stem}.pgn").write_text("\n\n".join(r[4] for r in rows), encoding="utf-8")
    report["pgn"] = f"{stem}.pgn"
    (args.out / f"{stem}.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
