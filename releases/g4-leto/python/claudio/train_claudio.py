"""Claudio training: ONE learner process and one learner thread (the owner's rule).

Claudio plays Timdgei or a strength-limited Stockfish and learns
from every game the way TonkAI always learned:

- the network: survival + the owner's shaped objective (80% survival, 20%
  simplify + material one horizon ahead), `--updates` learner steps per game,
  continuing Timdgei's own learner state and replay;
- its piece values are measured in these games too (kills/deaths, kind
  "claudio") and kept on their absolute weighted-K/D scale; the pawn is not
  forced to 1;
- labels are organic: no tablebases, only the rules and checkmates.

Stops after --minutes of wall time (everything counted: games, labels,
updates) and writes a Claudio checkpoint.

    python claudio/train_claudio.py --minutes 60 --out claudio/claudio-trained.pt
    python claudio/train_claudio.py --opponent stockfish --elo 2850 --max-games 1000 --out claudio/claudio-sf2850-1000.pt
"""
import argparse
import copy
import json
import sys
import time
from collections import Counter
from pathlib import Path

import chess

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import ROOT  # noqa: E402  (frozen base code first)

TIMDGEI = ROOT / "checkpoints/obj-max-continuo/latest.pt"
EVOLUTION = ROOT / "history/evolution/20260924T190310Z/evolution.json"


def mate_cut_scores(games):
    """Mean enemy-king-zone squares cut by each surviving mating piece.

    Every non-king piece of the mating side is a sample, including pieces that
    cut zero squares.  This avoids rewarding a type merely because zero-credit
    pieces of that type were omitted.  Both sides' checkmates teach the same
    type-level value; no hand-written piece table or pawn anchor is involved.
    """
    totals, samples = Counter(), Counter()
    mates = 0
    for actions in games:
        board = chess.Board()
        for action in actions:
            board.push_uci(action)
        if not board.is_checkmate():
            continue
        mates += 1
        loser, winner = board.turn, not board.turn
        king = board.king(loser)
        if king is None:
            continue
        zone = chess.BB_KING_ATTACKS[king] | chess.BB_SQUARES[king]
        for square, piece in board.piece_map().items():
            if piece.color != winner or piece.piece_type == chess.KING:
                continue
            totals[piece.piece_type] += chess.popcount(board.attacks_mask(square) & zone)
            samples[piece.piece_type] += 1
    values = {kind: (totals[kind] / samples[kind] if samples[kind] else 0.0)
              for kind in range(chess.PAWN, chess.KING)}
    return values, {"checkmates": mates,
                    "samples": {kind: samples[kind] for kind in range(chess.PAWN, chess.KING)},
                    "cut_squares": {kind: totals[kind] for kind in range(chess.PAWN, chess.KING)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--minutes", type=float, default=60.0)
    parser.add_argument("--updates", type=int, default=6, help="learner steps per game (golden: 6)")
    parser.add_argument("--refresh", type=int, default=5, help="games between copying the learner into the player")
    parser.add_argument("--random-plies", type=int, default=2, help="random opening plies (exploration)")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--opponent", choices=("timdgei", "stockfish"), default="timdgei")
    parser.add_argument("--elo", type=int, default=2850, help="nominal Stockfish Elo")
    parser.add_argument("--move-time", type=float, default=0.002, help="Stockfish seconds per move")
    parser.add_argument("--mate-control-weight", type=float, default=0.0,
                        help="piece-value credit for enemy king-zone squares controlled at checkmate")
    parser.add_argument("--kd-value-weight", type=float, default=0.5,
                        help="final value blend; 0.5 means 50%% absolute weighted K/D + 50%% mate cuts")
    parser.add_argument("--from-claudio", type=Path,
                        help="continue the network, values, settings and history from this Claudio checkpoint")
    parser.add_argument("--normalize-pawn", action="store_true",
                        help="after the organic blend, divide all final values by the learned pawn value")
    parser.add_argument("--set", action="append", default=[], help="Claudio setting name=value")
    parser.add_argument("--out", type=Path, default=ROOT / "claudio/claudio-trained.pt")
    parser.add_argument("--log", type=Path, default=ROOT / "claudio/results/train.jsonl")
    parser.add_argument("--max-games", type=int, default=0, help="stop early after this many games (tests)")
    args = parser.parse_args()
    if not 0.0 <= args.kd_value_weight <= 1.0:
        parser.error("--kd-value-weight must be between 0 and 1")

    start = time.perf_counter()
    deadline = start + 60.0 * args.minutes
    import psutil_free_priority  # noqa: F401
    import numpy as np
    import torch
    torch.set_num_threads(1)
    from match_checkpoints import _historical_config
    import evolve_tonkinho as evo
    from tonkai import fast_rules
    from tonkai.captures import CaptureStats, parse_weights
    from tonkai.cognitive_agent import CognitiveAgent
    from tonkai.cognitive_trainer import CognitiveTrainer
    from tonkai.encoding import encode, pack_many
    from tonkai.king_safety_model import load_cached
    from tonkai.opponents import CognitiveOpponent
    from tonkai.uci_opponent import UCIOpponent
    from tonkai.sequence_replay import SequenceEpisode
    from claudio_agent import Claudio, ClaudioConfig
    from claudio_load import architecture_of, save_claudio

    data = torch.load(TIMDGEI, map_location="cpu", weights_only=False)
    cfg = _historical_config(data)
    cfg.syzygy_path, cfg.tablebase_adjudicate = "", False           # organic labels: rules only
    cfg.num_workers, cfg.updates_per_game = 1, args.updates
    cfg.mate_control_weight = args.mate_control_weight
    cfg.validate()
    trainer = CognitiveTrainer(cfg)
    trainer.load_state_dict(data)
    stats = CaptureStats(cfg.hot_piece_kills, opponent_weights=parse_weights(cfg.value_opponents))
    stats.load_state_dict(data["capture_stats"])
    values = (stats.piece_values(cfg.piece_values_kind) or stats.piece_values(cfg.piece_values_fallback))
    king = load_cached(cfg.king_safety_model)
    history = [{"from": str(TIMDGEI), "games_before": data["games"], "values_inherited": values}]
    claudio_config = None
    if args.from_claudio:
        source = torch.load(args.from_claudio, map_location="cpu", weights_only=False)
        if source.get("format") != "claudio-v1":
            raise ValueError(f"{args.from_claudio} is not a Claudio checkpoint")
        trainer.net.load_state_dict(source["net"])
        values = {int(kind): float(value) for kind, value in source["values"].items()}
        claudio_config = source["config"]
        history = list(source.get("history", []))
        history.append({"continued_from": str(args.from_claudio), "values_inherited": values})
    del data

    rival = None
    if args.opponent == "timdgei":
        evo._init(str(TIMDGEI))                                      # the frozen opponent, its own copy
        genes = dict(json.loads(EVOLUTION.read_text(encoding="utf-8"))["base_genes"], tactic_search="exact")
        rcfg = evo._config(genes)
    else:
        rival = UCIOpponent(cfg.opponent_path, elo=args.elo, move_time=args.move_time)
    ccfg = ClaudioConfig.with_settings(args.set, base=claudio_config)
    actor = copy.deepcopy(trainer.net).eval().requires_grad_(False)
    proxy = CognitiveAgent(actor, copy.copy(cfg), trainer.knowledge, kd_values=values)

    args.log.parent.mkdir(parents=True, exist_ok=True)
    log = open(args.log, "w", encoding="utf-8")
    totals = {"win": 0, "draw": 0, "loss": 0}
    game, updates, spent = 0, 0, {"play": 0.0, "labels": 0.0, "learn": 0.0}
    completed_games = []
    try:
        while time.perf_counter() < deadline and (not args.max_games or game < args.max_games):
            t0 = time.perf_counter()
            me = chess.WHITE if game % 2 == 0 else chess.BLACK
            rng = np.random.default_rng([args.seed, game])
            claudio = Claudio(actor, values, ccfg, rng=np.random.default_rng([args.seed, game, 1]), king_model=king)
            if args.opponent == "timdgei":
                rival = CognitiveOpponent(CognitiveAgent(evo._state["net"], rcfg, evo._state["knowledge"],
                                                         kd_values=evo._state["kd"]), rcfg, rng)
            else:
                rival.reset()
            board = chess.Board()
            while True:
                board.reset()
                for _ in range(args.random_plies):
                    moves = list(board.legal_moves)
                    board.push(moves[int(rng.integers(len(moves)))])
                if not board.is_game_over(claim_draw=True):
                    break
            while not board.is_game_over(claim_draw=True) and len(board.move_stack) < cfg.max_plies:
                board.push(claudio.choose(board) if board.turn == me else rival.choose(board, rng))
            t1 = time.perf_counter()

            # Labels: every real position from Claudio's side, its victory measure,
            # and how surprising each opponent move was to its own prediction.
            replay, fast = chess.Board(), fast_rules.board(chess.Board())
            rows, proxies, surprises = [encode(replay, me)], [proxy.position_proxy(replay, me)], [0.0]
            memory = actor.initial_memory(1)
            with torch.inference_mode():
                for move in board.move_stack:
                    latent = actor.encode(torch.from_numpy(rows[-1][None]))
                    memory = actor.memory_cell(latent, memory)
                    surprise = 0.0
                    if replay.turn != me:
                        codes = list(fast.legal_moves())
                        x = torch.from_numpy(fast_rules.encode_after_moves(fast, codes, me))
                        context, _ = actor.candidates(x, memory)
                        probabilities = actor.predict_logits(context).softmax(0).numpy()
                        surprise = -float(np.log(max(float(probabilities[codes.index(fast_rules.code(move))]), 1e-12)))
                    replay.push(move)
                    fast.push(fast_rules.code(move))
                    rows.append(encode(replay, me))
                    proxies.append(proxy.position_proxy(replay, me))
                    surprises.append(surprise)
            outcome = board.outcome(claim_draw=True)
            terminated = outcome is not None
            won = terminated and outcome.winner == me
            lost = terminated and outcome.winner is not None and outcome.winner != me
            result = "win" if won else "loss" if lost else "draw"
            completed_games.append(tuple(move.uci() for move in board.move_stack))
            opponent_name = "Timdgei" if args.opponent == "timdgei" else f"Stockfish nominal {args.elo}"
            episode = SequenceEpisode(packed=pack_many(np.stack(rows)), lost=lost, terminated=terminated,
                                      opponent=opponent_name, surprises=np.asarray(surprises, dtype=np.float32),
                                      actions=tuple(m.uci() for m in board.move_stack),
                                      result_value=(1.0 if won else 0.0 if lost else 0.5) if terminated else None,
                                      proxies=np.asarray(proxies, dtype=np.float32))
            trainer.ingest(episode, [], me)
            stats.add(board.move_stack, me, result, detail_kind="claudio")
            t2 = time.perf_counter()
            for _ in range(args.updates):
                if trainer.update() is not None:
                    updates += 1
            game += 1
            if game % args.refresh == 0:
                actor.load_state_dict(trainer.net.state_dict())
            t3 = time.perf_counter()
            spent["play"] += t1 - t0
            spent["labels"] += t2 - t1
            spent["learn"] += t3 - t2
            totals[result] += 1
            row = {"game": game, "colour": "white" if me else "black", "result": result,
                   "plies": len(board.move_stack), "updates": updates, "minutes": (t3 - start) / 60,
                   "seconds": {k: round(v, 1) for k, v in spent.items()}, "totals": dict(totals),
                   "moves": [move.uci() for move in board.move_stack]}
            if game % 25 == 0:
                row["values_claudio"] = stats.piece_values("claudio")
            log.write(json.dumps(row) + "\n")
            log.flush()
    finally:
        if args.opponent == "stockfish" and rival is not None:
            rival.close()
        log.close()
    actor.load_state_dict(trainer.net.state_dict())
    own_relative = stats.piece_values("claudio")
    scores_x3 = stats.piece_scores(kind="claudio")
    # CaptureStats exposes the unnormalised level-weighted K/D multiplied by
    # three as `piece_scores`. Undo that presentation scale: unlike
    # `piece_values`, this does not divide every piece by the pawn.
    own_absolute = (None if scores_x3 is None else
                    {kind: score / 3.0 for kind, score in scores_x3.items()})
    mate_cuts, mate_evidence = mate_cut_scores(completed_games)
    kd_values = own_absolute or values
    blended_values = {kind: (args.kd_value_weight * kd_values[kind]
                             + (1.0 - args.kd_value_weight) * mate_cuts[kind])
                      for kind in range(chess.PAWN, chess.KING)}
    if args.normalize_pawn:
        pawn = blended_values[chess.PAWN]
        if pawn <= 0:
            raise ValueError(f"cannot normalise a non-positive learned pawn value: {pawn}")
        used_values = {kind: value / pawn for kind, value in blended_values.items()}
    else:
        used_values = blended_values
    trained_against = "Timdgei (frozen)" if args.opponent == "timdgei" else f"Stockfish nominal {args.elo}"
    history.append({"trained_games": game, "updates": updates, "minutes": (time.perf_counter() - start) / 60,
                    "opponent": trained_against, "results": totals,
                    "values_measured_absolute": own_absolute,
                    "values_relative_to_pawn_debug": own_relative,
                    "mate_cut_squares_mean": mate_cuts,
                    "mate_cut_evidence": mate_evidence,
                    "value_formula": {"weighted_kd": args.kd_value_weight,
                                      "mate_cut_squares": 1.0 - args.kd_value_weight},
                    "values_before_normalization": blended_values,
                    "normalized_to_pawn": args.normalize_pawn,
                    "values_used": used_values,
                    "seconds": spent, "settings": args.set,
                    "mate_control_weight": args.mate_control_weight})
    save_claudio(args.out, trainer.net, architecture_of(cfg), used_values, ccfg, history, king_model=king)
    trainer_state = args.out.with_suffix(".stats.json")
    trainer_state.write_text(json.dumps({"values_inherited": values,
                                         "values_claudio_absolute": own_absolute,
                                         "values_relative_to_pawn_debug": own_relative,
                                         "mate_cut_squares_mean": mate_cuts,
                                         "mate_cut_evidence": mate_evidence,
                                         "value_formula": {"weighted_kd": args.kd_value_weight,
                                                           "mate_cut_squares": 1.0 - args.kd_value_weight},
                                         "values_before_normalization": blended_values,
                                         "normalized_to_pawn": args.normalize_pawn,
                                         "values_used": used_values,
                                         "results": totals, "games": game, "updates": updates}, indent=1),
                             encoding="utf-8")
    print(f"trained {game} games, {updates} updates, {totals}, {(time.perf_counter() - start) / 60:.1f} min "
          f"-> {args.out}", flush=True)


if __name__ == "__main__":
    main()
