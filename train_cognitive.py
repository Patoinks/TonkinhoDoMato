"""Opt-in V3 for the NEXT opponent. Never migrates or replaces a running V2 job.

Example (after the current run):
 python train_cognitive.py --from-survival checkpoints/survival/latest.pt --opponent-elo 1600 --games 500 --game-cap 500
"""
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import signal
import time
import uuid

import numpy as np
import torch

from tonkai.captures import CaptureStats
from tonkai.cognitive_config import CognitiveConfig
from tonkai.cognitive_pool import CognitivePool
from tonkai.cognitive_trainer import CognitiveTrainer
from tonkai.uci_opponent import UCIOpponent
from tonkai.win_history import LongestWinRecorder, WinArchive
from tonkai.opening_history import OpeningHistory
from tonkai.web_publish import publish_completed, publish_wins
from tonkai.resign import (ResignController, add_resign_args, hope_threshold,
                           resign_config_from_args)


ROOT = Path(__file__).resolve().parent
ENGINE = ROOT / "engines/stockfish-11/stockfish-11-win/Windows/stockfish_20011801_x64.exe"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--from-survival", type=Path)
    source.add_argument("--resume", type=Path)
    parser.add_argument("--opponent-elo", type=int, required=True, help="explicit next opponent nominal Elo")
    parser.add_argument("--opponent-path", type=Path)
    parser.add_argument("--games", type=int, default=500)
    parser.add_argument("--game-cap", type=int, default=500, help="Persistent campaign cap, including resumed games")
    parser.add_argument("--new-campaign", action="store_true",
                        help="Start a fresh capped campaign from a completed V3, in a new checkpoint directory")
    parser.add_argument("--opponent-move-time", type=float)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--history-dir", type=Path, default=ROOT / "history")
    parser.add_argument("--max-plies", type=int)
    parser.add_argument("--min-replay", type=int)
    parser.add_argument("--experiments-every", type=int)
    parser.add_argument("--generalization-every", type=int)
    parser.add_argument("--win-pgn-path", default=None)
    parser.add_argument("--wins-dir", type=Path, help="archive of every win (default: <history-dir>/wins)")
    parser.add_argument("--planning-nodes", type=int)
    parser.add_argument("--log-every", type=int)
    parser.add_argument("--stop-file", type=Path)
    parser.add_argument("--capture-target-weight", type=float,
                        help="auxiliary target: piece head predicts future captures (0 = off)")
    parser.add_argument("--hot-piece-kills", type=int,
                        help="measurement: an enemy piece with this many kills is 'hot'")
    parser.add_argument("--objective", choices=["survival", "victory", "shaped"],
                        help="shaped = survival blended with the owner's victory measure")
    parser.add_argument("--victory-weight", type=float,
                        help="shaped objective: share of the victory measure in the value target")
    parser.add_argument("--kd-weight", type=float,
                        help="share of the pooled per-type K/D in decision piece values (0 = off)")
    add_resign_args(parser)
    args = parser.parse_args(argv)
    if args.games < 1:
        parser.error("games must be positive")
    if args.game_cap < 1:
        parser.error("game-cap must be positive")
    source_path = (args.resume or args.from_survival).resolve(strict=True)
    state = torch.load(source_path, map_location="cpu", weights_only=False)
    expected = "tonkai-cognitive-v3" if args.resume else "tonkai-survival-v2"
    if state.get("format") != expected:
        parser.error(f"expected a {expected} checkpoint")
    if (args.from_survival and source_path == (ROOT / "checkpoints/survival/latest.pt").resolve()
            and state.get("status") not in ("completed", "stopped")):
        parser.error("The production V2 checkpoint is still running; wait for its final save before starting V3")
    cfg = CognitiveConfig.from_dict(state["config"])
    if args.from_survival:
        defaults = CognitiveConfig()
        for name in CognitiveConfig.__annotations__:
            setattr(cfg, name, getattr(defaults, name))
    cfg.device = "cpu"
    cfg.opponent_elo = args.opponent_elo
    cfg.opponent_path = str((args.opponent_path or Path(cfg.opponent_path or ENGINE)).resolve(strict=True))
    for argument, field in (("workers", "num_workers"), ("checkpoint_dir", "checkpoint_dir"),
                            ("opponent_move_time", "opponent_move_time"),
                            ("max_plies", "max_plies"), ("min_replay", "min_replay"),
                            ("experiments_every", "experiments_every"), ("planning_nodes", "planning_nodes"),
                            ("generalization_every", "generalization_every"), ("win_pgn_path", "win_pgn_path"),
                            ("log_every", "log_every"), ("capture_target_weight", "capture_target_weight"),
                            ("hot_piece_kills", "hot_piece_kills"), ("objective", "objective"),
                            ("victory_weight", "victory_weight"), ("kd_weight", "kd_weight")):
        value = getattr(args, argument)
        if value is not None:
            setattr(cfg, field, str(value) if isinstance(value, Path) else value)
    cfg.validate()
    output = (Path(cfg.checkpoint_dir) / "latest.pt").resolve()
    if args.new_campaign and (not args.resume or state.get("status") not in ("completed", "stopped")
                              or output == source_path or output.exists()):
        parser.error("A new campaign needs a completed/stopped V3 and a new checkpoint directory")
    if args.from_survival and (output == source_path or output.exists()):
        parser.error("V3 requires a new checkpoint directory, separate from its V2 source")
    # Never allow the next version to write into the live V2 checkpoint location.
    if output.parent == (ROOT / "checkpoints/survival").resolve():
        parser.error("V2 checkpoint directory is reserved")
    torch.set_num_threads(1)
    trainer = CognitiveTrainer(cfg)
    trainer.load_state_dict(state)
    rng = np.random.default_rng(cfg.seed + 4)
    if "actor_rng" in state:
        rng.bit_generator.state = state["actor_rng"]
    pretrained_games = state.get("pretrained_games", 0)
    campaign_start = trainer.games if args.new_campaign else state.get("campaign_start_games", trainer.games)
    campaign_cap = args.game_cap if args.new_campaign else min(args.game_cap, state.get("campaign_cap", args.game_cap))
    campaign_done = trainer.games - campaign_start
    args.games = min(args.games, campaign_cap - campaign_done)
    if args.games <= 0:
        print(f"Campaign cap already reached ({campaign_done}/{campaign_cap}); no new games.", flush=True)
        return 0
    openings = OpeningHistory(output.parent, () if args.new_campaign else state.get("opening_history", ()))
    initial_results = {} if args.new_campaign else state.get("campaign_results", {})
    initial_reasons = {} if args.new_campaign else state.get("campaign_draw_reasons", {})
    if args.resume and not args.new_campaign and not initial_results and state.get("run_id"):
        prior_history = args.history_dir / (state['run_id'] + '.jsonl')
        if prior_history.exists():
            for line in reversed(prior_history.read_text(encoding='utf-8').splitlines()):
                prior = json.loads(line)
                if prior.get('kind') == 'metrics':
                    initial_results = {key: prior.get(key, 0) for key in ('wins', 'draws', 'losses', 'truncated')}
                    initial_reasons = prior.get('draw_reasons', {})
                    break
    resign_cfg = resign_config_from_args(args)
    resign = (ResignController(resign_cfg, hope_threshold(resign_cfg, cfg.objective, cfg.survival_gamma,
                                                            cfg.victory_weight))
              if resign_cfg.enabled else None)
    capture_stats = CaptureStats(cfg.hot_piece_kills)
    if not args.new_campaign:
        if resign is not None and state.get("resign"):
            # A state measured against another threshold (e.g. another objective)
            # would keep the old scale; start its verification again instead.
            if math.isclose(state["resign"]["initial_threshold"], resign.initial_threshold):
                resign.load_state_dict(state["resign"])
            else:
                print(f"[resign] threshold changed {state['resign']['initial_threshold']:.4f} -> "
                      f"{resign.initial_threshold:.4f}; verification restarts in shadow", flush=True)
        if state.get("capture_stats") and state["capture_stats"].get("hot_kills") == cfg.hot_piece_kills:
            capture_stats.load_state_dict(state["capture_stats"])
    del state
    initial_games = trainer.games
    run_id = datetime.now(timezone.utc).strftime("cognitive-%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:8]
    args.history_dir.mkdir(parents=True, exist_ok=True)
    run_dir = args.history_dir / run_id
    run_dir.mkdir()
    stop_file = args.stop_file or run_dir / "STOP"
    history_path = args.history_dir / (run_id + ".jsonl")
    stopping = False

    def stop(signum, frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    status, completed, submitted, plies = "running", 0, 0, 0
    counts, reasons, diagnostics = Counter(initial_results), Counter(initial_reasons), Counter()
    started = time.perf_counter()
    history = history_path.open("a", encoding="utf-8", buffering=1)
    loss = {}
    pool = None
    external = None
    recorder = LongestWinRecorder(cfg.win_pgn_path)
    archive = WinArchive(args.wins_dir or args.history_dir / "wins")
    # Deploys are throttled: at most one running, and one every 10 minutes.
    site = {"process": None, "last": -1e9, "pending": False}

    def publish_archive(force=False):
        running = site["process"] is not None and site["process"].poll() is None
        if not site["pending"] or running or (not force and time.monotonic() - site["last"] < 600):
            return
        site.update(process=publish_wins(archive.directory), last=time.monotonic(), pending=False)

    def record(kind, **data):
        history.write(json.dumps({"kind": kind, "run_id": run_id, **data,
                                  "timestamp": datetime.now(timezone.utc).isoformat()}, allow_nan=False) + "\n")

    def metrics(final=False):
        seconds = time.perf_counter() - started
        result = dict(games=completed, final=final, state=status, objective=cfg.objective, wins=counts["wins"],
                      losses=counts["losses"], draws=counts["draws"], truncated=counts["truncated"],
                      draw_reasons=dict(reasons), games_per_second=completed / max(.001, seconds),
                      positions_per_second=plies / max(.001, seconds), survival_plies_mean=plies / max(1, completed),
                      train_batches=trainer.steps, cognitive=dict(diagnostics), loss=loss,
                      archive=len(trainer.knowledge.memory.entries), skills=len(trainer.knowledge.skills.skills),
                      counterfactuals=trainer.experiments, generalizations=trainer.generalizations,
                      experiment_positions=trainer.experiment_positions, captures=capture_stats.summary())
        if resign is not None:
            result["resigned"] = counts["resigned"]
            result["resign"] = resign.summary()
        record("metrics", **result)
        print(json.dumps(result), flush=True)

    def save():
        trainer.save(output, actor_rng=deepcopy(rng.bit_generator.state), pretrained_games=pretrained_games,
                     source_checkpoint=str(source_path), run_id=run_id, status=status,
                     campaign_start_games=campaign_start, campaign_cap=campaign_cap, opening_history=openings.state(),
                     campaign_results=dict(counts), campaign_draw_reasons=dict(reasons),
                     capture_stats=capture_stats.state_dict(),
                     resign=resign.state_dict() if resign is not None else None)

    try:
        external = UCIOpponent(cfg.opponent_path, cfg.opponent_elo, cfg.opponent_move_time)
        launch = dict(pid=os.getpid(), run_id=run_id, resume_games=initial_games, additional_games=args.games,
                      pretrained_games=pretrained_games, objective=cfg.objective, version="cognitive-v3",
                      history=str(history_path.resolve()), checkpoint=str(output), source_checkpoint=str(source_path),
                      stop_file=str(stop_file.resolve()), opponent=external.info,
                      campaign_start_games=campaign_start, campaign_cap=campaign_cap,
                      opening_report=str(output.parent / "opening-report.json"),
                      opponent_mixture={"external": 1, "historical": 0, "current": 0},
                      capture_target_weight=cfg.capture_target_weight, kd_weight=cfg.kd_weight,
                      victory_weight=cfg.victory_weight,
                      resign=resign.summary()["config"] if resign is not None else None)
        (run_dir / "launch.json").write_text(json.dumps(launch, indent=2), encoding="utf-8")
        record("run", config=cfg.to_dict(), launch=launch)
        print(f"V3 {run_id}: {args.games} games, {cfg.num_workers} actors, next opponent {external.description}", flush=True)
        if resign is not None:
            print(resign.line(), flush=True)
        pool = CognitivePool(cfg, cfg.opponent_path)
        while completed < args.games:
            stopping = stopping or stop_file.exists()
            while len(pool.pending) < pool.capacity and submitted < args.games and not stopping:
                seed, game_number = rng.integers(0, 2**31 - 1), campaign_done + submitted + 1
                extras = {}
                if resign is not None:
                    extras["resign_plan"] = resign.plan()
                kd_values = capture_stats.piece_values() if cfg.kd_weight > 0 else None
                if kd_values is not None:
                    extras["kd_values"] = kd_values
                pool.submit(trainer.net, trainer.knowledge, seed, game_number, **extras)
                submitted += 1
            if not pool.pending:
                break
            episode, board, colour, stats, trace = pool.receive()
            trainer.ingest(episode, trace, colour)
            completed += 1
            openings.add(stats["game_number"], board, colour, stats["opponent"])
            if stats["game_number"] <= 5:
                record("opening", **{k:v for k,v in openings.rows[stats["game_number"]].items() if k != "pgn"})
            plies += stats["plies"]
            diagnostics.update(stats["cognitive"])
            resign_report = stats.get("resign")
            resigned = bool(resign_report and resign_report["resigned"])
            if not episode.terminated:
                counts["truncated"] += 1
            elif episode.lost:
                counts["losses"] += 1
                if resigned:
                    counts["resigned"] += 1
            elif stats["result"] == "1/2-1/2":
                counts["draws"] += 1
                reasons[board.outcome(claim_draw=True).termination.name.lower()] += 1
            else:
                counts["wins"] += 1
            if resigned:
                outcome = "resigned"
            elif episode.terminated:
                outcome = ("loss" if episode.lost else "draw" if stats["result"] == "1/2-1/2" else "win")
            else:
                outcome = "unfinished"
            capture_stats.add(board.move_stack, colour, outcome)
            if resign is not None:
                for message in resign.update(resign_report, lost=episode.lost):
                    print(message, flush=True)
                    record("resign", game=completed, message=message, **resign.summary())
            if episode.terminated and not resigned:     # a resigned board has no outcome
                try:
                    termination = board.outcome(claim_draw=True).termination.name.lower()
                    recorder.consider(board, colour, stats["opponent"], stats["result"], termination)
                    if archive.add(board, colour, stats["opponent"], stats["result"], termination,
                                   run_id=run_id, game=trainer.games, objective=cfg.objective):
                        site["pending"] = True
                        record("win", game=completed, total_games=trainer.games, plies=len(board.move_stack))
                except (OSError, ValueError) as error:
                    record("warning", component="win_history", error=str(error))
                publish_archive()
            # The separate engine only plays legal experimental continuations; its
            # evaluations never become labels and branches never count as real games.
            if not stopping:
                for event in trainer.investigate(episode, trace, colour, external):
                    record("experiment", experiment_type=event["kind"], **{key: value for key, value in event.items() if key != "kind"})
            for _ in range(cfg.updates_per_game):
                update = trainer.update()
                if update:
                    loss = update
            if completed % cfg.log_every == 0:
                metrics()
            if completed % cfg.checkpoint_every == 0:
                save()
        status = "completed" if completed == args.games else "stopped"
    except BaseException as error:
        status = "failed"
        record("error", error=repr(error))
        raise
    finally:
        try:
            if pool:
                pool.close()
            if external:
                external.close()
        finally:
            save()
            metrics(final=True)
            history.close()
    publish_completed(output, completed=status == "completed")
    publish_archive(force=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
