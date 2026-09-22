"""Train the recurrent survival model against Stockfish and a historical population.

Legacy training and checkpoints are never changed. --init-from migrates shared
weights; --resume restores a V2 learner, replay, population and random states.
"""
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import time
import uuid

import numpy as np
import torch

from tonkai.population import Population
from tonkai.survival_agent import SurvivalAgent, NeuralOpponent, play_episode
from tonkai.survival_config import SurvivalConfig
from tonkai.survival_trainer import SurvivalTrainer
from tonkai.temporal_model import SurvivalNet
from tonkai.uci_opponent import UCIOpponent
from tonkai.win_history import LongestWinRecorder
from tonkai.web_publish import publish_completed


ROOT = Path(__file__).resolve().parent
DEFAULT_ENGINE = ROOT / "engines/stockfish-11/stockfish-11-win/Windows/stockfish_20011801_x64.exe"


def now():
    return datetime.now(timezone.utc).isoformat()


def cpu_weights(net):
    return {key: value.detach().cpu().clone() for key, value in net.state_dict().items()}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--init-from", type=Path)
    source.add_argument("--resume", type=Path)
    ap.add_argument("--games", type=int, default=20000, help="additional V2 games")
    ap.add_argument("--opponent-path", type=Path, default=None)
    ap.add_argument("--opponent-elo", type=int, default=None)
    ap.add_argument("--opponent-time", type=float, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default=None)
    ap.add_argument("--checkpoint-dir", type=Path, default=None)
    ap.add_argument("--history-dir", type=Path, default=ROOT / "history")
    ap.add_argument("--stop-file", type=Path, default=None, help="finish queued games and save when this file appears")
    ap.add_argument("--max-plies", type=int, default=None)
    ap.add_argument("--log-every", type=int, default=None)
    ap.add_argument("--checkpoint-every", type=int, default=None)
    ap.add_argument("--min-replay", type=int, default=None)
    ap.add_argument("--sequence-batch-size", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--external-probability", type=float, default=None)
    ap.add_argument("--historical-probability", type=float, default=None)
    ap.add_argument("--torch-threads", type=int, default=1)
    args = ap.parse_args(argv)
    if args.games < 1 or args.torch_threads < 1:
        ap.error("games and torch-threads must be positive")
    path = args.resume or args.init_from
    if not path.is_file():
        ap.error(f"checkpoint does not exist: {path}")
    state = torch.load(path, map_location="cpu", weights_only=False)
    cfg = SurvivalConfig.from_dict(state["config"]) if args.resume else SurvivalConfig()
    if args.init_from:
        if state.get("format") == "tonkai-survival-v2":
            ap.error("use --resume for a survival checkpoint")
        for name in ("latent_size", "hidden_size", "piece_hidden", "head_hidden",
                     "type_embed", "square_embed", "side_embed"):
            if name in state.get("config", {}):
                setattr(cfg, name, state["config"][name])
    mapping = {"opponent_time": "opponent_move_time", "workers": "num_workers"}
    for option in ("opponent_elo", "opponent_time", "workers", "device", "checkpoint_dir",
                   "max_plies", "log_every", "checkpoint_every", "min_replay", "sequence_batch_size",
                   "seed", "external_probability", "historical_probability"):
        value = getattr(args, option)
        if value is not None:
            setattr(cfg, mapping.get(option, option), str(value) if isinstance(value, Path) else value)
    cfg.validate()
    args.opponent_path = (args.opponent_path or Path(cfg.opponent_path or DEFAULT_ENGINE)).resolve()
    cfg.opponent_path = str(args.opponent_path)
    checkpoint = Path(cfg.checkpoint_dir) / "latest.pt"
    if args.init_from and checkpoint.exists():
        ap.error(f"refusing to overwrite {checkpoint}; use --resume or a new --checkpoint-dir")
    torch.set_num_threads(args.torch_threads)
    torch.manual_seed(cfg.seed)
    net = SurvivalNet(cfg)
    migration = net.load_legacy(state) if args.init_from else None
    trainer = SurvivalTrainer(cfg, net)
    population = Population(cfg, trainer.net, cfg.seed + 3)
    rng = np.random.default_rng(cfg.seed + 4)
    if args.resume:
        trainer.load_state_dict(state)
        population.load_state_dict(state["population"])
        rng.bit_generator.state = state["actor_rng"]
    pretrained_games = state.get("pretrained_games", state.get("games", 0) if args.init_from else 0)
    del state
    run_id = datetime.now(timezone.utc).strftime("survival-%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:8]
    args.history_dir.mkdir(parents=True, exist_ok=True)
    launch_dir = args.history_dir / run_id
    launch_dir.mkdir()
    history_path = args.history_dir / (run_id + ".jsonl")
    stop_file = args.stop_file or launch_dir / "STOP"
    initial_games = trainer.games
    launch = {"pid": os.getpid(), "started_utc": now(), "run_id": run_id,
              "resume_games": initial_games, "pretrained_games": pretrained_games,
              "additional_games": args.games, "objective": "survival",
              "opponent_mixture": {"external": cfg.external_probability,
                                   "historical": cfg.historical_probability,
                                   "current": 1 - cfg.external_probability - cfg.historical_probability},
              "history": str(history_path.resolve()), "checkpoint": str(checkpoint.resolve()),
              "source_checkpoint": str(path.resolve()), "stop_file": str(stop_file.resolve())}
    stopping = False

    def stop(signum, frame):
        nonlocal stopping
        stopping = True
        print("Finishing queued games, then saving survival checkpoint.", flush=True)

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    counts = Counter()
    opponents = Counter()
    draw_reasons = Counter()
    plies, completed, submitted, learned_positions = 0, 0, 0, 0
    losses = {}
    started = time.perf_counter()
    pool = None
    external = None
    inflight = {}
    recorder = LongestWinRecorder(cfg.win_pgn_path)
    status = "running"
    history = history_path.open("a", encoding="utf-8", buffering=1)

    def record(kind, **data):
        history.write(json.dumps({"kind": kind, "run_id": run_id, **data, "timestamp": now()}, allow_nan=False) + "\n")

    def metrics(final=False):
        elapsed = time.perf_counter() - started
        data = {"games": completed, "final": final, "state": status, "objective": "survival",
                "wins": counts["wins"], "draws": counts["draws"], "losses": counts["losses"],
                "draw_reasons": dict(draw_reasons),
                "truncated": counts["truncated"], "plies": plies,
                "survival_plies_mean": plies / max(1, completed),
                "games_per_second": completed / max(.001, elapsed),
                "positions_per_second": learned_positions / max(.001, elapsed),
                "elapsed_seconds": elapsed, "train_batches": trainer.steps,
                "opponents": dict(opponents), "loss": losses,
                "population_size": len(population), "replay_positions": len(trainer.replay)}
        record("metrics", **data)
        print(json.dumps(data), flush=True)

    def save():
        trainer.save(checkpoint, population=population.state_dict(), actor_rng=deepcopy(rng.bit_generator.state),
                     pretrained_games=pretrained_games, source_checkpoint=str(path.resolve()),
                     migration=migration, run_id=run_id, status=status,
                     opponent=launch.get("opponent"))

    def opponent_selection():
        kind = population.choose_kind(rng)
        if kind == "historical":
            enemy, name = population.historical(rng)
        elif kind == "current":
            enemy, name = trainer.net, f"current@{trainer.games}"
        else:
            enemy, name = None, external.description
        return kind, enemy, name

    try:
        external = UCIOpponent(args.opponent_path, cfg.opponent_elo, cfg.opponent_move_time)
        launch["opponent"] = external.info
        (launch_dir / "launch.json").write_text(json.dumps(launch, indent=2), encoding="utf-8")
        record("run", config=cfg.to_dict(), launch=launch, migration=migration)
        print(f"Run {run_id}: {args.games} survival games; {cfg.num_workers} actors; {external.description}", flush=True)
        if cfg.num_workers > 1:
            from tonkai.survival_pool import SurvivalActorPool
            pool = SurvivalActorPool(cfg, str(args.opponent_path), cfg.num_workers)
        else:
            actor_net = population.current(trainer.net)
            actor = SurvivalAgent(actor_net, cfg)
        idle = list(range(cfg.num_workers))
        while completed < args.games:
            stopping = stopping or stop_file.exists()
            if pool:
                while idle and submitted < args.games and not stopping:
                    worker_id = idle.pop()
                    kind, enemy, name = opponent_selection()
                    seed = int(rng.integers(0, 2**31 - 1))
                    pool.submit(worker_id, cpu_weights(trainer.net), kind,
                                cpu_weights(enemy) if enemy is not None else None, name, seed)
                    inflight[worker_id] = kind
                    submitted += 1
                if not inflight:
                    break
                worker_id, episode, board, colour, stats = pool.receive(timeout=120)
                kind = inflight.pop(worker_id)
                idle.append(worker_id)
            else:
                if stopping:
                    break
                kind, enemy, name = opponent_selection()
                opponent = external if kind == "external" else NeuralOpponent(population.current(enemy), cfg, rng)
                actor_net.load_state_dict(cpu_weights(trainer.net))
                episode, board, colour, stats = play_episode(actor, opponent, cfg, rng, opponent_name=name)
            trainer.replay.add(episode)
            trainer.games += 1
            completed += 1
            plies += stats["plies"]
            opponents[kind] += 1
            if not episode.terminated:
                counts["truncated"] += 1
            elif episode.lost:
                counts["losses"] += 1
            elif stats["result"] == "1/2-1/2":
                counts["draws"] += 1
                outcome = board.outcome(claim_draw=True)
                reason = outcome.termination.name.lower() if outcome else "unknown"
                draw_reasons[reason] += 1
                record("draw", game=completed, total_games=trainer.games, reason=reason,
                       opponent=stats["opponent"], plies=stats["plies"])
            else:
                counts["wins"] += 1
            if kind == "external" and episode.terminated:
                try:
                    recorder.consider(board, colour, stats["opponent"], stats["result"],
                                      board.outcome(claim_draw=True).termination.name.lower())
                except (OSError, ValueError) as error:
                    record("warning", component="win_history", error=str(error))
                    print(f"Win history warning: {error}", flush=True)
            for _ in range(cfg.updates_per_game):
                updated = trainer.update()
                if updated:
                    losses = updated
                    learned_positions += updated["learned_positions"]
            if trainer.games % cfg.snapshot_every == 0:
                population.snapshot(trainer.net, trainer.games)
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
    publish_completed(checkpoint, completed=status == "completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
