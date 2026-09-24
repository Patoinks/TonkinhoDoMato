"""Opt-in V3 for the NEXT opponent. Never migrates or replaces a running V2 job.

Example (after the current run):
 python train_cognitive.py --from-survival checkpoints/survival/latest.pt --opponent-elo 1600 --games 500 --game-cap 500
"""
import argparse
from collections import Counter, defaultdict
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

from tonkai.king_safety_model import load_cached as load_king_safety
from tonkai.captures import CaptureStats, parse_weights, replay_games

# v3.1: opponent mixture, frozen self-play series, weighted opponent piece
# values, forced resignation and no repetition draw when ahead.
MODEL_NAME, MODEL_VERSION = "Tonkinho do Mato v3.1", "3.1"
# v4: the same, with the learned Emergent King Safety model deciding king safety.
V4_NAME, V4_VERSION = "Tonkinho do Mato v4", "4"
# v5: learned-value attack/protection and protected mobility, equally weighted.
V5_NAME, V5_VERSION = "Tonkinho do Mato v5", "5"
from tonkai import opponents
from tonkai import fast_rules
from tonkai.cognitive_config import CognitiveConfig
from tonkai.cognitive_pool import CognitivePool
from tonkai.cognitive_trainer import CognitiveTrainer
from tonkai.uci_opponent import UCIOpponent
from tonkai.win_history import FeaturedGameRecorder, LongestWinRecorder, WinArchive
from tonkai.opening_history import OpeningHistory
from tonkai.web_publish import publish_completed, publish_wins
from tonkai.resign import (ResignController, add_resign_args, hope_threshold,
                           resign_config_from_args)


ROOT = Path(__file__).resolve().parent
ENGINE = ROOT / "engines/stockfish-11/stockfish-11-win/Windows/stockfish_20011801_x64.exe"


def episode_owner_color_row(row):
    """TonkAI's colour in a stored replay row."""
    from types import SimpleNamespace
    from tonkai.captures import episode_owner_color
    return episode_owner_color(SimpleNamespace(packed=row["packed"], initial_fen=row["initial_fen"]))


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
    parser.add_argument("--archive-wins", action="store_true",
                        help="record wins (archive, featured and longest game); off by the owner's choice")
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
    parser.add_argument("--mix-self", type=float, help="share of games against TonkAI itself")
    parser.add_argument("--mix-tiny", type=float, help="share of games against TinyOpponent")
    parser.add_argument("--mix-fairy", type=float, help="share of games against Fairy-Stockfish")
    parser.add_argument("--fairy-path", type=Path, help="Fairy-Stockfish executable")
    parser.add_argument("--fairy-elo", type=int, help="Fairy-Stockfish nominal UCI_Elo")
    parser.add_argument("--self-snapshot-every", type=int,
                        help="self-play against a frozen TonkAI saved every N games (0 = current weights)")
    parser.add_argument("--king-safety-model", type=Path,
                        help="v4: a learned King Safety model (train_king_safety.py) replaces the hand-written one")
    parser.add_argument("--king-safety-log-every", type=int, help="games between logged KS trajectories")
    parser.add_argument("--mate-search-moves", type=int, help="play a forced mate found within N own moves")
    parser.add_argument("--mate-defense-moves", type=int,
                        help="avoid moves after which the opponent forces mate within N moves")
    parser.add_argument("--mate-node-limit", type=int, help="nodes per mate search")
    parser.add_argument("--threat-explore", type=float,
                        help="share of exploring decisions that try a move creating a mate threat")
    parser.add_argument("--threat-moves", type=int, help="mate threat = forced mate within N moves after a pass")
    parser.add_argument("--rival-dir", type=Path,
                        help="duel: every game against the newest snapshot another campaign writes here")
    parser.add_argument("--rival-set", action="append", default=[], dest="rival_settings",
                        help="duel: the rival's decision setting name=value (repeatable)")
    parser.add_argument("--rival-name", help="duel: the rival's display name")
    parser.add_argument("--set", action="append", default=[], dest="settings",
                        help="any decision setting name=value (repeatable), e.g. from a tournament player")
    parser.add_argument("--exchange-depth", type=int, help="exact capture search depth for piece safety (0 = off)")
    parser.add_argument("--exchange-labels", type=int, choices=(0, 1),
                        help="1: the victory measure's piece part is the exact exchange result")
    parser.add_argument("--trade-explore", type=float, help="share of exploring decisions that try a non-losing capture")
    parser.add_argument("--mate-labels", type=int,
                        help="victory measure = proven result when a forced mate within N moves exists")
    parser.add_argument("--avoid-repetition-margin", type=float,
                        help="when ahead by this many pawns (own piece values), avoid drawing by repetition")
    parser.add_argument("--value-opponents",
                        help="piece values take the opponent side only from these kinds, e.g. 'fairy=0.35,external=0.65'")
    parser.add_argument("--self-games-per-snapshot", type=int,
                        help="play this many games against each new frozen TonkAI, right after it is saved")
    parser.add_argument("--checkpoint-every", type=int, help="save latest.pt every N games")
    parser.add_argument("--compile-learner", action="store_true",
                        help="torch.compile the learner's recurrent unroll (~25%% faster update after a "
                             "~5 min compile; needs MSVC on PATH, see scripts/start_v3_mix.ps1)")
    parser.add_argument("--dedicate-learner-core", action="store_true",
                        help="Windows: learner owns physical core 0 at above-normal priority; actors use the rest")
    add_resign_args(parser)
    parser.add_argument('--compact-sequences', action='store_true')
    parser.add_argument('--incremental-encoder', action='store_true')
    parser.add_argument('--batch-search', action='store_true')
    parser.add_argument('--inference-broker', action='store_true')
    parser.add_argument('--cuda-graphs', action='store_true')
    parser.add_argument('--learner-device', choices=('cpu', 'cuda'), default='cpu')
    args = parser.parse_args(argv)
    if args.cuda_graphs and (args.learner_device != 'cuda' or args.compact_sequences or args.compile_learner):
        parser.error('CUDA graphs need --learner-device cuda and a separate trial from compact/compile')
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
    cfg.device = args.learner_device
    for flag in ('compact_sequences', 'incremental_encoder', 'batch_search', 'inference_broker', 'cuda_graphs'):
        setattr(cfg, flag, bool(getattr(args, flag)))
    cfg.opponent_elo = args.opponent_elo
    cfg.opponent_path = str((args.opponent_path or Path(cfg.opponent_path or ENGINE)).resolve(strict=True))
    if args.king_safety_model is not None:
        cfg.king_safety_model = str(args.king_safety_model.resolve(strict=True))
    model_name, model_version = ((V5_NAME, V5_VERSION) if cfg.piece_activity_weight else
                                 (V4_NAME, V4_VERSION) if cfg.king_safety_model else
                                 (MODEL_NAME, MODEL_VERSION))
    if args.fairy_path is not None:
        cfg.fairy_path = str(args.fairy_path.resolve(strict=True))
    for argument, field in (("workers", "num_workers"), ("checkpoint_dir", "checkpoint_dir"),
                            ("opponent_move_time", "opponent_move_time"),
                            ("max_plies", "max_plies"), ("min_replay", "min_replay"),
                            ("experiments_every", "experiments_every"), ("planning_nodes", "planning_nodes"),
                            ("generalization_every", "generalization_every"), ("win_pgn_path", "win_pgn_path"),
                            ("log_every", "log_every"), ("capture_target_weight", "capture_target_weight"),
                            ("hot_piece_kills", "hot_piece_kills"), ("objective", "objective"),
                            ("victory_weight", "victory_weight"), ("kd_weight", "kd_weight"),
                            ("mix_self", "mix_self"), ("mix_tiny", "mix_tiny"), ("mix_fairy", "mix_fairy"),
                            ("self_snapshot_every", "self_snapshot_every"), ("checkpoint_every", "checkpoint_every"),
                            ("self_games_per_snapshot", "self_games_per_snapshot"),
                            ("value_opponents", "value_opponents"),
                            ("avoid_repetition_margin", "avoid_repetition_margin"),
                            ("king_safety_log_every", "king_safety_log_every"),
                            ("mate_search_moves", "mate_search_moves"), ("mate_defense_moves", "mate_defense_moves"),
                            ("mate_node_limit", "mate_node_limit"), ("threat_explore", "threat_explore"),
                            ("threat_moves", "threat_moves"), ("mate_labels", "mate_labels"),
                            ("exchange_depth", "exchange_depth"), ("trade_explore", "trade_explore"),
                            ("fairy_elo", "fairy_elo")):
        value = getattr(args, argument)
        if value is not None:
            setattr(cfg, field, str(value) if isinstance(value, Path) else value)
    if args.exchange_labels is not None:
        cfg.exchange_labels = bool(args.exchange_labels)
    def parse_settings(items):
        out = {}
        for item in items:
            key, _, raw = item.partition("=")
            if not hasattr(cfg, key):
                parser.error(f"unknown setting {key}")
            current = getattr(CognitiveConfig(), key)
            kind = type(current) if current is not None else float
            out[key] = (raw.lower() in ("1", "true")) if kind is bool else kind(raw)
        return out
    for key, value in parse_settings(args.settings).items():
        setattr(cfg, key, value)
    rival_settings = parse_settings(args.rival_settings)
    if args.rival_dir is not None:
        cfg.mix_self, cfg.mix_tiny, cfg.mix_fairy = 1.0, 0.0, 0.0     # a duel: every game is against the rival
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
    if args.cuda_graphs:
        from tonkai.cuda_learner import CudaUnroll
        trainer._unroll = CudaUnroll(trainer)
    if args.compile_learner:
        # Same maths, fused kernels: only last-bit float differences. Fixed batch
        # shapes, so it compiles once per process (learner and target unrolls).
        trainer._unroll = torch.compile(trainer._unroll, dynamic=False)
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
    capture_stats = CaptureStats(cfg.hot_piece_kills, opponent_weights=parse_weights(cfg.value_opponents))
    if not args.new_campaign:
        if resign is not None and state.get("resign"):
            # A state measured against another threshold (e.g. another objective)
            # would keep the old scale; start its verification again instead.
            if math.isclose(state["resign"]["initial_threshold"], resign.initial_threshold):
                resign.load_state_dict(state["resign"])
            else:
                mode = ("active at owner's request" if resign_cfg.force
                        else "verification restarts in shadow")
                print(f"[resign] threshold changed {state['resign']['initial_threshold']:.4f} -> "
                      f"{resign.initial_threshold:.4f}; {mode}", flush=True)
    # The measured piece values belong to the game, not to a campaign: a new
    # campaign keeps them (without them the exact exchange search is off).
    if state.get("capture_stats") and state["capture_stats"].get("hot_kills") == cfg.hot_piece_kills:
        capture_stats.load_state_dict(state["capture_stats"])
    values_kind = getattr(cfg, "piece_values_kind", "self")
    if values_kind != "self" and values_kind not in capture_stats.by_kind and state.get("replay"):
        # The chosen record starts from the stored real games of that kind.
        from tonkai.captures import replay_kind
        elo = values_kind.partition("@")[2]
        seeded = 0
        for row in state["replay"]["episodes"]:
            if replay_kind(row["opponent"]) == "external" and f"Elo {elo};" in row["opponent"] and row["actions"]:
                colour = episode_owner_color_row(row)
                capture_stats.add_pairs(values_kind, list(row["actions"]), colour, row["initial_fen"] or None)
                seeded += 1
        print(f"[piece values] {values_kind}: seeded from {seeded} stored games -> "
              f"{capture_stats.piece_values(values_kind)}", flush=True)
    if capture_stats.needs_seed() and state.get("replay"):
        # Piece values v2: the new measures start from the stored real games.
        seeded = capture_stats.seed_new_measures(replay_games(state["replay"]["episodes"]))
        print(f"[piece values] v2 measures seeded from {seeded} stored games", flush=True)
    saved_rival = None if args.new_campaign else state.get("self_play_rival")
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
    by_kind = defaultdict(Counter)   # opponent kind -> outcome counts, this run
    by_version = defaultdict(Counter)  # frozen rival version -> outcome counts, this run
    external_games = 0               # games against real opponents, this run
    all_plies = 0                    # every played ply, self-play included
    started = time.perf_counter()
    history = history_path.open("a", encoding="utf-8", buffering=1)
    loss = {}
    execution = dict(learner_seconds=0.0, learner_updates=0, learned_positions=0, actor_seconds=0.0)
    pool = None
    external = None
    # Owner's choice: training records no wins unless --archive-wins.
    recorder = LongestWinRecorder(cfg.win_pgn_path if args.archive_wins else None)
    # wins.json is rewritten in batches (every 2 min and at each checkpoint), not per win.
    wins_dir = (args.wins_dir or args.history_dir / "wins") if args.archive_wins else None
    archive = WinArchive(wins_dir, flush_seconds=120)
    featured = FeaturedGameRecorder(wins_dir / "featured_game.pgn" if wins_dir else None)
    # Nominal strengths for ranking the featured game; TinyOpponent measured ~800 vs Fairy.
    strength = {"external": cfg.opponent_elo, "fairy": cfg.fairy_elo, "tiny": 800}
    # Deploys are throttled: at most one running, and one every 10 minutes.
    site = {"process": None, "last": -1e9, "pending": False}

    def publish_archive(force=False):
        running = site["process"] is not None and site["process"].poll() is None
        if not site["pending"] or running or (not force and time.monotonic() - site["last"] < 600):
            return
        archive.flush()
        site.update(process=publish_wins(archive.directory), last=time.monotonic(), pending=False)

    def record(kind, **data):
        history.write(json.dumps({"kind": kind, "run_id": run_id, **data,
                                  "timestamp": datetime.now(timezone.utc).isoformat()}, allow_nan=False) + "\n")

    def metrics(final=False):
        seconds = time.perf_counter() - started
        result = dict(games=completed, final=final, state=status, objective=cfg.objective, model=model_name,
                      rules_backend=fast_rules.BACKEND, wins=counts["wins"],
                      losses=counts["losses"], draws=counts["draws"], truncated=counts["truncated"],
                      draw_reasons=dict(reasons), games_per_second=completed / max(.001, seconds),
                      positions_per_second=all_plies / max(.001, seconds), survival_plies_mean=plies / max(1, external_games),
                      train_batches=trainer.steps, cognitive=dict(diagnostics), loss=loss,
                      archive=len(trainer.knowledge.memory.entries), skills=len(trainer.knowledge.skills.skills),
                      counterfactuals=trainer.experiments, generalizations=trainer.generalizations,
                      experiment_positions=trainer.experiment_positions, captures=capture_stats.summary())
        result['execution'] = {**execution, 'learner_device': cfg.device,
            'update_seconds_mean': execution['learner_seconds'] / max(1, execution['learner_updates']),
            **{flag:getattr(cfg,flag,False) for flag in ('compact_sequences','incremental_encoder','batch_search','inference_broker','cuda_graphs')}}
        if cfg.device == 'cuda':
            result['execution']['gpu_allocated_mb'] = torch.cuda.memory_allocated() / 1048576
        if pool is not None and getattr(pool, 'broker', None) is not None:
            result['execution']['broker'] = dict(pool.broker.stats)
        if opponents.uses_mixture(cfg):
            result["opponents"] = {kind: dict(row) for kind, row in by_kind.items()}
        if rival["version"] is not None:
            result["self_play_rival"] = {"version": rival["version"], "file": Path(rival["path"]).name,
                                         "name": f"Tonkinho congelado (jogo {rival['version']})"}
            result["self_versions"] = {str(version): dict(row) for version, row in sorted(by_version.items())}
        if resign is not None:
            result["resigned"] = counts["resigned"]
            result["resign"] = resign.summary()
        record("metrics", **result)
        print(json.dumps(result), flush=True)

    # Frozen self-play rival: (version = TonkAI's game count, snapshot bytes).
    snapshot_dir = output.parent / "snapshots"
    rival = {"version": None, "payload": None, "path": None, "remaining": 0}

    def snapshot_rival():
        from tonkai.actor_snapshot import pack_snapshot
        payload = pack_snapshot(trainer.net, trainer.knowledge)
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        path = snapshot_dir / f"tonkinho-jogo-{trainer.games}.pt"
        torch.save({"format": "tonkinho-snapshot-v1", "version": trainer.games, "payload": payload}, path)
        rival.update(version=trainer.games, payload=payload, path=str(path),
                     remaining=cfg.self_games_per_snapshot)
        record("self_snapshot", version=trainer.games, file=path.name, bytes=len(payload))

    def load_rival(saved):
        path = Path(saved["path"])
        # A rival saved without its series counter predates series play: start fresh.
        if not path.is_file() or (cfg.self_games_per_snapshot and "remaining" not in saved):
            return False
        snapshot = torch.load(path, map_location="cpu", weights_only=False)
        rival.update(version=snapshot["version"], payload=snapshot["payload"], path=str(path),
                     remaining=int(saved.get("remaining", 0)))
        return True

    def save():
        try:
            archive.flush()
        except (OSError, ValueError) as error:
            record("warning", component="win_history", error=str(error))
        trainer.save(output, actor_rng=deepcopy(rng.bit_generator.state), pretrained_games=pretrained_games,
                     source_checkpoint=str(source_path), run_id=run_id, status=status,
                     campaign_start_games=campaign_start, campaign_cap=campaign_cap, opening_history=openings.state(),
                     campaign_results=dict(counts), campaign_draw_reasons=dict(reasons),
                     capture_stats=capture_stats.state_dict(),
                     resign=resign.state_dict() if resign is not None else None,
                     model_name=model_name, model_version=model_version,
                     self_play_rival=({"version": rival["version"], "path": rival["path"],
                                       "remaining": rival["remaining"]}
                                      if rival["version"] is not None else None))

    try:
        external = UCIOpponent(cfg.opponent_path, cfg.opponent_elo, cfg.opponent_move_time)
        launch = dict(pid=os.getpid(), run_id=run_id, resume_games=initial_games, additional_games=args.games,
                      rules_backend=fast_rules.BACKEND,
                      pretrained_games=pretrained_games, objective=cfg.objective, version="cognitive-v3",
                      model_name=model_name, model_version=model_version,
                      king_safety_model=cfg.king_safety_model or None,
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
        if opponents.uses_mixture(cfg):
            print("Opponent mixture: " + ", ".join(f"{kind} {share:.0%}" for kind, share
                                                    in opponents.mixture(cfg).items() if share > 0), flush=True)
        if resign is not None:
            print(resign.line(), flush=True)
        if args.dedicate_learner_core:
            from tonkai.cognitive_pool import LEARNER_MASK, pin_process
            # The main-process engine (experiments) was started above and keeps
            # the default mask; pin the learner only now, before the actors spawn.
            print(f"[cpu] learner pinned to core 0: {pin_process(LEARNER_MASK, above_normal=True)}", flush=True)
        if cfg.self_snapshot_every and not (saved_rival and load_rival(saved_rival)):
            snapshot_rival()
        if rival['version'] is not None:
            print(f"Self-play rival: Tonkinho congelado (jogo {rival['version']}), novo a cada "
                  f"{cfg.self_snapshot_every} jogos", flush=True)
        other = {"version": None, "payload": None, "file": None}

        def refresh_other():
            """Duel: the newest snapshot the other campaign has written."""
            files = sorted(args.rival_dir.glob("tonkinho-jogo-*.pt"), key=lambda p: int(p.stem.rsplit("-", 1)[1]))
            if files and files[-1].name != other["file"]:
                snapshot = torch.load(files[-1], map_location="cpu", weights_only=False)
                other.update(version=snapshot["version"], payload=snapshot["payload"], file=files[-1].name)
                record("duel_rival", version=other["version"], file=other["file"])
                print(f"[duel] rival = {args.rival_name or 'rival'} (jogo {other['version']})", flush=True)

        if args.rival_dir is not None:
            for _ in range(600):                              # the other campaign may still be starting
                refresh_other()
                if other["payload"] is not None:
                    break
                time.sleep(1)
            if other["payload"] is None:
                raise RuntimeError(f"no rival snapshot in {args.rival_dir}")
        pool = (CognitivePool(cfg, cfg.opponent_path, dedicated=True) if args.dedicate_learner_core
                else CognitivePool(cfg, cfg.opponent_path))
        while completed < args.games:
            stopping = stopping or stop_file.exists()
            while len(pool.pending) < pool.capacity and submitted < args.games and not stopping:
                seed, game_number = rng.integers(0, 2**31 - 1), campaign_done + submitted + 1
                extras = {}
                if resign is not None:
                    extras["resign_plan"] = resign.plan()
                # Always measured values: kd_weight only sets their share in the blend,
                # it must not switch off the exact exchange search.
                kd_values = (capture_stats.piece_values(getattr(cfg, "piece_values_kind", "self"))
                             or capture_stats.piece_values())
                if kd_values is not None:
                    extras["kd_values"] = kd_values
                if args.rival_dir is not None:
                    if submitted % 25 == 0:
                        refresh_other()
                    extras["kind"] = "self"
                    extras["rival"] = (other["version"], other["payload"], rival_settings,
                                       f"{args.rival_name or 'Rival'} (jogo {other['version']})")
                elif rival["remaining"] > 0 and rival["payload"] is not None:
                    extras["kind"] = "self"          # the series against the newest frozen TonkAI
                    rival["remaining"] -= 1
                elif opponents.uses_mixture(cfg):
                    extras["kind"] = opponents.choose_kind(cfg, rng)
                if "kind" in extras and "rival" not in extras:
                    if extras["kind"] == "self" and rival["payload"] is not None:
                        extras["rival"] = (rival["version"], rival["payload"])
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
            diagnostics.update(stats["cognitive"])
            all_plies += stats["plies"]
            execution["actor_seconds"] += stats.get("actor_seconds", 0.0)
            resign_report = stats.get("resign")
            resigned = bool(resign_report and resign_report["resigned"])
            kind = stats.get("opponent_kind", "external")
            # Results and survival are against real opponents; self-play is
            # reported per opponent kind only.
            if kind == "self":
                pass
            elif not episode.terminated:
                counts["truncated"] += 1
            elif episode.lost:
                counts["losses"] += 1
                if resigned:
                    counts["resigned"] += 1
            elif stats["result"] == "1/2-1/2":
                counts["draws"] += 1
                reasons["tablebase" if stats.get("tablebase") else
                        board.outcome(claim_draw=True).termination.name.lower()] += 1
            else:
                counts["wins"] += 1
            if kind != "self":
                plies += stats["plies"]
                external_games += 1
            if resigned:
                outcome = "resigned"
            elif episode.terminated:
                outcome = ("loss" if episode.lost else "draw" if stats["result"] == "1/2-1/2" else "win")
            else:
                outcome = "unfinished"
            capture_stats.add(board.move_stack, colour, outcome, opponent_kind=kind,
                              detail_kind=f"external@{cfg.opponent_elo}" if kind == "external" else None)
            by_kind[kind][outcome] += 1
            if kind == "self" and "opponent_version" in stats:
                by_version[stats["opponent_version"]][outcome] += 1
            # v4: how the learned KS of both kings moved through the game.
            if cfg.king_safety_model and (completed % max(1, cfg.king_safety_log_every) == 0
                                          or (outcome == "win" and kind not in ("self", "tiny"))):
                # Only the logged games pay for the KS series, rebuilt here in one batch.
                series = stats.get("king_safety") or load_king_safety(cfg.king_safety_model).series(board.move_stack)
                record("king_safety", game=completed, total_games=trainer.games, opponent=stats["opponent"],
                       opponent_kind=kind, tonkai="white" if colour else "black", result=stats["result"],
                       outcome=outcome, series=series)
            if resign is not None:
                for message in resign.update(resign_report, lost=episode.lost):
                    print(message, flush=True)
                    record("resign", game=completed, message=message, **resign.summary())
            # A win over itself is no win, and TinyOpponent is only a sparring partner:
            # only real engines enter the win history and the site.
            archived = args.archive_wins and kind not in ("self", "tiny")
            if episode.terminated and not resigned and not stats.get("stalled") and archived:
                try:
                    termination = ("tablebase" if stats.get("tablebase") else
                                   board.outcome(claim_draw=True).termination.name.lower())
                    recorder.consider(board, colour, stats["opponent"], stats["result"], termination)
                    if archive.add(board, colour, stats["opponent"], stats["result"], termination,
                                   run_id=run_id, game=trainer.games, objective=cfg.objective):
                        site["pending"] = True
                        record("win", game=completed, total_games=trainer.games, plies=len(board.move_stack))
                except (OSError, ValueError) as error:
                    record("warning", component="win_history", error=str(error))
                publish_archive()
            # The site's single history game: longest vs the strongest opponent, any result.
            if episode.terminated and archived:
                try:
                    termination = ("resignation" if resigned else "tablebase" if stats.get("tablebase") else
                                   board.outcome(claim_draw=True).termination.name.lower())
                    elo = strength.get(kind, 0)
                    name = f"{stats['opponent']} · {elo} Elo" if kind != "tiny" else stats["opponent"]
                    if featured.consider(board, colour, name, elo, stats["result"], termination):
                        site["pending"] = True
                        record("featured", game=completed, total_games=trainer.games, opponent=name, elo=elo,
                               plies=len(board.move_stack), result=stats["result"])
                except (OSError, ValueError, AttributeError) as error:
                    record("warning", component="featured_game", error=str(error))
                publish_archive()
            # The separate engine only plays legal experimental continuations; its
            # evaluations never become labels and branches never count as real games.
            if not stopping:
                for event in trainer.investigate(episode, trace, colour, external):
                    record("experiment", experiment_type=event["kind"], **{key: value for key, value in event.items() if key != "kind"})
            for _ in range(cfg.updates_per_game):
                update_started = time.perf_counter()
                update = trainer.update()
                execution['learner_seconds'] += time.perf_counter() - update_started
                if update:
                    loss = update
                    execution['learner_updates'] += 1
                    execution['learned_positions'] += update['learned_positions']
            if cfg.self_snapshot_every and trainer.games - rival["version"] >= cfg.self_snapshot_every:
                snapshot_rival()        # games already in flight keep the rival they were given
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
