"""Paired deterministic learning runs. Reads one checkpoint; never saves/publishes weights."""
import argparse
from dataclasses import fields, is_dataclass
import hashlib
from io import BytesIO
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import tiny_opponent
from tonkai import fast_rules
from tonkai.captures import CaptureStats
from tonkai.cognitive_agent import CognitiveAgent
from tonkai.cognitive_config import CognitiveConfig
from tonkai.cognitive_trainer import CognitiveTrainer
from tonkai.survival_agent import play_episode


class Opponent:
    description = "deterministic TinyOpponent benchmark (not Elo-calibrated)"

    def __init__(self):
        self.engine = tiny_opponent.TinyOpponent(depth=2, max_nodes=500, randomness=.35, seed=71)

    def reset(self):
        self.engine.set_seed(71)

    def choose(self, board, rng):
        import chess
        return chess.Move.from_uci(self.engine.choose_move(board.fen()))


def exact(a, b, path="state"):
    if isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor) and torch.equal(a, b), path
    elif isinstance(a, np.ndarray):
        assert isinstance(b, np.ndarray) and a.dtype == b.dtype and np.array_equal(a, b), path
    elif isinstance(a, dict):
        assert a.keys() == b.keys(), path
        for k in a:
            exact(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, (tuple, list)):
        assert type(a) is type(b) and len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            exact(x, y, f"{path}[{i}]")
    elif is_dataclass(a):
        for field in fields(a):
            exact(getattr(a, field.name), getattr(b, field.name), f"{path}.{field.name}")
    else:
        assert a == b, path


def trial(blob, native, games):
    # Include the learner's native encoding/targets, not only actor rules.
    fast_rules.ENABLED = native
    state = torch.load(BytesIO(blob), map_location="cpu", weights_only=False)
    cfg = CognitiveConfig.from_dict(state["config"])
    cfg.device = "cpu"
    trainer = CognitiveTrainer(cfg)
    trainer.load_state_dict(state)
    captures = CaptureStats()
    if "capture_stats" in state:
        captures.load_state_dict(state["capture_stats"])
    opponent, rng = Opponent(), np.random.default_rng(20260923)
    played, updates, traces, decisions, experiments = [], [], [], [], []
    started = time.perf_counter()
    for i in range(games):
        agent = CognitiveAgent(trainer.net, cfg, trainer.knowledge, kd_values=captures.piece_values())
        agent.use_fast_rules = native
        choose = agent.choose
        def audited(*args, **kwargs):
            choice = choose(*args, **kwargs)
            decisions.append((choice.scores.copy(), choice.index))
            return choice
        agent.choose = audited
        episode, board, colour, stats = play_episode(agent, opponent, cfg, rng, opponent_name=opponent.description)
        played.append(episode)
        traces.append(agent.trace)
        trainer.ingest(episode, agent.trace, colour)
        outcome = "unfinished" if not episode.terminated else "loss" if episode.lost else "draw" if stats["result"] == "1/2-1/2" else "win"
        captures.add(board.move_stack, colour, outcome)
        experiments.extend(trainer.investigate(episode, agent.trace, colour, opponent))
        for _ in range(cfg.updates_per_game):
            updates.append(trainer.update())
    seconds = time.perf_counter() - started
    plies = sum(len(e.actions) for e in played)
    report = dict(backend="cpp" if native else "python", games=games, seconds=seconds,
                  games_per_second=games/seconds, plies=plies, plies_per_second=plies/seconds,
                  updates=len(updates), decisions=len(decisions), experiments=len(experiments))
    evidence = (trainer.state_dict(), captures.state_dict(), rng.bit_generator.state,
                played, updates, traces, decisions, experiments)
    return report, evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--games", type=int, default=6)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not fast_rules.AVAILABLE or args.games < 1:
        parser.error("Build tonkai_fast first and use a positive game count")
    if args.output.resolve() == args.checkpoint.resolve():
        parser.error("Output must not overwrite the checkpoint")
    torch.set_num_threads(1)
    blob = args.checkpoint.read_bytes()
    digest = hashlib.sha256(blob).hexdigest()
    # Warm both routes before timing; every trial resets the entire learner/RNG.
    trial(blob, False, 1)
    trial(blob, True, 1)
    reports, reference = [], None
    for native in (False, True, True, False):
        report, evidence = trial(blob, native, args.games)
        if reference is None:
            reference = evidence
        else:
            exact(reference, evidence)
        reports.append(report)
        print(json.dumps(report), flush=True)
    slow = np.mean([r["games_per_second"] for r in reports if r["backend"] == "python"])
    fast = np.mean([r["games_per_second"] for r in reports if r["backend"] == "cpp"])
    result = dict(checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=digest,
                  runs=reports, exact_match=True, speedup=float(fast/slow),
                  checkpoint_unchanged=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()==digest,
                  conditions="One CPU actor; deterministic TinyOpponent; updates and scheduled experiments; no resignation; no production writes. Not a Rodent throughput/Elo estimate.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k != "runs"}), flush=True)


if __name__ == "__main__":
    main()
